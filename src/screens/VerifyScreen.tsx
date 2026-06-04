// Orchestrates the real-time face verification pipeline.
// Processes frames via VisionCamera, executing BlazeFace detection and MobileFaceNet
// embedding generation continuously to compare against enrolled identity vectors.

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  ActivityIndicator,
  Animated,
  Dimensions,
  Platform,
  StatusBar,
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
} from 'react-native';
import Reanimated, { FadeInDown, FadeInUp, FadeOutUp, runOnUI } from 'react-native-reanimated';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
} from 'react-native-vision-camera';
import { scheduleOnRN } from 'react-native-worklets';

import { useFaceDetector }                                      from '../ml/useFaceDetector';
import { useEmbedder }                                          from '../ml/useEmbedder';
import { decodeFaces, type FaceBox }                             from '../ml/blazeface';
import { resizeToFloat32, cropFaceToUint8, uint8ToFloat32Normalized, typedArrayToBuffer } from '../ml/preprocessing';
import { applyCLAHE }                                                from '../ml/clahe';
import { bestMatch, l2Normalize }                                 from '../utils/mathUtils';
import { getAllFaces, type EnrolledFace }                         from '../storage/faceStore';
import { generateLivenessSequence, getStepPrompt, checkSequenceTask } from '../liveness/ActiveLivenessSequence';
import { FaceOverlay }                                            from '../components/FaceOverlay';


const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

/**
 * Minimum cosine similarity to accept a match.
 * MobileFaceNet embeddings for the same person typically cluster in [0.65, 0.95];
 * different people generally fall below 0.50.
 * Raise toward 0.65 for stricter security.
 */
const VERIFY_THRESHOLD       = 0.70;

/** Target processing rate. 200 ms ≈ 5 FPS balances latency and CPU load. */
const PROCESS_INTERVAL_MS    = 200;

/** Rolling window size for smoothing displayed confidence (reduces flicker). */
const SMOOTH_WINDOW   = 5;

const DETECT_SIZE     = 128;
const EMBED_SIZE      = 160;
const FACE_CROP_PADDING = 0.2;


interface VerifyResult {
  box:          FaceBox | null;
  name:         string | null;
  confidence:   number;
  fps:          number;
  imageWidth:   number;
  imageHeight:  number;
  orientation:  string;
  livenessPrompt: string | null;
  debugMesh?: {x: number, y: number}[];
}

const DEFAULT_RESULT: VerifyResult = {
  box: null, name: null, confidence: 0, fps: 0,
  imageWidth: 0, imageHeight: 0, orientation: 'portrait', livenessPrompt: null,
};


export const VerifyScreen: React.FC = () => {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [cameraPos, setCameraPos] = useState<'front' | 'back'>('front');
  const device = useCameraDevice(cameraPos);

  const { model: detectModel, anchors, isLoading: detectLoading, error: detectError } = useFaceDetector();
  const { model: embedModel,           isLoading: embedLoading,  error: embedError  } = useEmbedder();

  const [result,        setResult]        = useState<VerifyResult>(DEFAULT_RESULT);
  const [enrolledFaces, setEnrolledFaces] = useState<EnrolledFace[]>([]);
  const [cameraLayout,  setCameraLayout]  = useState({ width: SCREEN_W, height: SCREEN_H });
  const confAnim = useRef(new Animated.Value(0)).current;

  // We cannot use a mutable ref for data sent to the worklet because worklets freeze
  // objects passed to them. We derive a plain array from the state and pass it down.
  const onResultRef    = useRef<(r: VerifyResult) => void>(() => {});

  // Worklet-side perf counters are tracked via globalThis in the worklet thread
  // to avoid mutating frozen ref objects.
  const confidenceHistory = useRef<number[]>([]);

  // Processes incoming verification results, applying a rolling average
  // to confidence scores to reduce UI flicker.

  const onResult = useCallback((r: VerifyResult) => {
    if (r.confidence > 0) {
      confidenceHistory.current.push(r.confidence);
      if (confidenceHistory.current.length > SMOOTH_WINDOW) {
        confidenceHistory.current.shift();
      }
      const smooth = confidenceHistory.current.reduce((a, b) => a + b, 0) / confidenceHistory.current.length;
      setResult({ ...r, confidence: smooth });
    } else {
      confidenceHistory.current = [];
      setResult(r);
    }
  }, []);

  useEffect(() => { onResultRef.current = onResult; }, [onResult]);

  // Stable bridge called from worklet — the ref indirection ensures the worklet
  // always calls the current handler even as the component re-renders.
  const onResultBridge = useCallback((r: VerifyResult) => {
    onResultRef.current(r);
  }, []);

  // Synchronizes the local enrolled faces state with MMKV storage upon mount.

  useEffect(() => {
    const load = () => {
      const faces = getAllFaces();
      setEnrolledFaces(faces);
    };
    load();
  }, []);

  const enrolledShared = useMemo(() => {
    // number[] arrays for serialisation into the worklet
    return enrolledFaces.map(f => ({
      id:       f.id,
      name:     f.name,
      embArray: Array.from(f.embedding),
    }));
  }, [enrolledFaces]);

  // ─── Confidence bar animation ─────────────────────────────────────────────

  useEffect(() => {
    Animated.timing(confAnim, {
      toValue:         result.confidence,
      duration:        180,
      useNativeDriver: false,
    }).start();
  }, [result.confidence, confAnim]);

  // ─── Permission ───────────────────────────────────────────────────────────

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  // ─── Frame pipeline ───────────────────────────────────────────────────────

  const onFrame = useCallback((frame: any) => {
    'worklet';
    const globalObj = globalThis as any;

    if (!detectModel || !embedModel || !anchors) {
      frame.dispose();
      return;
    }

    const now = Date.now();

    if (globalObj._successCooldown && now < globalObj._successCooldown) {
      frame.dispose();
      return;
    }

    // Gate to PROCESS_INTERVAL_MS
    const lastProc = globalObj._lastProcessed || 0;
    if (now - lastProc < PROCESS_INTERVAL_MS) {
      frame.dispose();
      return;
    }
    globalObj._lastProcessed = now;

    globalObj._frameCount = (globalObj._frameCount || 0) + 1;
    const lastFpsTime = globalObj._lastFpsTime || now;
    const elapsed = now - lastFpsTime;
    if (elapsed >= 1000) {
      globalObj._fpsValue    = Math.round((globalObj._frameCount * 1000) / elapsed);
      globalObj._frameCount  = 0;
      globalObj._lastFpsTime = now;
    }
    const currentFps = globalObj._fpsValue || 0;

      try {
        const pixels    = new Uint8Array(frame.getPixelBuffer());
        const orient    = frame.orientation;
        const isWrongDim = frame.width > frame.height;
        const frameW    = isWrongDim ? frame.height : frame.width;
        const frameH    = isWrongDim ? frame.width  : frame.height;
        const fmt       = frame.pixelFormat;
        const bpp       = (fmt as string) === 'rgb' ? 3 : 4;
        const rowBytes  = frameW * bpp;

        // ── Step 1: BlazeFace detection ────────────────────────────────
        const tDetectStart = Date.now();

        let topFace: FaceBox | null = null;

        if (!globalObj._detectInput) globalObj._detectInput = new Float32Array(DETECT_SIZE * DETECT_SIZE * 3);
        const detectInput = globalObj._detectInput;

        try {
          resizeToFloat32(pixels, frameW, frameH, DETECT_SIZE, DETECT_SIZE, rowBytes, fmt, detectInput);
          const detOut       = detectModel.runSync([typedArrayToBuffer(detectInput)]);
          const regressors   = new Float32Array(detOut[0]);
          const scores       = new Float32Array(detOut[1]);
          const faces        = decodeFaces(regressors, scores, anchors);
          topFace = faces.length > 0 ? faces[0] : null;
        } catch (e: any) {
          console.log('[VerifyScreen] Detection error:', e?.message || e);
          scheduleOnRN(onResultBridge, {
            ...DEFAULT_RESULT, fps: currentFps, imageWidth: frameW, imageHeight: frameH, orientation: orient,
          } as VerifyResult);
          return;
        }
        const tDetectEnd = Date.now();

        if (!topFace) {
          if (globalObj._verifySeqState?.passed) {
            const expiry = globalObj._sessionExpiry || 0;
            scheduleOnRN(onResultBridge, {
              box: null, name: null, confidence: 0,
              fps: currentFps, imageWidth: frameW, imageHeight: frameH, orientation: orient,
              livenessPrompt: null,
            } as any);
            if (now > expiry) {
              globalObj._verifySeqState = null;
              globalObj._sessionExpiry  = null;
            }
          } else {
            scheduleOnRN(onResultBridge, {
              box: null, name: null, confidence: 0,
              fps: currentFps, imageWidth: frameW, imageHeight: frameH, orientation: orient,
              livenessPrompt: null,
            } as VerifyResult);
          }
          return;
        } else {
          globalObj._lastFaceSeen = now;
          if (globalObj._verifySeqState?.passed) {
            globalObj._sessionExpiry = now + 3000;
          }
        }

        // ── Liveness Check ────────────────────────────────────
        let seqState = globalObj._verifySeqState;
        if (!seqState) {
          seqState = generateLivenessSequence();
          globalObj._verifySeqState = seqState;
        }

        let livenessPassed = seqState.passed;

        // Extract landmark dots for the liveness overlay (before checking)
        let debugMesh: {x: number, y: number}[] | undefined = undefined;
        if (!livenessPassed && topFace.landmarks) {
          const lm = topFace.landmarks as Float32Array;
          const pts: {x: number, y: number}[] = [];
          for (let i = 0; i < lm.length / 2; i++) {
            pts.push({ x: lm[i * 2], y: lm[i * 2 + 1] });
          }
          debugMesh = pts;
        }

        if (!livenessPassed) {
          if (topFace.landmarks) {
            const taskPassed = checkSequenceTask(topFace.landmarks as any, seqState);
            if (taskPassed) {
              seqState.currentTaskIndex++;
              seqState.framesInCurrentTask = 0;
              if (seqState.currentTaskIndex >= seqState.tasks.length) {
                seqState.passed = true;
                livenessPassed  = true;
                // Start session countdown
                globalObj._sessionExpiry = now + 3000;
              }
            }
          }
        }

        const livenessPrompt = livenessPassed ? null : getStepPrompt(seqState);

        if (!livenessPassed) {
          scheduleOnRN(onResultBridge, {
            box: topFace, name: null, confidence: 0,
            fps: currentFps, imageWidth: frameW, imageHeight: frameH, orientation: orient,
            livenessPrompt: livenessPrompt,
            debugMesh,
          } as VerifyResult);
          return;
        }

        // ── Step 2: MobileFaceNet embedding (with CLAHE) ─────────────────
        const tEmbedStart = Date.now();

        let embedding: any = new Float32Array(0);
        try {
          // Allocate reusable worklet-thread buffers on first use
          if (!globalObj._cropU8)    globalObj._cropU8    = new Uint8Array(EMBED_SIZE * EMBED_SIZE * 3);
          if (!globalObj._embedInput) globalObj._embedInput = new Float32Array(EMBED_SIZE * EMBED_SIZE * 3);

          const cropU8     = globalObj._cropU8     as Uint8Array;
          const embedInput = globalObj._embedInput as Float32Array;

          // 2a. Crop face into Uint8Array RGB buffer
          cropFaceToUint8(pixels, frameW, frameH, topFace, EMBED_SIZE, EMBED_SIZE, rowBytes, fmt, cropU8, FACE_CROP_PADDING);

          // 2b. CLAHE adaptive contrast enhancement (in-place on cropU8)
          applyCLAHE(cropU8, EMBED_SIZE, EMBED_SIZE, 3, 0, 1, 2, cropU8);

          // 2c. Normalise Uint8 → Float32 [-1, 1] for MobileFaceNet
          uint8ToFloat32Normalized(cropU8, embedInput);

          const embOut     = embedModel.runSync([typedArrayToBuffer(embedInput)]);
          embedding        = l2Normalize(new Float32Array(embOut[0] as ArrayBuffer));
        } catch (e: any) {
          console.log('[VerifyScreen] Embed error:', e?.message || e);
          scheduleOnRN(onResultBridge, {
            box: topFace, name: null, confidence: 0,
            fps: currentFps, imageWidth: frameW, imageHeight: frameH, orientation: orient,
            livenessPrompt: 'Error extracting embedding',
          } as VerifyResult);
          return;
        }
        const tEmbedEnd = Date.now();

        // ── Step 3: Match against enrolled faces ─────────────────────────
        const tMatchStart = Date.now();
        const enrolled = enrolledShared.map(f => ({
          id:        f.id,
          name:      f.name,
          embedding: new Float32Array(f.embArray),
        }));

        const match = bestMatch(embedding, enrolled, VERIFY_THRESHOLD);
        const tMatchEnd = Date.now();

        // ── Performance log ──────────────────────────────────────────────
        const totalMs = tMatchEnd - tDetectStart;
        // Log every ~5th frame to avoid spamming console
        globalObj._perfLogCounter = (globalObj._perfLogCounter || 0) + 1;
        if (globalObj._perfLogCounter % 5 === 0) {
          console.log(
            `[PERF] detect: ${tDetectEnd - tDetectStart}ms | embed: ${tEmbedEnd - tEmbedStart}ms | match: ${tMatchEnd - tMatchStart}ms | total: ${totalMs}ms`,
          );
        }

        // CRITICAL: send livenessPrompt: null so livenessActive becomes false
        // and the result panel shows the name/confidence instead of liveness overlay.
        scheduleOnRN(onResultBridge, {
          box:        topFace,
          name:       match?.name ?? null,
          confidence: match?.similarity ?? 0,
          fps: currentFps, imageWidth: frameW, imageHeight: frameH, orientation: orient,
          livenessPrompt: null,
        } as VerifyResult);

        // On a successful match: reset liveness and start cooldown so the next
        // attempt requires fresh liveness verification.
        if (match?.name) {
          globalObj._verifySeqState  = null;
          globalObj._sessionExpiry   = null;
          globalObj._successCooldown = now + 3000;
        }

      } finally {
        frame.dispose();
      }
  }, [detectModel, embedModel, anchors, enrolledShared, onResultBridge]);

  const frameOutput = useFrameOutput(useMemo(() => ({
    pixelFormat: 'rgb',
    enablePhysicalBufferRotation: true,
    onFrame,
  }), [onFrame]));

  // Memoize outputs array — prevents Camera from restarting the pipeline on
  // every render, which causes intermittent frame drops.
  const cameraOutputs = useMemo(() => [frameOutput], [frameOutput]);

  // ─── Derived ──────────────────────────────────────────────────────────────

  const isLoading   = detectLoading || embedLoading;
  const loadError   = detectError   || embedError;

  const livenessActive = !!(result.livenessPrompt && !result.name);
  // Suppress UNKNOWN label during liveness phase
  const overlayStatus =
    result.name ? 'matched' :
    (result.box && !livenessActive) ? 'unknown' :
    'scanning';

  // Only show the UNKNOWN label after liveness is verified
  const overlayLabel =
    result.name ? `${result.name}  ${(result.confidence * 100).toFixed(1)}%` :
    (result.box && !livenessActive) ? 'UNKNOWN' :
    undefined;

  // ─── Actions ─────────────────────────────────────────────────────────────────

  const handleRetryLiveness = useCallback(() => {
    runOnUI(() => {
      'worklet';
      (globalThis as any)._verifySeqState  = null;
      (globalThis as any)._sessionExpiry   = null;
      (globalThis as any)._successCooldown = null;
    })();
  }, []);

  // ─── Render guards ────────────────────────────────────────────────────────

  if (!hasPermission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.infoText}>Camera permission required</Text>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.centered}>
        <Text style={styles.infoText}>No front camera found</Text>
      </View>
    );
  }

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="#0a0c0f" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Verification</Text>
        <TouchableOpacity style={styles.flipBtn} onPress={() => setCameraPos(p => p === 'front' ? 'back' : 'front')}>
          <Text style={styles.flipBtnText}>FLIP</Text>
        </TouchableOpacity>
        <View style={styles.headerRight}>
          <View style={[styles.dot, result.box ? styles.dotActive : styles.dotIdle]} />
          <Text style={styles.fpsText}>{result.fps > 0 ? `${result.fps} FPS` : '— FPS'}</Text>
        </View>
      </View>

      {/* Camera viewport */}
      <View style={styles.cameraWrap} onLayout={(e) => setCameraLayout(e.nativeEvent.layout)}>
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          isActive
          outputs={cameraOutputs}
          resizeMode="contain"
        />

        {isLoading && (
          <View style={styles.overlay}>
            <ActivityIndicator size="large" color="#f0a500" />
            <Text style={styles.overlayText}>Initialising AI models…</Text>
            <Text style={styles.overlaySubText}>This takes ~5 s on first launch</Text>
          </View>
        )}

        {!!loadError && !isLoading && (
          <View style={styles.overlay}>
            <Text style={styles.errorText}>{loadError}</Text>
          </View>
        )}

        {!isLoading && result.box && (
          <FaceOverlay
            box={result.box}
            label={overlayLabel}
            status={overlayStatus}
            frameWidth={cameraLayout.width}
            frameHeight={cameraLayout.height}
            imageWidth={result.imageWidth}
            imageHeight={result.imageHeight}
            imageOrientation={result.orientation}
            debugMesh={result.debugMesh}
            livenessMode={livenessActive}
          />
        )}
        
        {/* Liveness Prompt Overlay */}
        {result.livenessPrompt && !result.name && (
          <Reanimated.View 
            entering={FadeInUp.springify().damping(20).stiffness(200)} 
            exiting={FadeOutUp} 
            style={styles.livenessOverlay}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 12 }}>
              <Text style={styles.livenessText}>{result.livenessPrompt}</Text>
              <TouchableOpacity onPress={handleRetryLiveness} style={styles.livenessRetryBtn}>
                <Text style={styles.livenessRetryText}>Retry</Text>
              </TouchableOpacity>
            </View>
          </Reanimated.View>
        )}

        {/* Result panel */}
        <Reanimated.View 
          entering={FadeInDown.springify().damping(20).stiffness(200)}
          style={styles.resultPanel}
        >
          {result.name ? (
            <View>
              <View style={styles.matchRow}>
                <View style={styles.matchLeft}>
                  <View style={styles.matchDot} />
                  <Text style={styles.matchName} numberOfLines={1}>{result.name}</Text>
                </View>
                <Text style={styles.confPercent}>{(result.confidence * 100).toFixed(1)}%</Text>
              </View>
              <View style={styles.confBarTrack}>
                <Animated.View
                  style={[
                    styles.confBarFill,
                    {
                      width: confAnim.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] }),
                      backgroundColor: confAnim.interpolate({
                        inputRange:  [VERIFY_THRESHOLD, 0.85, 1],
                        outputRange: ['#FF9500', '#34C759', '#34C759'],
                      }),
                    },
                  ]}
                />
              </View>
            </View>
          ) : result.box && !livenessActive ? (
            <View style={styles.unknownRow}>
              <Text style={styles.unknownLabel}>UNKNOWN PERSON</Text>
              <Text style={styles.unknownSub}>
                No match above {Math.round(VERIFY_THRESHOLD * 100)}% threshold
              </Text>
            </View>
          ) : livenessActive ? (
            <Text style={styles.scanningLabel}>Complete liveness check above…</Text>
          ) : (
            <Text style={styles.scanningLabel}>
              {isLoading
                ? 'Loading…'
                : enrolledFaces.length === 0
                  ? 'No faces enrolled — open the Enroll tab to add faces'
                  : 'Scanning for face…'}
            </Text>
          )}
        </Reanimated.View>
      </View>

      {/* Footer */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          {enrolledFaces.length} face{enrolledFaces.length !== 1 ? 's' : ''} enrolled
        </Text>
        <Text style={styles.footerText}>
          Threshold  {Math.round(VERIFY_THRESHOLD * 100)}%
        </Text>
      </View>
    </View>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  root:    { flex: 1, backgroundColor: '#0a0c0f' },
  centered: {
    flex: 1, backgroundColor: '#0a0c0f',
    alignItems: 'center', justifyContent: 'center',
  },
  infoText: { color: '#9da3b4', fontSize: 15 },

  header: {
    flexDirection:     'row',
    justifyContent:    'space-between',
    alignItems:        'center',
    paddingTop:        Platform.OS === 'ios' ? 56 : 20,
    paddingHorizontal: 20,
    paddingBottom:     14,
    backgroundColor:   '#0d1117',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#1e2433',
  },
  headerTitle: {
    color: '#e8eaf0',
    fontSize: 18,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
  flipBtn: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#1f2430',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#303846',
  },
  flipBtnText: {
    color: '#e8eaf0',
    fontSize: 12,
  },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 7 },
  dot:         { width: 9, height: 9, borderRadius: 4.5 },
  dotIdle:     { backgroundColor: '#2d3348' },
  dotActive:   { backgroundColor: '#4ade80' },
  fpsText:     { color: '#4a5568', fontSize: 12, fontWeight: '700', letterSpacing: 0.5 },

  cameraWrap: { flex: 1, backgroundColor: '#000', overflow: 'hidden' },

  overlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(10,12,15,0.90)',
    alignItems:      'center',
    justifyContent:  'center',
    gap: 12,
  },
  overlayText:    { color: '#e8eaf0', fontSize: 16, fontWeight: '600' },
  overlaySubText: { color: '#4a5568', fontSize: 13 },
  errorText:      { color: '#ff4d4f', fontSize: 14, textAlign: 'center', padding: 24 },

  resultPanel: {
    position:          'absolute',
    bottom:            30,
    left:              20,
    right:             20,
    backgroundColor:   'rgba(28,28,30,0.85)',
    borderRadius:      24,
    paddingHorizontal: 24,
    paddingVertical:   20,
    minHeight:         80,
    justifyContent:    'center',
    shadowColor:       '#000',
    shadowOffset:      { width: 0, height: 10 },
    shadowOpacity:     0.4,
    shadowRadius:      16,
    elevation:         10,
  },

  matchRow:   { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  matchLeft:  { flexDirection: 'row', alignItems: 'center', flex: 1, gap: 12 },
  matchDot:   { width: 12, height: 12, borderRadius: 6, backgroundColor: '#34C759' },
  matchName:  { color: '#FFFFFF', fontSize: 22, fontWeight: '800', letterSpacing: 0.3, flexShrink: 1 },
  confPercent: { color: '#34C759', fontSize: 20, fontWeight: '800', marginLeft: 12 },
  confBarTrack: { height: 6, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 3, overflow: 'hidden' },
  confBarFill:  { height: '100%', borderRadius: 3 },

  unknownRow:   { gap: 6 },
  unknownLabel: { color: '#FF3B30', fontSize: 18, fontWeight: '800', letterSpacing: 0.5 },
  unknownSub:   { color: '#8E8E93', fontSize: 13, fontWeight: '500' },

  scanningLabel: { color: '#8E8E93', fontSize: 15, textAlign: 'center', fontWeight: '600', letterSpacing: 0.3 },

  livenessOverlay: {
    position: 'absolute',
    top: 40,
    alignSelf: 'center',
    backgroundColor: 'rgba(28,28,30,0.85)',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 30,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 8,
  },
  livenessText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.4,
  },
  livenessRetryBtn: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  livenessRetryText: {
    color: '#FFFFFF',
    fontSize: 13,
    fontWeight: '700',
  },

  footer: {
    flexDirection:     'row',
    justifyContent:    'space-between',
    alignItems:        'center',
    paddingHorizontal: 20,
    paddingVertical:   9,
    backgroundColor:   '#0d1117',
    borderTopWidth:    StyleSheet.hairlineWidth,
    borderTopColor:    '#1e2433',
  },
  footerText: { color: '#3d4561', fontSize: 11, fontWeight: '700', letterSpacing: 0.5 },
});