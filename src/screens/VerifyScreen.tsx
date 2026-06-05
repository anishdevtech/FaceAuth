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
import Reanimated from 'react-native-reanimated';
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
import { resizeToFloat32, alignFaceToUint8, uint8ToFloat32Normalized, typedArrayToBuffer } from '../ml/preprocessing';
import { applyCLAHE }                                                from '../ml/clahe';
import { bestMatch, l2Normalize }                                 from '../utils/mathUtils';
import { getAllFaces, type EnrolledFace }                         from '../storage/faceStore';
import { logAuthEvent }                                           from '../storage/authSync';
import { generateLivenessSequence, checkSequenceTask } from '../liveness/ActiveLivenessSequence';
import { FaceOverlay }                                            from '../components/FaceOverlay';
import { useTranslation } from '../i18n';
import { Settings } from 'lucide-react-native';
import { LanguagePickerModal } from '../components/LanguagePickerModal';

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

/**
 * Minimum cosine similarity to accept a match.
 * MobileFaceNet embeddings for the same person typically cluster in [0.65, 0.95];
 * different people generally fall below 0.50.
 * Raise toward 0.65 for stricter security.
 */
const VERIFY_THRESHOLD       = 0.70;

/** Target processing rate. 200 ms ≈ 5 FPS balances latency and CPU load. */
const PROCESS_INTERVAL_MS    = 350;

/** Rolling window size for smoothing displayed confidence (reduces flicker). */
const SMOOTH_WINDOW   = 5;

const DETECT_SIZE     = 128;
const EMBED_SIZE      = 112;


interface VerifyResult {
  box:          FaceBox | null;
  id?:          string | null;
  name:         string | null;
  confidence:   number;
  fps:          number;
  imageWidth:   number;
  imageHeight:  number;
  orientation:  string;
  livenessPrompt: string | null;
  debugMesh?: {x: number, y: number}[];
  isFinal?: boolean;
}

const DEFAULT_RESULT: VerifyResult = {
  box: null, name: null, confidence: 0, fps: 0,
  imageWidth: 0, imageHeight: 0, orientation: 'portrait', livenessPrompt: null,
  isFinal: false,
};


export const VerifyScreen: React.FC = () => {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [cameraPos, setCameraPos] = useState<'front' | 'back'>('front');
  const { t } = useTranslation();
  const [langModalVisible, setLangModalVisible] = useState(false);

  const device = useCameraDevice(cameraPos);

  const { model: detectModel, anchors, isLoading: detectLoading, error: detectError } = useFaceDetector();
  const { model: embedModel,           isLoading: embedLoading,  error: embedError  } = useEmbedder();

  const [result,        setResult]        = useState<VerifyResult>(DEFAULT_RESULT);
  const [isFrozen,      setIsFrozen]      = useState(false);
  const [resetCounter,  setResetCounter]  = useState(0);
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
    if (r.isFinal) {
      setIsFrozen(true);
      if (r.name && r.id) {
        logAuthEvent(r.id, r.name, 'verify', true);
      }
    }
    
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

    if (globalObj._lastResetCounter !== resetCounter) {
      globalObj._lastResetCounter = resetCounter;
      globalObj._verifySeqState  = null;
      globalObj._sessionExpiry   = null;
      globalObj._successCooldown = null;
    }

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
        const rowBytes  = frame.bytesPerRow;

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

          // Temporal Smoothing (EMA) for BlazeFace jitter reduction
          if (topFace) {
            if (!globalObj._smoothedFace) {
              globalObj._smoothedFace = {
                xmin: topFace.xmin, ymin: topFace.ymin, xmax: topFace.xmax, ymax: topFace.ymax,
                landmarks: topFace.landmarks ? new Float32Array(topFace.landmarks) : new Float32Array(12)
              };
            } else {
              const ALPHA = 0.5; // Exponential Moving Average blend factor
              const s = globalObj._smoothedFace;
              s.xmin = s.xmin + ALPHA * (topFace.xmin - s.xmin);
              s.ymin = s.ymin + ALPHA * (topFace.ymin - s.ymin);
              s.xmax = s.xmax + ALPHA * (topFace.xmax - s.xmax);
              s.ymax = s.ymax + ALPHA * (topFace.ymax - s.ymax);
              if (topFace.landmarks) {
                for (let i = 0; i < 12; i++) {
                  s.landmarks[i] = s.landmarks[i] + ALPHA * (topFace.landmarks[i] - s.landmarks[i]);
                }
              }
            }
            // Overwrite raw coordinates with mathematically smoothed coordinates
            const s = globalObj._smoothedFace;
            topFace.xmin = s.xmin;
            topFace.ymin = s.ymin;
            topFace.xmax = s.xmax;
            topFace.ymax = s.ymax;
            topFace.landmarks = s.landmarks;
          } else {
            // Reset smoothing if face is lost
            globalObj._smoothedFace = null;
          }

        } catch (e: any) {
          console.log('[VerifyScreen] Detection error:', e?.message || e);
          scheduleOnRN(onResultBridge, {
            ...DEFAULT_RESULT, fps: currentFps, imageWidth: frameW, imageHeight: frameH, orientation: orient,
          } as VerifyResult);
          return;
        }
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
        let debugMesh: {x: number, y: number}[] | undefined;
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

        const livenessPrompt = livenessPassed ? null : seqState.tasks[seqState.currentTaskIndex];

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

        let embedding: any = new Float32Array(0);
        try {
          // Allocate reusable worklet-thread buffers on first use
          if (!globalObj._cropU8)    globalObj._cropU8    = new Uint8Array(EMBED_SIZE * EMBED_SIZE * 3);
          if (!globalObj._embedInput) globalObj._embedInput = new Float32Array(EMBED_SIZE * EMBED_SIZE * 3);

          const cropU8     = globalObj._cropU8     as Uint8Array;
          const embedInput = globalObj._embedInput as Float32Array;

          // 2a. Align face into Uint8Array RGB buffer using Affine Transform
          alignFaceToUint8(pixels, frameW, frameH, topFace.landmarks as Float32Array, rowBytes, fmt, cropU8);

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
        // ── Step 3: Match against enrolled faces ─────────────────────────
        const enrolled = enrolledShared.map(f => ({
          id:        f.id,
          name:      f.name,
          embedding: new Float32Array(f.embArray),
        }));

        const match = bestMatch(embedding, enrolled, VERIFY_THRESHOLD);
        const tMatchEnd = Date.now();

        // ── Performance log ──────────────────────────────────────────────
        const totalMs = tMatchEnd - tDetectStart;
        console.log(`[DEBUG_PERF] ML Processing time (Detection -> Embedding -> Match): ${totalMs} ms`);

        // CRITICAL: send livenessPrompt: null so livenessActive becomes false
        // and the result panel shows the name/confidence instead of liveness overlay.
        scheduleOnRN(onResultBridge, {
          box:        topFace,
          id:         match?.id ?? null,
          name:       match?.name ?? null,
          confidence: match?.similarity ?? 0,
          fps: currentFps, imageWidth: frameW, imageHeight: frameH, orientation: orient,
          livenessPrompt: null,
          isFinal: true,
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
  }, [detectModel, embedModel, anchors, enrolledShared, onResultBridge, resetCounter]);

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
    setResetCounter(c => c + 1);
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
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
          <Text style={styles.screenTitle}>{t('tab_verify')}</Text>
        </View>
        <View style={styles.headerRight}>
          <TouchableOpacity onPress={() => setLangModalVisible(true)} style={styles.settingsIcon}>
            <Settings size={20} color="#6b7280" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.flipBtn} onPress={() => setCameraPos(p => p === 'front' ? 'back' : 'front')}>
            <Text style={styles.flipBtnText}>FLIP</Text>
          </TouchableOpacity>
          <View style={styles.telemetryPill}>
            <Text style={styles.metricLabel}>FPS</Text>
            <Text style={[styles.metricValue, result.fps < 20 && { color: '#FF9F0A' }]}>
              {result.fps > 0 ? result.fps : '--'}
            </Text>
          </View>
        </View>
      </View>



      {/* Camera viewport */}
      <View style={StyleSheet.absoluteFill} onLayout={(e) => setCameraLayout(e.nativeEvent.layout)}>
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={!isFrozen}
          outputs={cameraOutputs}
          resizeMode="contain"
        />

        {isLoading && (
          <View style={styles.overlay}>
            <ActivityIndicator size="large" color="#f0a500" />
            <Text style={styles.overlayText}>{t('verify_initializing')}</Text>
            <Text style={styles.overlaySubText}>{t('initialization_note')}</Text>
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
          />
        )}
        
        {/* Liveness Prompt Overlay */}
        {result.livenessPrompt && !result.name && (
          <Reanimated.View 
            style={styles.livenessPrompt}
          >
            <View style={{ flexDirection: 'row', alignItems: 'center', gap: 12 }}>
              <Text style={styles.livenessText} numberOfLines={1} adjustsFontSizeToFit>
                {result.livenessPrompt === 'turn_left' ? t('liveness_left') :
                 result.livenessPrompt === 'turn_right' ? t('liveness_right') :
                 t('liveness_left')}
              </Text>
              <TouchableOpacity onPress={handleRetryLiveness} style={styles.livenessRetryBtn}>
                <Text style={styles.livenessRetryText}>↻</Text>
              </TouchableOpacity>
            </View>
          </Reanimated.View>
        )}
      </View>

      {/* Bottom HUD */}
      {!livenessActive && (
        <View style={styles.bottomHUD}>
          <Text style={[styles.statusText, { color: overlayStatus === 'matched' ? '#34C759' : overlayStatus === 'unknown' ? '#FF3B30' : '#e8eaf0' }]}>
            {isFrozen && result.name ? t('verify_logged', { name: result.name.toUpperCase() }) :
             result.name ? t('verify_matched', { name: result.name.toUpperCase(), confidence: (result.confidence * 100).toFixed(1) }) :
             result.box ? t('unknown_subject') :
             isLoading ? t('verify_initializing') :
             t('verify_scanning')}
          </Text>
          {isFrozen && (
            <TouchableOpacity onPress={() => {
                setIsFrozen(false);
                setResult(DEFAULT_RESULT);
                setResetCounter(c => c + 1);
            }}>
              <Text style={{ fontFamily: 'DMSans-Bold', fontSize: 11, color: '#007AFF', letterSpacing: 1.0, textTransform: 'uppercase' }}>{t('next_scan')}</Text>
            </TouchableOpacity>
          )}
        </View>
      )}

      <LanguagePickerModal visible={langModalVisible} onClose={() => setLangModalVisible(false)} />
    </View>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0a0c0f' },
  centered: {
    flex: 1, backgroundColor: '#0a0c0f',
    alignItems: 'center', justifyContent: 'center',
  },
  infoText: { fontFamily: 'DMSans-Regular', color: '#6b7280', fontSize: 14 },

  header: {
    position: 'absolute',
    top: 0, left: 0, right: 0,
    zIndex: 10,
    paddingTop: Platform.OS === 'ios' ? 56 : 28,
    paddingHorizontal: 20,
    paddingBottom: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(13, 17, 23, 0.85)',
  },
  screenTitle: {
    fontFamily: 'BarlowCondensed-Bold',
    fontSize: 22,
    letterSpacing: 0.5,
    color: '#e8eaf0',
    textTransform: 'uppercase',
  },
  headerRight: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  settingsIcon: {
    padding: 6,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#1e2433',
  },
  flipBtn: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#1e2433',
  },
  flipBtnText: {
    fontFamily: 'DMSans-Bold',
    color: '#e8eaf0',
    fontSize: 10,
    letterSpacing: 0.5,
  },
  telemetryPill: {
    flexDirection: 'row',
    gap: 10,
    backgroundColor: 'rgba(13, 17, 23, 0.85)',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#1e2433',
    paddingHorizontal: 10,
    paddingVertical: 5,
    alignItems: 'center',
  },
  metricLabel: {
    fontFamily: 'IBMPlexMono-Regular',
    fontSize: 10,
    color: '#3d4451',
    textTransform: 'uppercase',
    letterSpacing: 1.0,
  },
  metricValue: {
    fontFamily: 'IBMPlexMono-SemiBold',
    fontSize: 14,
    color: '#34C759',
    letterSpacing: -0.5,
  },

  overlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(10,12,15,0.90)',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
  },
  overlayText:    { fontFamily: 'DMSans-SemiBold', color: '#e8eaf0', fontSize: 16 },
  overlaySubText: { fontFamily: 'DMSans-Regular', color: '#6b7280', fontSize: 14 },
  errorText:      { fontFamily: 'DMSans-Bold', color: '#FF3B30', fontSize: 14, textAlign: 'center', padding: 24 },

  livenessPrompt: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 100 : 72,
    alignSelf: 'center',
    backgroundColor: 'rgba(0, 122, 255, 0.12)',
    borderWidth: 1,
    borderColor: 'rgba(0, 122, 255, 0.3)',
    borderRadius: 10,
    paddingHorizontal: 16,
    paddingVertical: 12,
    maxWidth: '90%',
  },
  livenessText: {
    fontFamily: 'BarlowCondensed-SemiBold',
    fontSize: 18,
    letterSpacing: 0.5,
    color: '#007AFF',
    textAlign: 'center',
    textTransform: 'uppercase',
    flexShrink: 1,
  },
  livenessRetryBtn: {
    backgroundColor: 'rgba(0,122,255,0.2)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 6,
  },
  livenessRetryText: {
    fontFamily: 'DMSans-Bold',
    color: '#007AFF',
    fontSize: 16,
  },

  bottomHUD: {
    position: 'absolute',
    bottom: 0, left: 0, right: 0,
    paddingHorizontal: 20,
    paddingVertical: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(10, 12, 15, 0.9)',
    borderTopWidth: 1,
    borderTopColor: '#1e2433',
  },
  statusText: {
    fontFamily: 'DMSans-Bold',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 1.5,
  },
});