/**
 * VerifyScreen — Real-time face identification.
 *
 * Per-frame pipeline (~5 FPS):
 * Frame  →  BlazeFace detect (128×128)  →  crop + pad  →
 * MobileFaceNet embed (112×112)  →  L2 normalise  →
 * cosine similarity vs enrolled set  →  display result
 */

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
} from 'react-native';
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
import { resizeToFloat32, cropAndResizeFace, typedArrayToBuffer } from '../ml/preprocessing';
import { bestMatch, l2Normalize }                                 from '../utils/mathUtils';
import { getAllFaces, type EnrolledFace }                         from '../storage/faceStore';
import { FaceOverlay }                                            from '../components/FaceOverlay';

// ─── Constants ────────────────────────────────────────────────────────────────

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

/** Enrolled faces are reloaded every 3 s in case the user enrols in another tab. */
const FACE_RELOAD_INTERVAL_MS = 3_000;

/** Rolling window size for smoothing displayed confidence (reduces flicker). */
const SMOOTH_WINDOW   = 5;

const DETECT_SIZE     = 128;
const EMBED_SIZE      = 160;
const FACE_CROP_PADDING = 0.2;

// ─── Types ────────────────────────────────────────────────────────────────────

interface VerifyResult {
  box:          FaceBox | null;
  name:         string | null;
  confidence:   number;
  fps:          number;
  imageWidth:   number;
  imageHeight:  number;
  orientation:  string;
}

const DEFAULT_RESULT: VerifyResult = {
  box: null, name: null, confidence: 0, fps: 0,
  imageWidth: 0, imageHeight: 0, orientation: 'portrait',
};

// ─── Component ────────────────────────────────────────────────────────────────

export const VerifyScreen: React.FC = () => {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');

  const { model: detectModel, anchors, isLoading: detectLoading, error: detectError } = useFaceDetector();
  const { model: embedModel,           isLoading: embedLoading,  error: embedError  } = useEmbedder();

  const [result,        setResult]        = useState<VerifyResult>(DEFAULT_RESULT);
  const [enrolledFaces, setEnrolledFaces] = useState<EnrolledFace[]>([]);
  const [cameraLayout,  setCameraLayout]  = useState({ width: SCREEN_W, height: SCREEN_H - 160 });

  const confAnim = useRef(new Animated.Value(0)).current;

  // We cannot use a mutable ref for data sent to the worklet because worklets freeze
  // objects passed to them. We derive a plain array from the state and pass it down.
  const onResultRef    = useRef<(r: VerifyResult) => void>(() => {});

  // Worklet-side perf counters are tracked via globalThis in the worklet thread
  // to avoid mutating frozen ref objects.
  const confidenceHistory = useRef<number[]>([]);

  // ─── Result handler with confidence smoothing ─────────────────────────────

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

  // ─── Enrolled face reload ─────────────────────────────────────────────────

  useEffect(() => {
    const load = () => {
      const faces = getAllFaces();
      setEnrolledFaces(faces);
    };
    load();
    const id = setInterval(load, FACE_RELOAD_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  const enrolledShared = useMemo(() => {
    return enrolledFaces.map(f => ({
      id:        f.id,
      name:      f.name,
      embedding: Array.from(f.embedding), // number[] is serialisable
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

  const frameOutput = useFrameOutput({
    pixelFormat: 'rgb',
    enablePhysicalBufferRotation: true,
    onFrame: (frame) => {
      'worklet';

      if (!detectModel || !embedModel || !anchors) {
        frame.dispose();
        return;
      }

      const now = Date.now();

      // Gate first — only increment frameCount for frames that will be processed
      const lastProc = (globalThis as any)._lastProcessed || 0;
      if (now - lastProc < PROCESS_INTERVAL_MS) {
        frame.dispose();
        return;
      }
      (globalThis as any)._lastProcessed = now;

      (globalThis as any)._frameCount = ((globalThis as any)._frameCount || 0) + 1;
      const lastFpsTime = (globalThis as any)._lastFpsTime || now;
      const elapsed = now - lastFpsTime;
      if (elapsed >= 1000) {
        (globalThis as any)._fpsValue   = Math.round(((globalThis as any)._frameCount * 1000) / elapsed);
        (globalThis as any)._frameCount = 0;
        (globalThis as any)._lastFpsTime = now;
      }
      const currentFps = (globalThis as any)._fpsValue || 0;

      try {
        const pixels    = new Uint8Array(frame.getPixelBuffer());
        const orient    = frame.orientation;
        // Since we enabled physical buffer rotation, the buffer IS rotated to match orient.
        // But VisionCamera might still report the unrotated dimensions, so we swap them.
        const isWrongDim = frame.width > frame.height; // assuming portrait app
        const frameW    = isWrongDim ? frame.height : frame.width;
        const frameH    = isWrongDim ? frame.width  : frame.height;
        const fmt       = frame.pixelFormat;
        // Recalculate row bytes because frame.bytesPerRow might still report the unrotated value
        const bpp       = (fmt as string) === 'rgb' ? 3 : 4;
        const rowBytes  = frameW * bpp;

        // ── Step 1: BlazeFace detection ──────────────────────────────────

        let topFace: FaceBox | null = null;

        try {
          const detectInput  = resizeToFloat32(pixels, frameW, frameH, DETECT_SIZE, DETECT_SIZE, rowBytes, fmt);
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

        if (!topFace) {
          scheduleOnRN(onResultBridge, {
            box: null, name: null, confidence: 0,
            fps: currentFps, imageWidth: frameW, imageHeight: frameH, orientation: orient,
          } as VerifyResult);
          return;
        }

        // ── Step 2: MobileFaceNet embedding ──────────────────────────────

        let embedding: Float32Array;

        try {
          const faceInput  = cropAndResizeFace(pixels, frameW, frameH, topFace, EMBED_SIZE, EMBED_SIZE, rowBytes, fmt, FACE_CROP_PADDING);
          const embOut     = embedModel.runSync([typedArrayToBuffer(faceInput)]);
          embedding        = l2Normalize(new Float32Array(embOut[0]));
        } catch (e: any) {
          console.log('[VerifyScreen] Embed error:', e?.message || e);
          scheduleOnRN(onResultBridge, {
            box: topFace, name: null, confidence: 0,
            fps: currentFps, imageWidth: frameW, imageHeight: frameH, orientation: orient,
          } as VerifyResult);
          return;
        }

        // ── Step 3: Match against enrolled faces ─────────────────────────

        // Build Float32Array embeddings in the worklet from the shared number[]
        const enrolled = enrolledShared.map(f => ({
          id:        f.id,
          name:      f.name,
          embedding: new Float32Array(f.embedding),   // safe to construct in worklet
        }));

        const match = bestMatch(embedding, enrolled, VERIFY_THRESHOLD);

        scheduleOnRN(onResultBridge, {
          box: topFace, name: match?.name ?? null, confidence: match?.similarity ?? 0,
          fps: currentFps, imageWidth: frameW, imageHeight: frameH, orientation: orient,
        } as VerifyResult);

      } finally {
        frame.dispose();
      }
    },
  });

  // Memoize outputs array — prevents Camera from restarting the pipeline on
  // every render, which causes intermittent frame drops.
  const cameraOutputs = useMemo(() => [frameOutput], [frameOutput]);

  // ─── Derived ──────────────────────────────────────────────────────────────

  const isLoading   = detectLoading || embedLoading;
  const loadError   = detectError   || embedError;

  const overlayStatus =
    result.name ? 'matched'  :
    result.box  ? 'unknown'  :
                  'scanning';

  const overlayLabel =
    result.name ? `${result.name}  ${Math.round(result.confidence * 100)}%` :
    result.box  ? 'UNKNOWN' :
                  undefined;

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
        <Text style={styles.headerTitle}>Live Verification</Text>
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
          />
        )}

        {/* Result panel */}
        <View style={styles.resultPanel}>
          {result.name ? (
            <View>
              <View style={styles.matchRow}>
                <View style={styles.matchLeft}>
                  <View style={styles.matchDot} />
                  <Text style={styles.matchName} numberOfLines={1}>{result.name}</Text>
                </View>
                <Text style={styles.confPercent}>{Math.round(result.confidence * 100)}%</Text>
              </View>
              <View style={styles.confBarTrack}>
                <Animated.View
                  style={[
                    styles.confBarFill,
                    {
                      width: confAnim.interpolate({ inputRange: [0, 1], outputRange: ['0%', '100%'] }),
                      backgroundColor: confAnim.interpolate({
                        inputRange:  [VERIFY_THRESHOLD, 0.75, 1],
                        outputRange: ['#f0a500', '#4ade80', '#4ade80'],
                      }),
                    },
                  ]}
                />
              </View>
            </View>
          ) : result.box ? (
            <View style={styles.unknownRow}>
              <Text style={styles.unknownLabel}>UNKNOWN PERSON</Text>
              <Text style={styles.unknownSub}>
                No match above {Math.round(VERIFY_THRESHOLD * 100)}% threshold
              </Text>
            </View>
          ) : (
            <Text style={styles.scanningLabel}>
              {isLoading
                ? 'Loading…'
                : enrolledFaces.length === 0
                  ? 'No faces enrolled — open the Enroll tab to add faces'
                  : 'Scanning for face…'}
            </Text>
          )}
        </View>
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
  headerTitle: { fontSize: 18, fontWeight: '700', color: '#e8eaf0', letterSpacing: 0.3 },
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
    bottom: 0, left: 0, right: 0,
    backgroundColor:   'rgba(10,12,15,0.90)',
    paddingHorizontal: 20,
    paddingTop:        14,
    paddingBottom:     18,
    borderTopWidth:    StyleSheet.hairlineWidth,
    borderTopColor:    '#1e2433',
    minHeight:         72,
    justifyContent:    'center',
  },

  matchRow:   { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  matchLeft:  { flexDirection: 'row', alignItems: 'center', flex: 1, gap: 10 },
  matchDot:   { width: 10, height: 10, borderRadius: 5, backgroundColor: '#4ade80' },
  matchName:  { color: '#4ade80', fontSize: 20, fontWeight: '800', letterSpacing: 0.3, flexShrink: 1 },
  confPercent: { color: '#4ade80', fontSize: 18, fontWeight: '700', marginLeft: 12 },
  confBarTrack: { height: 4, backgroundColor: '#1e2433', borderRadius: 2, overflow: 'hidden' },
  confBarFill:  { height: '100%', borderRadius: 2 },

  unknownRow:   { gap: 4 },
  unknownLabel: { color: '#ff4d4f', fontSize: 18, fontWeight: '800' },
  unknownSub:   { color: '#4a5568', fontSize: 12 },

  scanningLabel: { color: '#9da3b4', fontSize: 14, textAlign: 'center', lineHeight: 20 },

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