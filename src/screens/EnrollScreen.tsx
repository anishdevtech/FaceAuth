/**
 * EnrollScreen — Register a new face.
 *
 * Flow:
 *   1. Live camera preview with BlazeFace detection overlay.
 *   2. User presses "Capture" → triggers a 5-frame embedding burst.
 *   3. Embeddings are averaged and L2-normalised into a canonical face vector.
 *   4. User types a name → embedding saved to MMKV → success alert.
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
  Alert,
  Animated,
  Dimensions,
  Platform,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
} from 'react-native-vision-camera';
import { scheduleOnRN } from 'react-native-worklets';

import { useFaceDetector }                                   from '../ml/useFaceDetector';
import { useEmbedder }                                       from '../ml/useEmbedder';
import { decodeFaces, type FaceBox }                        from '../ml/blazeface';
import { resizeToFloat32, cropAndResizeFace, typedArrayToBuffer } from '../ml/preprocessing';
import { l2Normalize }                                       from '../utils/mathUtils';
import { saveFace }                                          from '../storage/faceStore';
import { FaceOverlay }                                       from '../components/FaceOverlay';

// ─── Constants ────────────────────────────────────────────────────────────────

const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

const CAPTURE_BURST      = 5;
const FACE_CROP_PADDING  = 0.25;
const DETECT_SIZE        = 128;
const EMBED_SIZE         = 160;
const PREVIEW_INTERVAL_MS = 300;

// ─── Types ────────────────────────────────────────────────────────────────────

type EnrollState =
  | 'idle'
  | 'capturing'
  | 'captured'
  | 'saving'
  | 'error';

interface ImageMeta {
  width:       number;
  height:      number;
  orientation: string;
}

// ─── Component ────────────────────────────────────────────────────────────────

export const EnrollScreen: React.FC = () => {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');

  const { model: detectModel, anchors, isLoading: detectLoading, error: detectError } = useFaceDetector();
  const { model: embedModel,          isLoading: embedLoading,  error: embedError  } = useEmbedder();

  const [faceBox,       setFaceBox]       = useState<FaceBox | null>(null);
  const [imageMeta,     setImageMeta]     = useState<ImageMeta>({ width: 0, height: 0, orientation: 'portrait' });
  const [enrollState,   setEnrollState]   = useState<EnrollState>('idle');
  const [captureCount,  setCaptureCount]  = useState(0);
  const [finalEmbedding, setFinalEmbedding] = useState<Float32Array | null>(null);
  const [personName,    setPersonName]    = useState('');
  const [showNameInput, setShowNameInput] = useState(false);
  const [cameraLayout,  setCameraLayout]  = useState({ width: SCREEN_W, height: SCREEN_H * 0.75 });

  // Worklet-safe refs — React state cannot be reliably read inside a worklet
  // because the closure is captured at render time. Refs always expose the
  // current value without re-capturing.
  const embeddingsAccum   = useRef<number[][]>([]);

  const progressAnim = useRef(new Animated.Value(0)).current;

  // ─── Derived ──────────────────────────────────────────────────────────────
  const isCapturing  = enrollState === 'capturing';

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  useEffect(() => {
    Animated.spring(progressAnim, {
      toValue:       captureCount / CAPTURE_BURST,
      useNativeDriver: false,
      tension:       80,
      friction:      8,
    }).start();
  }, [captureCount, progressAnim]);

  // ─── Worklet callbacks (scheduled to JS thread) ───────────────────────────

  const onFaceDetected = useCallback(
    (box: FaceBox | null, meta: ImageMeta) => {
      setFaceBox(box);
      setImageMeta(meta);
    },
    [],
  );

  const onEmbeddingCaptured = useCallback((embedding: number[]) => {
    if (embeddingsAccum.current.length >= CAPTURE_BURST) return;

    embeddingsAccum.current.push(embedding);
    const count = embeddingsAccum.current.length;
    setCaptureCount(count);

    if (count < CAPTURE_BURST) {
      return;
    }

    // All frames captured — average and normalise
    const dim = embedding.length;
    const avg = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      let sum = 0;
      for (const emb of embeddingsAccum.current) sum += emb[i];
      avg[i] = sum / CAPTURE_BURST;
    }

    setFinalEmbedding(l2Normalize(avg));
    setEnrollState('captured');
    setShowNameInput(true);
  }, []);

  const onWorkletError = useCallback((msg: string) => {
    console.error('[EnrollScreen]', msg);
    setEnrollState('error');
  }, []);

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

      try {
        const now      = Date.now();
        const pixels   = new Uint8Array(frame.getPixelBuffer());
        const orient   = frame.orientation;
        const isWrongDim = frame.width > frame.height; // Assuming app is portrait
        const frameW   = isWrongDim ? frame.height : frame.width;
        const frameH   = isWrongDim ? frame.width  : frame.height;
        const fmt      = frame.pixelFormat;
        const bpp      = (fmt as string) === 'rgb' ? 3 : 4;
        const rowBytes = frameW * bpp;
        const meta: ImageMeta = { width: frameW, height: frameH, orientation: orient };

        const lastPrev = (globalThis as any)._lastPreviewTime || 0;
        if (now - lastPrev >= PREVIEW_INTERVAL_MS) {
          (globalThis as any)._lastPreviewTime = now;

          let topFace: FaceBox | null = null;
          try {
            const inputTensor  = resizeToFloat32(pixels, frameW, frameH, DETECT_SIZE, DETECT_SIZE, rowBytes, fmt);
            const detOutputs = detectModel.runSync([typedArrayToBuffer(inputTensor)]);
            const regressors   = new Float32Array(detOutputs[0]);
            const scores       = new Float32Array(detOutputs[1]);
            const faces        = decodeFaces(regressors, scores, anchors);
            topFace = faces.length > 0 ? faces[0] : null;
          } catch {
            // Detection failed this frame — skip overlay update
          }

          scheduleOnRN(onFaceDetected, topFace, meta);

          if (isCapturing && topFace) {
            try {
              const faceInput  = cropAndResizeFace(pixels, frameW, frameH, topFace, EMBED_SIZE, EMBED_SIZE, rowBytes, fmt, FACE_CROP_PADDING);
              const embOutputs = embedModel.runSync([typedArrayToBuffer(faceInput)]);
              const rawEmb     = new Float32Array(embOutputs[0]);
              scheduleOnRN(onEmbeddingCaptured, Array.from(l2Normalize(rawEmb)));
            } catch {
              scheduleOnRN(onWorkletError, 'Embedding failed — ensure face is clearly visible.');
            }
          }
        }
      } finally {
        frame.dispose();
      }
    },
  });

  const cameraOutputs = useMemo(() => [frameOutput], [frameOutput]);

  // ─── Handlers ─────────────────────────────────────────────────────────────

  const handleCapture = useCallback(() => {
    if (!faceBox) {
      Alert.alert('No face detected', 'Position your face clearly in the frame.');
      return;
    }
    if (!detectModel || !embedModel) {
      Alert.alert('Models loading', 'Please wait for AI models to finish loading.');
      return;
    }
    embeddingsAccum.current = [];
    setCaptureCount(0);
    setFinalEmbedding(null);
    setEnrollState('capturing');
  }, [faceBox, detectModel, embedModel]);

  const handleSave = useCallback(async () => {
    if (!finalEmbedding || !personName.trim()) {
      Alert.alert('Enter a name', 'Please type the person\'s name before saving.');
      return;
    }
    setEnrollState('saving');
    try {
      await saveFace(personName.trim(), finalEmbedding);
      setShowNameInput(false);
      setPersonName('');
      embeddingsAccum.current = [];
      setCaptureCount(0);
      setFinalEmbedding(null);
      setEnrollState('idle');
      Alert.alert('Enrolled', `"${personName.trim()}" has been registered successfully.`);
    } catch (e: any) {
      Alert.alert('Save failed', e?.message ?? 'An unexpected error occurred.');
      setEnrollState('error');
    }
  }, [finalEmbedding, personName]);

  const handleCancel = useCallback(() => {
    embeddingsAccum.current     = [];
    setShowNameInput(false);
    setPersonName('');
    setCaptureCount(0);
    setFinalEmbedding(null);
    setEnrollState('idle');
  }, []);

  // ─── Derived ──────────────────────────────────────────────────────────────

  const isLoading    = detectLoading || embedLoading;
  const loadError    = detectError   || embedError;

  const statusMessage = isLoading
    ? '⏳ Loading AI models…'
    : isCapturing
      ? `Capturing  ${captureCount} / ${CAPTURE_BURST}`
      : faceBox
        ? '✓ Face detected — press Capture'
        : 'Scanning for face…';

  // ─── Render guards ────────────────────────────────────────────────────────

  if (!hasPermission) {
    return (
      <View style={styles.centered}>
        <Text style={styles.statusText}>Camera permission required</Text>
        <TouchableOpacity style={styles.btn} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.centered}>
        <Text style={styles.statusText}>No front camera found</Text>
      </View>
    );
  }

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="#0a0c0f" />

      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>Enroll New Face</Text>
          <Text style={styles.headerSub}>Stand still · look at camera · press Capture</Text>
        </View>
        {isCapturing && (
          <View style={styles.burstBadge}>
            <Text style={styles.burstBadgeText}>{captureCount}/{CAPTURE_BURST}</Text>
          </View>
        )}
      </View>

      {/* Camera viewport */}
      <View style={styles.cameraContainer} onLayout={(e) => setCameraLayout(e.nativeEvent.layout)}>
        <Camera
          style={StyleSheet.absoluteFill}
          device={device}
          isActive={!showNameInput}
          outputs={cameraOutputs}
          resizeMode="contain"
        />

        {isLoading && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator size="large" color="#f0a500" />
            <Text style={styles.loadingText}>Loading AI models…</Text>
          </View>
        )}

        {!!loadError && !isLoading && (
          <View style={styles.loadingOverlay}>
            <Text style={styles.errorText}>{loadError}</Text>
          </View>
        )}

        {!isLoading && (
          <FaceOverlay
            box={faceBox}
            label={faceBox ? (isCapturing ? `${captureCount}/${CAPTURE_BURST}` : 'DETECTED') : undefined}
            status={faceBox ? 'matched' : 'scanning'}
            frameWidth={cameraLayout.width}
            frameHeight={cameraLayout.height}
            imageWidth={imageMeta.width}
            imageHeight={imageMeta.height}
            imageOrientation={imageMeta.orientation}
          />
        )}

        {isCapturing && (
          <View style={styles.progressContainer}>
            <Animated.View
              style={[
                styles.progressFill,
                {
                  width: progressAnim.interpolate({
                    inputRange:  [0, 1],
                    outputRange: ['0%', '100%'],
                  }),
                },
              ]}
            />
          </View>
        )}

        <View style={styles.statusBanner}>
          <Text style={styles.statusBannerText}>{statusMessage}</Text>
        </View>
      </View>

      {/* Capture button */}
      {!showNameInput && (
        <View style={styles.controls}>
          <TouchableOpacity
            style={[styles.captureBtn, (!faceBox || isLoading || isCapturing) && styles.captureBtnDisabled]}
            onPress={handleCapture}
            disabled={!faceBox || isLoading || isCapturing}
            activeOpacity={0.75}
          >
            {isCapturing
              ? <ActivityIndicator size="small" color="#0a0c0f" />
              : <View style={styles.captureInner} />
            }
          </TouchableOpacity>
          <Text style={styles.captureHint}>{isCapturing ? 'CAPTURING…' : 'CAPTURE'}</Text>
        </View>
      )}

      {/* Name input sheet */}
      {showNameInput && (
        <View style={styles.nameSheet}>
          <Text style={styles.nameSheetTitle}>Who is this person?</Text>
          <Text style={styles.nameSheetSub}>{CAPTURE_BURST}-frame embedding captured and averaged.</Text>
          <TextInput
            style={styles.nameInput}
            placeholder="Enter full name…"
            placeholderTextColor="#3d4561"
            value={personName}
            onChangeText={setPersonName}
            autoFocus
            returnKeyType="done"
            onSubmitEditing={handleSave}
            autoCapitalize="words"
            autoCorrect={false}
          />
          <View style={styles.nameActions}>
            <TouchableOpacity style={styles.cancelBtn} onPress={handleCancel} activeOpacity={0.75}>
              <Text style={styles.cancelBtnText}>Cancel</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.saveBtn, (!personName.trim() || enrollState === 'saving') && styles.saveBtnDisabled]}
              onPress={handleSave}
              disabled={!personName.trim() || enrollState === 'saving'}
              activeOpacity={0.8}
            >
              {enrollState === 'saving'
                ? <ActivityIndicator size="small" color="#0a0c0f" />
                : <Text style={styles.saveBtnText}>Save Face</Text>
              }
            </TouchableOpacity>
          </View>
        </View>
      )}
    </View>
  );
};

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  root:    { flex: 1, backgroundColor: '#0a0c0f' },
  centered: {
    flex: 1, backgroundColor: '#0a0c0f',
    alignItems: 'center', justifyContent: 'center', padding: 32,
  },

  header: {
    paddingTop:        Platform.OS === 'ios' ? 56 : 20,
    paddingHorizontal: 20,
    paddingBottom:     14,
    backgroundColor:   '#0d1117',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#1e2433',
    flexDirection:     'row',
    justifyContent:    'space-between',
    alignItems:        'flex-end',
  },
  headerTitle: { fontSize: 18, fontWeight: '700', color: '#e8eaf0', letterSpacing: 0.3 },
  headerSub:   { fontSize: 12, color: '#4a5568', marginTop: 3, letterSpacing: 0.2 },
  burstBadge:  { backgroundColor: '#f0a500', borderRadius: 12, paddingHorizontal: 10, paddingVertical: 4 },
  burstBadgeText: { color: '#0a0c0f', fontWeight: '800', fontSize: 13 },

  cameraContainer: { flex: 1, width: '100%', overflow: 'hidden', backgroundColor: '#000' },

  loadingOverlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(10,12,15,0.88)',
    alignItems:      'center',
    justifyContent:  'center',
    gap: 12,
  },
  loadingText: { color: '#9da3b4', fontSize: 15, fontWeight: '600' },
  errorText:   { color: '#ff4d4f', fontSize: 14, textAlign: 'center', padding: 24 },

  progressContainer: {
    position: 'absolute', top: 0, left: 0, right: 0,
    height: 3, backgroundColor: 'rgba(240,165,0,0.2)',
  },
  progressFill: { height: '100%', backgroundColor: '#f0a500', borderRadius: 2 },

  statusBanner: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    backgroundColor:  'rgba(0,0,0,0.65)',
    paddingVertical:  9, paddingHorizontal: 16,
  },
  statusBannerText: { color: '#e8eaf0', fontSize: 13, textAlign: 'center', letterSpacing: 0.2 },

  controls:          { alignItems: 'center', paddingVertical: 18, backgroundColor: '#0d1117' },
  captureBtn: {
    width: 72, height: 72, borderRadius: 36,
    borderWidth: 3.5, borderColor: '#f0a500',
    alignItems: 'center', justifyContent: 'center',
  },
  captureBtnDisabled: { borderColor: '#282d3d', opacity: 0.5 },
  captureInner: { width: 54, height: 54, borderRadius: 27, backgroundColor: '#f0a500' },
  captureHint:  { color: '#4a5568', fontSize: 10, marginTop: 7, letterSpacing: 2, fontWeight: '700' },

  nameSheet: {
    padding:        24,
    paddingBottom:  Platform.OS === 'ios' ? 36 : 24,
    backgroundColor: '#0d1117',
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#1e2433',
  },
  nameSheetTitle: { color: '#e8eaf0', fontSize: 17, fontWeight: '700', marginBottom: 4 },
  nameSheetSub:   { color: '#4a5568', fontSize: 12, marginBottom: 16 },
  nameInput: {
    backgroundColor: '#161b27',
    borderWidth: 1, borderColor: '#252b3b', borderRadius: 10,
    paddingHorizontal: 14, paddingVertical: 13,
    color: '#e8eaf0', fontSize: 16, marginBottom: 16,
  },
  nameActions:    { flexDirection: 'row', gap: 12 },
  cancelBtn: {
    flex: 1, paddingVertical: 13, borderRadius: 10,
    borderWidth: 1, borderColor: '#252b3b', alignItems: 'center',
  },
  cancelBtnText:    { color: '#9da3b4', fontWeight: '600', fontSize: 15 },
  saveBtn:          { flex: 2, paddingVertical: 13, borderRadius: 10, backgroundColor: '#f0a500', alignItems: 'center' },
  saveBtnDisabled:  { backgroundColor: '#282d3d', opacity: 0.5 },
  saveBtnText:      { color: '#0a0c0f', fontWeight: '800', fontSize: 15 },

  statusText: { color: '#9da3b4', fontSize: 15, marginBottom: 20, textAlign: 'center', lineHeight: 22 },
  btn:        { backgroundColor: '#f0a500', paddingHorizontal: 28, paddingVertical: 13, borderRadius: 10 },
  btnText:    { color: '#0a0c0f', fontWeight: '800', fontSize: 15 },
});