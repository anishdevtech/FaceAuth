// Provides the user interface and orchestration logic for registering new facial profiles.
// The enrollment workflow consists of:
// 1. Liveness verification via randomized head tracking.
// 2. Capture of multiple consecutive frames to generate a robust aggregate embedding.
// 3. Name assignment and persistent storage of the generated feature vector.

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
  KeyboardAvoidingView,
  Platform,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  Alert,
  View,
} from 'react-native';
import Reanimated, { runOnUI } from 'react-native-reanimated';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
} from 'react-native-vision-camera';
import { scheduleOnRN } from 'react-native-worklets';

import { resizeToFloat32, alignFaceToUint8, uint8ToFloat32Normalized, typedArrayToBuffer } from '../ml/preprocessing';
import { applyCLAHE }                                                from '../ml/clahe';
import { useFaceDetector } from '../ml/useFaceDetector';
import { decodeFaces, type FaceBox } from '../ml/blazeface';
import { useEmbedder } from '../ml/useEmbedder';
import { l2Normalize }                                       from '../utils/mathUtils';
import { saveFace }                                          from '../storage/faceStore';
import { logAuthEvent }                                      from '../storage/authSync';
import { generateLivenessSequence, getStepPrompt, checkSequenceTask } from '../liveness/ActiveLivenessSequence';
import { FaceOverlay }   from '../components/FaceOverlay';
import { ConfirmModal } from '../components/ConfirmModal';
import { useTranslation } from '../i18n';
import { Settings } from 'lucide-react-native';
import { LanguagePickerModal } from '../components/LanguagePickerModal';


const { width: SCREEN_W, height: SCREEN_H } = Dimensions.get('window');

const CAPTURE_BURST      = 5;
const DETECT_SIZE        = 128;
const EMBED_SIZE         = 112;
const PREVIEW_INTERVAL_MS = 350;


type EnrollState =
  | 'idle'
  | 'liveness_check'
  | 'capturing'
  | 'captured'
  | 'saving'
  | 'error';

interface ImageMeta {
  width:       number;
  height:      number;
  orientation: string;
}


export const EnrollScreen: React.FC = () => {
  const { hasPermission, requestPermission } = useCameraPermission();
  const [cameraPos, setCameraPos] = useState<'front' | 'back'>('front');
  const { t } = useTranslation();
  const [langModalVisible, setLangModalVisible] = useState(false);
  const device = useCameraDevice(cameraPos);

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
  const [livenessPrompt, setLivenessPrompt] = useState<string | null>(null);

  // State management for the generic confirmation modal.
  const [modal, setModal] = useState<{
    visible: boolean;
    icon: string;
    title: string;
    subtitle?: string;
    confirmText?: string;
    cancelText?: string;
    confirmDestructive?: boolean;
    onConfirm: () => void;
    onCancel?: () => void;
  }>({
    visible: false,
    icon: '',
    title: '',
    onConfirm: () => {},
  });
  const hideModal = useCallback(() => setModal(m => ({ ...m, visible: false })), []);

  const embeddingsAccum   = useRef<number[][]>([]);
  const progressAnim = useRef(new Animated.Value(0)).current;

  // Derived boolean flags for UI rendering.
  const isCapturing  = enrollState === 'capturing';
  const isLiveness   = enrollState === 'liveness_check';

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

  // Callbacks executed by the UI thread, triggered from VisionCamera worklets.

  const onFaceDetected = useCallback(
    (box: FaceBox | null, meta: ImageMeta, prompt: string | null) => {
      setFaceBox(box);
      setImageMeta(meta);
      setLivenessPrompt(prompt);
    },
    [],
  );

  const onLivenessPassed = useCallback(() => {
    setEnrollState('capturing');
    setLivenessPrompt(null);
  }, []);

  const onEmbeddingCaptured = useCallback((embedding: number[]) => {
    if (embeddingsAccum.current.length >= CAPTURE_BURST) return;

    embeddingsAccum.current.push(embedding);
    const count = embeddingsAccum.current.length;
    setCaptureCount(count);

    if (count < CAPTURE_BURST) {
      return;
    }

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
  // onFrame deps include isLiveness/isCapturing directly — the closure captures
  // the current value. When state changes, onFrame is recreated and
  // useFrameOutput picks up the new processor (2 transitions total, no perf issue).

  const onFrame = useCallback((frame: any) => {
    'worklet';
    const globalObj = globalThis as any;

    if (!detectModel || !embedModel || !anchors) {
      frame.dispose();
      return;
    }

    try {
      const now      = Date.now();
      const pixels   = new Uint8Array(frame.getPixelBuffer());
      const orient   = frame.orientation;
      const isWrongDim = frame.width > frame.height;
      const frameW   = isWrongDim ? frame.height : frame.width;
      const frameH   = isWrongDim ? frame.width  : frame.height;
      const fmt      = frame.pixelFormat;
      const rowBytes = frame.bytesPerRow;
      const meta: ImageMeta = { width: frameW, height: frameH, orientation: orient };

      const lastPrev = globalObj._lastPreviewTime || 0;
      if (now - lastPrev < PREVIEW_INTERVAL_MS) return;
      globalObj._lastPreviewTime = now;

      // isLiveness and isCapturing come from the closure (correct values).
      let topFace: FaceBox | null = null;
      if (!globalObj._detectInput) globalObj._detectInput = new Float32Array(DETECT_SIZE * DETECT_SIZE * 3);

      try {
        resizeToFloat32(pixels, frameW, frameH, DETECT_SIZE, DETECT_SIZE, rowBytes, fmt, globalObj._detectInput);
        const detOutputs = detectModel.runSync([typedArrayToBuffer(globalObj._detectInput)]);
        const regressors = new Float32Array(detOutputs[0]);
        const scores     = new Float32Array(detOutputs[1]);
        const faces      = decodeFaces(regressors, scores, anchors);
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
          globalObj._smoothedFace = null;
        }
      } catch { /* detection error — skip frame */ }

      let promptStr: string | null = null;
      if (isLiveness && topFace) {
        let seqState = globalObj._enrollSeqState;
        // Always start a fresh sequence for a new liveness session.
        // Reset if null OR if previous session already passed.
        if (!seqState || seqState.passed) {
          seqState = generateLivenessSequence();
          globalObj._enrollSeqState = seqState;
        }
        if (topFace.landmarks) {
          const taskPassed = checkSequenceTask(topFace.landmarks as any, seqState);
          if (taskPassed) {
            seqState.currentTaskIndex++;
            seqState.framesInCurrentTask = 0;
            if (seqState.currentTaskIndex >= seqState.tasks.length) {
              seqState.passed = true;
              scheduleOnRN(onLivenessPassed);
            }
          }
        }
        promptStr = getStepPrompt(seqState);
        } else if (isLiveness && !topFace) {
          promptStr = 'position_face';
        }

        scheduleOnRN(onFaceDetected, topFace, meta, promptStr);

      if (isCapturing && topFace) {
        try {
          // Allocate reusable worklet-thread buffers on first use
          if (!globalObj._cropU8)    globalObj._cropU8    = new Uint8Array(EMBED_SIZE * EMBED_SIZE * 3);
          if (!globalObj._embedInput) globalObj._embedInput = new Float32Array(EMBED_SIZE * EMBED_SIZE * 3);

          const cropU8     = globalObj._cropU8     as Uint8Array;
          const embedInput = globalObj._embedInput as Float32Array;

          // Step 1: Align face into Uint8Array RGB buffer
          alignFaceToUint8(pixels, frameW, frameH, topFace.landmarks as Float32Array, rowBytes, fmt, cropU8);

          // Step 2: CLAHE adaptive contrast enhancement (in-place on cropU8)
          applyCLAHE(cropU8, EMBED_SIZE, EMBED_SIZE, 3, 0, 1, 2, cropU8);

          // Step 3: Normalise Uint8 → Float32 [-1, 1] for MobileFaceNet
          uint8ToFloat32Normalized(cropU8, embedInput);

          const embOutputs = embedModel.runSync([typedArrayToBuffer(embedInput)]);
          const rawEmb     = new Float32Array(embOutputs[0] as ArrayBuffer);
          scheduleOnRN(onEmbeddingCaptured, Array.from(l2Normalize(rawEmb)));
        } catch {
          scheduleOnRN(onWorkletError, 'Embedding failed — ensure face is clearly visible.');
        }
      }
    } finally {
      frame.dispose();
    }
  }, [detectModel, embedModel, anchors, isLiveness, isCapturing,
      onFaceDetected, onLivenessPassed, onEmbeddingCaptured, onWorkletError]);

  const frameOutput = useFrameOutput(useMemo(() => ({
    pixelFormat: 'rgb',
    enablePhysicalBufferRotation: true,
    onFrame,
  }), [onFrame]));

  const cameraOutputs = useMemo(() => [frameOutput], [frameOutput]);

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
    setLivenessPrompt(null);
    setEnrollState('liveness_check');
  }, [faceBox, detectModel, embedModel]);

  const handleSave = useCallback(async () => {
    if (!finalEmbedding || !personName.trim()) return;
    setEnrollState('saving');
    const name = personName.trim();
    try {
      const newId = await saveFace(name, finalEmbedding);
      logAuthEvent(newId, name, 'enroll', true);
      setShowNameInput(false);
      setPersonName('');
      embeddingsAccum.current = [];
      setCaptureCount(0);
      setFinalEmbedding(null);
      setLivenessPrompt(null);
      setEnrollState('idle');
      setModal({
        visible: true,
        icon: '🎉',
        title: 'Face Enrolled!',
        subtitle: `"${name}" has been registered successfully.`,
        confirmText: 'Done',
        onConfirm: () => setModal(m => ({ ...m, visible: false })),
      });
    } catch (e: any) {
      logAuthEvent(null, name, 'enroll', false);
      setEnrollState('error');
      setModal({
        visible: true,
        icon: '⚠️',
        title: 'Save Failed',
        subtitle: e?.message ?? 'An unexpected error occurred.',
        confirmText: 'OK',
        confirmDestructive: false,
        onConfirm: () => setModal(m => ({ ...m, visible: false })),
      });
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

  const handleRetryLiveness = useCallback(() => {
    runOnUI(() => {
      'worklet';
      (globalThis as any)._enrollSeqState = null;
    })();
  }, []);

  const isLoading    = detectLoading || embedLoading;
  const loadError    = detectError   || embedError;

  let statusMessage = '';
  if (isLoading) {
    statusMessage = t('enroll_loading');
  } else if (isCapturing) {
    statusMessage = `Capturing  ${captureCount} / ${CAPTURE_BURST}`;
  } else if (isLiveness) {
    if (!livenessPrompt) {
      statusMessage = 'Liveness check...';
    } else if (livenessPrompt === 'turn_left') {
      statusMessage = t('liveness_left');
    } else if (livenessPrompt === 'turn_right') {
      statusMessage = t('liveness_right');
    } else {
      statusMessage = t('liveness_left'); // fallback
    }
  } else if (faceBox) {
    statusMessage = '\u2713 Face detected — press Capture';
  } else {
    statusMessage = 'Scanning for face…';
  }

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
        <Text style={styles.statusText}>No camera found</Text>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      keyboardVerticalOffset={0}
    >
      <StatusBar barStyle="light-content" backgroundColor="#0a0c0f" />

      <View style={styles.header}>
        <View style={{ flex: 1, paddingRight: 10 }}>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
            <Text style={styles.screenTitle}>{t('enroll_title')}</Text>
          </View>
          <Text style={styles.subtitle}>{t('enroll_subtitle')}</Text>
        </View>
        <View style={{ flexDirection: 'row', gap: 10, alignItems: 'center' }}>
          <TouchableOpacity onPress={() => setLangModalVisible(true)} style={styles.settingsIcon}>
            <Settings size={20} color="#6b7280" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.flipBtn} onPress={() => setCameraPos(p => p === 'front' ? 'back' : 'front')}>
            <Text style={styles.flipBtnText}>{t('enroll_flip')}</Text>
          </TouchableOpacity>
          {isCapturing && (
            <View style={styles.modeBadge}>
              <Text style={styles.modeBadgeText}>{captureCount}/{CAPTURE_BURST}</Text>
            </View>
          )}
        </View>
      </View>

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
            <Text style={styles.loadingText}>{t('enroll_loading')}</Text>
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

        <Reanimated.View 
          style={styles.statusBanner}
        >
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 12 }}>
            <Text style={styles.statusBannerText} numberOfLines={1} adjustsFontSizeToFit>{statusMessage}</Text>
            {isLiveness && (
              <TouchableOpacity onPress={handleRetryLiveness} style={styles.livenessRetryBtn}>
                <Text style={styles.livenessRetryText}>↻</Text>
              </TouchableOpacity>
            )}
          </View>
        </Reanimated.View>
      </View>

      {!showNameInput && (
        <Reanimated.View 
          style={styles.controls}
        >
          <TouchableOpacity
            style={[styles.captureBtn, (!faceBox || isLoading || isCapturing || isLiveness) && styles.captureBtnDisabled]}
            onPress={handleCapture}
            disabled={!faceBox || isLoading || isCapturing || isLiveness}
            activeOpacity={0.75}
          >
            {isCapturing
              ? <ActivityIndicator size="small" color="#007AFF" />
              : <View style={styles.captureInner} />
            }
          </TouchableOpacity>
        </Reanimated.View>
      )}

      {showNameInput && (
        <Reanimated.View 
          style={styles.modalContainer}
        >
          <Text style={styles.modalTitle}>{t('enroll_who')}</Text>
          <Text style={styles.modalSub}>{t('enroll_enter_name')}</Text>
          <TextInput
            style={styles.textInput}
            placeholder="Full name…"
            placeholderTextColor="#3d4451"
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
              <Text style={styles.cancelBtnText}>{t('cancel')}</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.ctaButton, (!personName.trim() || enrollState === 'saving') && styles.saveBtnDisabled]}
              onPress={handleSave}
              disabled={!personName.trim() || enrollState === 'saving'}
              activeOpacity={0.8}
            >
              {enrollState === 'saving'
                ? <ActivityIndicator size="small" color="#000000" />
                : <Text style={styles.ctaText}>{t('enroll_save')}</Text>
              }
            </TouchableOpacity>
          </View>
        </Reanimated.View>
      )}

      <ConfirmModal
        visible={modal.visible}
        icon={modal.icon}
        title={modal.title}
        subtitle={modal.subtitle}
        confirmText={modal.confirmText}
        cancelText={modal.cancelText}
        confirmDestructive={modal.confirmDestructive}
        onConfirm={modal.onConfirm}
        onCancel={modal.onCancel ?? hideModal}
      />

      <LanguagePickerModal visible={langModalVisible} onClose={() => setLangModalVisible(false)} />
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  root:    { flex: 1, backgroundColor: '#0a0c0f' },
  centered: {
    flex: 1, backgroundColor: '#0a0c0f',
    alignItems: 'center', justifyContent: 'center', padding: 32,
  },

  header: {
    paddingTop:        Platform.OS === 'ios' ? 56 : 28,
    paddingHorizontal: 20,
    paddingBottom:     14,
    backgroundColor:   '#0d1117',
    borderBottomWidth: 1,
    borderBottomColor: '#1e2433',
    flexDirection:     'row',
    justifyContent:    'space-between',
    alignItems:        'center',
  },
  screenTitle: {
    fontFamily: 'BarlowCondensed-Bold',
    fontSize: 28,
    letterSpacing: 1.0,
    color: '#e8eaf0',
    textTransform: 'uppercase',
  },
  subtitle: {
    fontFamily: 'DMSans-Regular',
    fontSize: 13,
    color: '#6b7280',
    marginTop: 2,
  },
  settingsIcon: {
    padding: 6,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#1e2433',
  },
  modeBadge: {
    backgroundColor: 'rgba(0, 122, 255, 0.15)',
    borderWidth: 1,
    borderColor: 'rgba(0, 122, 255, 0.3)',
    borderRadius: 6,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  modeBadgeText: {
    fontFamily: 'DMSans-Bold',
    fontSize: 11,
    color: '#007AFF',
    textTransform: 'uppercase',
    letterSpacing: 1.0,
  },
  flipBtn: {
    paddingHorizontal: 10, paddingVertical: 4,
    backgroundColor: 'transparent',
    borderWidth: 1, borderColor: '#1e2433',
    borderRadius: 6,
  },
  flipBtnText: { fontFamily: 'DMSans-Bold', color: '#6b7280', fontSize: 11, letterSpacing: 1.0, textTransform: 'uppercase' },

  cameraContainer: { flex: 1, width: '100%', overflow: 'hidden', backgroundColor: '#000' },

  loadingOverlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(10,12,15,0.90)',
    alignItems:      'center',
    justifyContent:  'center',
    gap: 12,
  },
  loadingText: { fontFamily: 'DMSans-SemiBold', color: '#e8eaf0', fontSize: 16 },
  errorText:   { fontFamily: 'DMSans-Bold', color: '#FF3B30', fontSize: 14, textAlign: 'center', padding: 24 },

  progressContainer: {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    height: 3, backgroundColor: 'rgba(255,255,255,0.08)',
  },
  progressFill: { height: '100%', backgroundColor: '#34C759' },

  statusBanner: {
    position: 'absolute', top: 20, alignSelf: 'center',
    backgroundColor: 'rgba(28,28,30,0.85)',
    paddingVertical: 10, paddingHorizontal: 20,
    borderRadius: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
    elevation: 8,
    maxWidth: '90%',
  },
  statusBannerText: { fontFamily: 'DMSans-SemiBold', color: '#FFFFFF', fontSize: 14, textAlign: 'center', letterSpacing: 0.3, flexShrink: 1 },
  livenessRetryBtn: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 16,
  },
  livenessRetryText: {
    fontFamily: 'DMSans-Bold',
    color: '#FFFFFF',
    fontSize: 16,
  },

  controls: { 
    alignItems: 'center', 
    paddingVertical: 20, 
    backgroundColor: '#0a0c0f' 
  },
  captureBtn: {
    width: 72, height: 72, borderRadius: 36,
    borderWidth: 3.5, borderColor: '#007AFF',
    alignItems: 'center', justifyContent: 'center',
  },
  captureBtnDisabled: { borderColor: '#1e2433', opacity: 0.5 },
  captureInner: { width: 54, height: 54, borderRadius: 27, backgroundColor: '#007AFF' },

  modalContainer: {
    position: 'absolute',
    bottom: 0, left: 0, right: 0,
    backgroundColor: '#111419',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    borderTopWidth: 1,
    borderTopColor: '#1e2433',
    paddingHorizontal: 24,
    paddingTop: 20,
    paddingBottom: 40 + (Platform.OS === 'ios' ? 20 : 0),
  },
  modalTitle: {
    fontFamily: 'BarlowCondensed-Bold',
    fontSize: 22,
    color: '#e8eaf0',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
    marginBottom: 4,
  },
  modalSub: { fontFamily: 'DMSans-Regular', color: '#6b7280', fontSize: 14, marginBottom: 16 },
  textInput: {
    backgroundColor: '#161b22',
    borderWidth: 1,
    borderColor: '#2d3748',
    borderRadius: 10,
    paddingHorizontal: 16, paddingVertical: 14,
    fontFamily: 'DMSans-Regular',
    color: '#e8eaf0', fontSize: 16, marginBottom: 16,
  },
  nameActions: { flexDirection: 'row', gap: 12 },
  cancelBtn: {
    flex: 1, paddingVertical: 14, borderRadius: 10,
    backgroundColor: 'transparent', alignItems: 'center', justifyContent: 'center',
    borderWidth: 1, borderColor: '#1e2433',
  },
  cancelBtnText: { fontFamily: 'DMSans-Bold', color: '#6b7280', fontSize: 12, letterSpacing: 1.0, textTransform: 'uppercase' },
  ctaButton: { flex: 2, paddingVertical: 16, borderRadius: 10, backgroundColor: '#34C759', alignItems: 'center', justifyContent: 'center' },
  saveBtnDisabled: { backgroundColor: '#1e2433', opacity: 0.5 },
  ctaText: { fontFamily: 'DMSans-Bold', color: '#000000', fontSize: 16, letterSpacing: 0.3 },

  statusText: { fontFamily: 'DMSans-Regular', color: '#6b7280', fontSize: 15, marginBottom: 20, textAlign: 'center' },
  btn:        { backgroundColor: '#007AFF', paddingHorizontal: 28, paddingVertical: 14, borderRadius: 10 },
  btnText:    { fontFamily: 'DMSans-Bold', color: '#FFFFFF', fontSize: 16 },
});