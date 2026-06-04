// Manages the loading, initialization, and lifecycle of the BlazeFace short-range TFLite model.
// Implements platform-specific asset extraction (Android) and bundle resolution (iOS).
// Ensures the model is safely disposed on unmount to prevent native memory leaks.
// Exposes pre-computed anchor boxes alongside the model for immediate inference utilization.

import { useEffect, useRef, useState }    from 'react';
import { Platform }                        from 'react-native';
import { loadTensorflowModel }             from 'react-native-fast-tflite';
import type { TfliteModel }                from 'react-native-fast-tflite/src/specs/Tflite.nitro';
import RNFS                                from 'react-native-fs';
import { BLAZEFACE_ANCHORS, type Anchor } from './blazeface';

const MODEL_ASSET_NAME = 'models/blaze_face_short_range.tflite';
const MODEL_FILE_NAME  = 'blaze_face_short_range.tflite';

function getModelDestPath(): string {
  return Platform.OS === 'ios'
    ? `${RNFS.MainBundlePath}/${MODEL_FILE_NAME}`
    : `${RNFS.DocumentDirectoryPath}/${MODEL_FILE_NAME}`;
}

/**
 * Ensures the model file is present on the device filesystem and matches the expected size.
 * Validates file integrity against partial writes during Android asset extraction.
 *
 * @returns Resolved absolute filesystem path ready for tensor inference.
 */
async function prepareModelFile(): Promise<string> {
  const destPath = getModelDestPath();

  if (Platform.OS === 'android') {
    let needsCopy = true;
    const exists  = await RNFS.exists(destPath);

    if (exists) {
      try {
        const [destStat, assetStat] = await Promise.all([
          RNFS.stat(destPath),
          RNFS.stat(`assets://${MODEL_ASSET_NAME}`).catch(() => null),
        ]);
        if (!assetStat || destStat.size === assetStat.size) {
          needsCopy = false;
        } else {
          await RNFS.unlink(destPath);
        }
      } catch {
        needsCopy = true;
      }
    }

    if (needsCopy) {
      await RNFS.copyFileAssets(MODEL_ASSET_NAME, destPath);
    }
  } else {
    const exists = await RNFS.exists(destPath);
    if (!exists) {
      throw new Error(
        `[useFaceDetector] Model not found in iOS bundle at: ${destPath}\n` +
        `Add "${MODEL_FILE_NAME}" to your Xcode target's "Copy Bundle Resources".`,
      );
    }
  }

  return destPath;
}

interface UseFaceDetectorResult {
  model:     TfliteModel | null;
  anchors:   readonly Anchor[];
  isLoading: boolean;
  error:     string | null;
}

export function useFaceDetector(): UseFaceDetectorResult {
  const [model,     setModel]   = useState<TfliteModel | null>(null);
  const [isLoading, setLoading] = useState(true);
  const [error,     setError]   = useState<string | null>(null);

  const modelRef = useRef<TfliteModel | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const destPath = await prepareModelFile();
        const loaded   = await loadTensorflowModel({ url: `file://${destPath}` }, []);

        if (cancelled) {
          loaded.dispose?.();
          return;
        }

        modelRef.current = loaded;
        setModel(loaded);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      modelRef.current?.dispose?.();
      modelRef.current = null;
    };
  }, []);

  return {
    model,
    // BLAZEFACE_ANCHORS is Object.freeze'd at module scope — safe to pass
    // directly to worklets (read-only, never mutated at runtime).
    anchors: BLAZEFACE_ANCHORS,
    isLoading,
    error,
  };
}