/**
 * useFaceDetector — Loads the BlazeFace short-range TFLite model.
 *
 * Platform notes:
 *   Android — model is copied from assets to DocumentDirectory on first run
 *             (or when the on-disk copy is corrupt / size-mismatched).
 *   iOS     — model must be added to the Xcode target's "Copy Bundle Resources"
 *             phase; it is already on disk and needs no copying.
 *
 * The loaded model is disposed on hook unmount to prevent native heap leaks.
 * Pre-computed BlazeFace anchors are returned alongside the model so callers
 * receive everything needed to run inference in one hook call.
 */

import { useEffect, useRef, useState }    from 'react';
import { Platform }                        from 'react-native';
import { loadTensorflowModel }             from 'react-native-fast-tflite';
import type { TfliteModel }                from 'react-native-fast-tflite/src/specs/Tflite.nitro';
import RNFS                                from 'react-native-fs';
import { BLAZEFACE_ANCHORS, type Anchor } from './blazeface';

// ─── Constants ────────────────────────────────────────────────────────────────

const MODEL_ASSET_NAME = 'models/blaze_face_short_range.tflite';
const MODEL_FILE_NAME  = 'blaze_face_short_range.tflite';

// ─── Platform-aware path resolution ──────────────────────────────────────────

function getModelDestPath(): string {
  return Platform.OS === 'ios'
    ? `${RNFS.MainBundlePath}/${MODEL_FILE_NAME}`
    : `${RNFS.DocumentDirectoryPath}/${MODEL_FILE_NAME}`;
}

/**
 * Ensures the model file is present and matches the expected asset size.
 * Identical integrity-check strategy as `useEmbedder`.
 *
 * @returns Resolved filesystem path ready for `loadTensorflowModel`.
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

// ─── Hook ─────────────────────────────────────────────────────────────────────

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