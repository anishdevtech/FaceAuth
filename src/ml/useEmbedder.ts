/**
 * useEmbedder — Loads the MobileFaceNet TFLite model.
 *
 * Platform notes:
 *   Android — model is copied from assets to DocumentDirectory on first run
 *             (or whenever the copy appears corrupt / mismatched in size).
 *   iOS     — model must be added to the Xcode target's "Copy Bundle Resources"
 *             phase; it is already on disk and needs no copying.
 *
 * The loaded model is disposed on hook unmount to prevent native heap leaks.
 */

import { useEffect, useRef, useState } from 'react';
import { Platform }                    from 'react-native';
import { loadTensorflowModel }         from 'react-native-fast-tflite';
import type { TfliteModel }            from 'react-native-fast-tflite/src/specs/Tflite.nitro';
import RNFS                            from 'react-native-fs';

// ─── Constants ────────────────────────────────────────────────────────────────

const MODEL_ASSET_NAME = 'models/mobile_face_net.tflite';
const MODEL_FILE_NAME  = 'mobile_face_net.tflite';

// ─── Platform-aware path resolution ──────────────────────────────────────────

function getModelDestPath(): string {
  return Platform.OS === 'ios'
    ? `${RNFS.MainBundlePath}/${MODEL_FILE_NAME}`
    : `${RNFS.DocumentDirectoryPath}/${MODEL_FILE_NAME}`;
}

/**
 * Ensures the model file is present and matches the expected asset size.
 *
 * A size mismatch indicates a partial write (e.g. killed mid-copy); the stale
 * file is removed and re-copied so the model never loads from a corrupt state.
 *
 * @returns Resolved filesystem path ready for `loadTensorflowModel`.
 */
async function prepareModelFile(): Promise<string> {
  const destPath = getModelDestPath();

  if (Platform.OS === 'android') {
    let needsCopy  = true;
    const exists   = await RNFS.exists(destPath);

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
        `[useEmbedder] Model not found in iOS bundle at: ${destPath}\n` +
        `Add "${MODEL_FILE_NAME}" to your Xcode target's "Copy Bundle Resources".`,
      );
    }
  }

  return destPath;
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

interface UseEmbedderResult {
  model:     TfliteModel | null;
  isLoading: boolean;
  error:     string | null;
}

export function useEmbedder(): UseEmbedderResult {
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

        console.log('[useEmbedder] Model loaded! Inputs:', JSON.stringify(loaded.inputs));
        console.log('[useEmbedder] Model loaded! Outputs:', JSON.stringify(loaded.outputs));

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

  return { model, isLoading, error };
}