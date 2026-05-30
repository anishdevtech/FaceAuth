/**
 * Frame preprocessing — worklet-safe pixel buffer transforms.
 *
 * All functions run on the VisionCamera frame-processor thread.
 *
 * Supported pixel formats (VisionCamera v4):
 *   'rgb'  — 3 bytes/px: R G B
 *   'bgra' — 4 bytes/px: B G R A  (default on Android)
 *   'rgba' — 4 bytes/px: R G B A
 */

import { clamp } from '../utils/mathUtils';

// ─── Pixel layout ─────────────────────────────────────────────────────────────

interface PixelLayout {
  bytesPerPixel: number;
  rOffset:       number;
  gOffset:       number;
  bOffset:       number;
}

/**
 * Resolves channel offsets from the VisionCamera pixel-format string.
 * Uses `startsWith()` to stay compatible across VisionCamera version variants
 * (e.g. 'bgra', 'bgra8888', 'bgra-8-bit').
 */
function resolvePixelLayout(pixelFormat: string): PixelLayout {
  'worklet';
  const fmt = pixelFormat.toLowerCase();

  if (fmt.startsWith('bgra')) {
    return { bytesPerPixel: 4, rOffset: 2, gOffset: 1, bOffset: 0 };
  }
  if (fmt.startsWith('rgba')) {
    return { bytesPerPixel: 4, rOffset: 0, gOffset: 1, bOffset: 2 };
  }
  if (fmt.startsWith('rgb')) {
    return { bytesPerPixel: 3, rOffset: 0, gOffset: 1, bOffset: 2 };
  }

  // Unknown format — assume BGRA (most common Android default)
  return { bytesPerPixel: 4, rOffset: 2, gOffset: 1, bOffset: 0 };
}

// ─── Buffer helpers ───────────────────────────────────────────────────────────

/**
 * Converts a TypedArray to a properly-sliced ArrayBuffer for
 * `model.runSync()`. The `.slice()` detaches the buffer as required by
 * react-native-fast-tflite v3+.
 */
export function typedArrayToBuffer(arr: Float32Array | Uint8Array): ArrayBuffer {
  'worklet';
  return (arr.buffer as ArrayBuffer).slice(arr.byteOffset, arr.byteOffset + arr.byteLength);
}

// ─── Resize ───────────────────────────────────────────────────────────────────

/**
 * Nearest-neighbour resize of a flat pixel buffer to `dstW × dstH`.
 *
 * Output: Float32Array in HWC layout, normalised to [0, 1] — the format
 * expected by BlazeFace (128×128×3 input).
 *
 * Pixel-centre sampling `(i + 0.5) / dst` maps destination pixels evenly
 * across the source extent, preventing the last row/column from never being
 * reached.
 *
 * @param pixels      Raw frame buffer from `frame.getPixelBuffer()`.
 * @param srcW        Logical frame width  (accounts for orientation).
 * @param srcH        Logical frame height (accounts for orientation).
 * @param dstW        Target width  (128 for BlazeFace).
 * @param dstH        Target height (128 for BlazeFace).
 * @param bytesPerRow Frame row stride in bytes (`frame.bytesPerRow`).
 * @param pixelFormat VisionCamera format string (`frame.pixelFormat`).
 */
export function resizeToFloat32(
  pixels:      Uint8Array,
  srcW:        number,
  srcH:        number,
  dstW:        number,
  dstH:        number,
  bytesPerRow: number,
  pixelFormat: string,
): Float32Array {
  'worklet';
  const { bytesPerPixel, rOffset, gOffset, bOffset } = resolvePixelLayout(pixelFormat);
  const output = new Float32Array(dstW * dstH * 3);

  for (let y = 0; y < dstH; y++) {
    const srcY       = clamp(Math.floor((y + 0.5) * srcH / dstH), 0, srcH - 1);
    const srcRowBase = srcY * bytesPerRow;

    for (let x = 0; x < dstW; x++) {
      const srcX   = clamp(Math.floor((x + 0.5) * srcW / dstW), 0, srcW - 1);
      const srcIdx = srcRowBase + srcX * bytesPerPixel;
      const dstIdx = (y * dstW + x) * 3;

      output[dstIdx    ] = pixels[srcIdx + rOffset] / 255;
      output[dstIdx + 1] = pixels[srcIdx + gOffset] / 255;
      output[dstIdx + 2] = pixels[srcIdx + bOffset] / 255;
    }
  }

  return output;
}

// ─── Face crop ────────────────────────────────────────────────────────────────

/**
 * Crops a padded face region from the full frame and resizes it to
 * `dstW × dstH`.
 *
 * Output: Float32Array in HWC layout, normalised to [−1, 1] — the format
 * expected by MobileFaceNet (112×112×3 input, pixel = (raw − 127.5) / 127.5).
 *
 * @param pixels      Raw frame buffer.
 * @param srcW        Logical frame width.
 * @param srcH        Logical frame height.
 * @param box         Normalised bounding box { xmin, ymin, xmax, ymax } in [0, 1].
 * @param dstW        Target width  (112 for MobileFaceNet).
 * @param dstH        Target height (112 for MobileFaceNet).
 * @param bytesPerRow Frame row stride in bytes.
 * @param pixelFormat VisionCamera format string.
 * @param padFactor   Extra padding fraction added around the box (default 20%).
 */
export function cropAndResizeFace(
  pixels:      Uint8Array,
  srcW:        number,
  srcH:        number,
  box: {
    ymin: number;
    xmin: number;
    ymax: number;
    xmax: number;
  },
  dstW:        number,
  dstH:        number,
  bytesPerRow: number,
  pixelFormat: string,
  padFactor    = 0.2,
): Float32Array {
  'worklet';
  const { bytesPerPixel, rOffset, gOffset, bOffset } = resolvePixelLayout(pixelFormat);

  const padX = (box.xmax - box.xmin) * padFactor;
  const padY = (box.ymax - box.ymin) * padFactor;

  const x1 = clamp(box.xmin - padX, 0, 1);
  const y1 = clamp(box.ymin - padY, 0, 1);
  const x2 = clamp(box.xmax + padX, 0, 1);
  const y2 = clamp(box.ymax + padY, 0, 1);

  const cropW = x2 - x1;
  const cropH = y2 - y1;

  // Guard against degenerate crops at frame edges
  if (cropW <= 0 || cropH <= 0) return new Float32Array(dstW * dstH * 3);

  const output = new Float32Array(dstW * dstH * 3);

  for (let y = 0; y < dstH; y++) {
    const normY      = y1 + ((y + 0.5) / dstH) * cropH;
    const srcY       = clamp(Math.floor(normY * srcH), 0, srcH - 1);
    const srcRowBase = srcY * bytesPerRow;

    for (let x = 0; x < dstW; x++) {
      const normX  = x1 + ((x + 0.5) / dstW) * cropW;
      const srcX   = clamp(Math.floor(normX * srcW), 0, srcW - 1);
      const srcIdx = srcRowBase + srcX * bytesPerPixel;
      const dstIdx = (y * dstW + x) * 3;

      output[dstIdx    ] = (pixels[srcIdx + rOffset] - 127.5) / 127.5;
      output[dstIdx + 1] = (pixels[srcIdx + gOffset] - 127.5) / 127.5;
      output[dstIdx + 2] = (pixels[srcIdx + bOffset] - 127.5) / 127.5;
    }
  }

  return output;
}