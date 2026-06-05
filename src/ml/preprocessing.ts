// Provides frame preprocessing utilities optimized for execution on the VisionCamera frame-processor thread (worklets).
// Responsible for parsing pixel layouts, resizing buffers, and cropping face regions for machine learning models.
// Supported pixel formats include RGB, BGRA, and RGBA.

import { clamp } from '../utils/mathUtils';

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
export function resolvePixelLayout(pixelFormat: string): PixelLayout {
  'worklet';
  const format = pixelFormat.toLowerCase();

  if (format.startsWith('bgra')) {
    return { bytesPerPixel: 4, rOffset: 2, gOffset: 1, bOffset: 0 };
  }
  if (format.startsWith('rgba')) {
    return { bytesPerPixel: 4, rOffset: 0, gOffset: 1, bOffset: 2 };
  }
  if (format.startsWith('rgb') || format.startsWith('yuv')) {
    return { bytesPerPixel: 3, rOffset: 0, gOffset: 1, bOffset: 2 };
  }
  if (format.startsWith('bgr')) {
    return { bytesPerPixel: 3, rOffset: 2, gOffset: 1, bOffset: 0 };
  }
  
  // Default fallback for unknown formats is BGRA, the standard Android default.
  return { bytesPerPixel: 4, rOffset: 2, gOffset: 1, bOffset: 0 };
}

/**
 * Converts a TypedArray to a properly-sliced ArrayBuffer for
 * `model.runSync()`. The `.slice()` detaches the buffer as required by
 * react-native-fast-tflite v3+.
 */
export function typedArrayToBuffer(arr: Float32Array | Uint8Array): ArrayBuffer {
  'worklet';
  return arr.slice().buffer as ArrayBuffer;
}

/**
 * Nearest-neighbour resize of a flat pixel buffer to `dstW Ã— dstH`.
 *
 * Output: Float32Array in HWC layout, normalised to [0, 1] â€” the format
 * expected by BlazeFace (128Ã—128Ã—3 input).
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
  outBuffer:   Float32Array,
): Float32Array {
  'worklet';
  const { bytesPerPixel, rOffset, gOffset, bOffset } = resolvePixelLayout(pixelFormat);
  
  // Cache LUT for the inner loop since dstW and srcW are constant per model
  const globalObj = globalThis as any;
  if (globalObj._lut_srcW !== srcW || globalObj._lut_dstW !== dstW || !globalObj._lutX) {
    globalObj._lut_srcW = srcW;
    globalObj._lut_dstW = dstW;
    const lut = new Float32Array(dstW);
    for (let x = 0; x < dstW; x++) lut[x] = clamp(Math.floor((x + 0.5) * srcW / dstW), 0, srcW - 1);
    globalObj._lutX = lut;
  }
  const lutX: Float32Array = globalObj._lutX;

  for (let y = 0; y < dstH; y++) {
    const srcY       = clamp(Math.floor((y + 0.5) * srcH / dstH), 0, srcH - 1);
    const srcRowBase = srcY * bytesPerRow;
    const dstRowBase = y * dstW * 3;

    for (let x = 0; x < dstW; x++) {
      const srcIdx = srcRowBase + lutX[x] * bytesPerPixel;
      const dstIdx = dstRowBase + x * 3;

      outBuffer[dstIdx    ] = pixels[srcIdx + rOffset] / 255;
      outBuffer[dstIdx + 1] = pixels[srcIdx + gOffset] / 255;
      outBuffer[dstIdx + 2] = pixels[srcIdx + bOffset] / 255;
    }
  }

  return outBuffer;
}

/**
 * Crops a padded face region from the full frame and resizes it to
 * `dstW Ã— dstH`.
 *
 * Output: Float32Array in HWC layout, normalised to [âˆ’1, 1] â€” the format
 * expected by MobileFaceNet (112Ã—112Ã—3 input, pixel = (raw âˆ’ 127.5) / 127.5).
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
  outBuffer:   Float32Array,
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
  if (cropW <= 0 || cropH <= 0) {
    outBuffer.fill(0);
    return outBuffer;
  }

  // Precompute X multipliers to hoist math out of inner loop
  const globalObj = globalThis as any;
  if (globalObj._crop_dstW !== dstW || !globalObj._crop_lutX) {
    globalObj._crop_dstW = dstW;
    const lut = new Float32Array(dstW);
    for (let x = 0; x < dstW; x++) lut[x] = (x + 0.5) / dstW;
    globalObj._crop_lutX = lut;
  }
  const lutX: Float32Array = globalObj._crop_lutX;
  const baseX = x1 * srcW;
  const scaleX = cropW * srcW;

  for (let y = 0; y < dstH; y++) {
    const normY      = y1 + ((y + 0.5) / dstH) * cropH;
    const srcY       = clamp(Math.floor(normY * srcH), 0, srcH - 1);
    const srcRowBase = srcY * bytesPerRow;
    const dstRowBase = y * dstW * 3;

    for (let x = 0; x < dstW; x++) {
      const srcX   = clamp(Math.floor(baseX + lutX[x] * scaleX), 0, srcW - 1);
      const srcIdx = srcRowBase + srcX * bytesPerPixel;
      const dstIdx = dstRowBase + x * 3;

      outBuffer[dstIdx    ] = (pixels[srcIdx + rOffset] - 127.5) / 127.5;
      outBuffer[dstIdx + 1] = (pixels[srcIdx + gOffset] - 127.5) / 127.5;
      outBuffer[dstIdx + 2] = (pixels[srcIdx + bOffset] - 127.5) / 127.5;
    }
  }

  return outBuffer;
}

/**
 * Crops a padded face region from the full frame and resizes it into a
 * Uint8Array with packed RGB (3 bpp) layout â€” the format expected by
 * `applyCLAHE()`.
 *
 * This is deliberately separate from `cropAndResizeFace` so the existing
 * float32 path remains untouched for callers that don't need CLAHE.
 */
export function cropFaceToUint8(
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
  outBuffer:   Uint8Array,
  padFactor    = 0.2,
): Uint8Array {
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

  if (cropW <= 0 || cropH <= 0) {
    outBuffer.fill(0);
    return outBuffer;
  }

  const globalObj = globalThis as any;
  if (globalObj._crop_u8_dstW !== dstW || !globalObj._crop_u8_lutX) {
    globalObj._crop_u8_dstW = dstW;
    const lut = new Float32Array(dstW);
    for (let x = 0; x < dstW; x++) lut[x] = (x + 0.5) / dstW;
    globalObj._crop_u8_lutX = lut;
  }
  const lutX: Float32Array = globalObj._crop_u8_lutX;
  const baseX  = x1 * srcW;
  const scaleX = cropW * srcW;

  for (let y = 0; y < dstH; y++) {
    const normY      = y1 + ((y + 0.5) / dstH) * cropH;
    const srcY       = clamp(Math.floor(normY * srcH), 0, srcH - 1);
    const srcRowBase = srcY * bytesPerRow;
    const dstRowBase = y * dstW * 3;  // always RGB = 3 bpp

    for (let x = 0; x < dstW; x++) {
      const srcX   = clamp(Math.floor(baseX + lutX[x] * scaleX), 0, srcW - 1);
      const srcIdx = srcRowBase + srcX * bytesPerPixel;
      const dstIdx = dstRowBase + x * 3;

      outBuffer[dstIdx    ] = pixels[srcIdx + rOffset];
      outBuffer[dstIdx + 1] = pixels[srcIdx + gOffset];
      outBuffer[dstIdx + 2] = pixels[srcIdx + bOffset];
    }
  }

  return outBuffer;
}

/**
 * Extracts a 112x112 face image aligned exactly to the MobileFaceNet/ArcFace
 * standard coordinates using a Similarity Transform (Affine Transformation).
 *
 * It takes the left and right eye landmarks, computes the inverse affine matrix,
 * and uses Bilinear Interpolation to sample the aligned pixels directly from
 * the raw camera frame.
 *
 * @param pixels      Raw frame buffer.
 * @param srcW        Logical frame width.
 * @param srcH        Logical frame height.
 * @param landmarks   Normalized [0..1] array of 12 floats from BlazeFace.
 * @param bytesPerRow Frame row stride in bytes.
 * @param pixelFormat VisionCamera format string.
 * @param outBuffer   Pre-allocated Uint8Array(112 * 112 * 3).
 */
export function alignFaceToUint8(
  pixels:      Uint8Array,
  srcW:        number,
  srcH:        number,
  landmarks:   Float32Array,
  bytesPerRow: number,
  pixelFormat: string,
  outBuffer:   Uint8Array,
): Uint8Array {
  'worklet';
  const { bytesPerPixel, rOffset, gOffset, bOffset } = resolvePixelLayout(pixelFormat);

  const REYE = 0;
  const LEYE = 1;

  // 1. Identify which eye is on the left side of the image
  let p1x, p1y, p2x, p2y;
  if (landmarks[REYE * 2] < landmarks[LEYE * 2]) {
    p1x = landmarks[REYE * 2] * srcW;
    p1y = landmarks[REYE * 2 + 1] * srcH;
    p2x = landmarks[LEYE * 2] * srcW;
    p2y = landmarks[LEYE * 2 + 1] * srcH;
  } else {
    p1x = landmarks[LEYE * 2] * srcW;
    p1y = landmarks[LEYE * 2 + 1] * srcH;
    p2x = landmarks[REYE * 2] * srcW;
    p2y = landmarks[REYE * 2 + 1] * srcH;
  }

  // 2. MobileFaceNet standard coordinates for 112x112
  const dstW = 112;
  const dstH = 112;
  const dx1 = 38.29;
  const dx2 = 73.53;
  const dy  = 51.69;

  // 3. Compute Inverse Similarity Transform Matrix components
  const d_dx = dx2 - dx1;
  const a = (p2x - p1x) / d_dx;
  const b = (p2y - p1y) / d_dx;

  const tx = p1x - a * dx1 + b * dy;
  const ty = p1y - b * dx1 - a * dy;

  // 4. Bilinear sampling from source image
  for (let y = 0; y < dstH; y++) {
    const sx_row = -b * y + tx;
    const sy_row =  a * y + ty;
    const dstRowBase = y * dstW * 3;

    for (let x = 0; x < dstW; x++) {
      const sx = a * x + sx_row;
      const sy = b * x + sy_row;

      const sx_i = Math.floor(sx);
      const sy_i = Math.floor(sy);

      const px = sx - sx_i;
      const py = sy - sy_i;
      const pxInv = 1 - px;
      const pyInv = 1 - py;

      const x1 = clamp(sx_i, 0, srcW - 1);
      const y1 = clamp(sy_i, 0, srcH - 1);
      const x2 = clamp(sx_i + 1, 0, srcW - 1);
      const y2 = clamp(sy_i + 1, 0, srcH - 1);

      const row1 = y1 * bytesPerRow;
      const row2 = y2 * bytesPerRow;
      const idx11 = row1 + x1 * bytesPerPixel;
      const idx12 = row1 + x2 * bytesPerPixel;
      const idx21 = row2 + x1 * bytesPerPixel;
      const idx22 = row2 + x2 * bytesPerPixel;

      const baseDst = dstRowBase + x * 3;
      
      // Interpolate R
      outBuffer[baseDst    ] = Math.round(
        pixels[idx11 + rOffset] * pxInv * pyInv +
        pixels[idx12 + rOffset] * px    * pyInv +
        pixels[idx21 + rOffset] * pxInv * py +
        pixels[idx22 + rOffset] * px    * py
      );
      
      // Interpolate G
      outBuffer[baseDst + 1] = Math.round(
        pixels[idx11 + gOffset] * pxInv * pyInv +
        pixels[idx12 + gOffset] * px    * pyInv +
        pixels[idx21 + gOffset] * pxInv * py +
        pixels[idx22 + gOffset] * px    * py
      );
      
      // Interpolate B
      outBuffer[baseDst + 2] = Math.round(
        pixels[idx11 + bOffset] * pxInv * pyInv +
        pixels[idx12 + bOffset] * px    * pyInv +
        pixels[idx21 + bOffset] * pxInv * py +
        pixels[idx22 + bOffset] * px    * py
      );
    }
  }

  return outBuffer;
}

/**
 * Converts a Uint8Array RGB buffer (e.g. CLAHE output) into a Float32Array
 * normalised to [âˆ’1, 1] for MobileFaceNet input.
 *
 * @param src   Uint8Array of packed RGB pixels (3 bpp, no alpha).
 * @param out   Pre-allocated Float32Array of the same pixel count Ã— 3.
 */
export function uint8ToFloat32Normalized(
  src: Uint8Array,
  out: Float32Array,
): Float32Array {
  'worklet';
  const len = src.length;  // width * height * 3
  for (let i = 0; i < len; i++) {
    out[i] = (src[i] - 127.5) / 127.5;
  }
  return out;
}

