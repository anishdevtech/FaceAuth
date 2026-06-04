// Implements image quality assessment utilizing Laplacian Variance for blur detection.
// Low-variance (blurry) face crops yield unreliable embeddings, which may cause false negatives during verification.
// Executed post-detection to reject out-of-focus frames before computationally expensive embedding generation.

// Threshold for acceptable image sharpness based on empirical variance.
// Configured to reject blurry captures from mid-range mobile sensors.
const SHARPNESS_THRESHOLD = 80;

// Computes the Laplacian variance (sharpness metric) of a given grayscale image buffer.
// Applies a 3x3 Laplacian convolution kernel over the image to measure edge density.
/**
 * @param pixels  RGB pixel buffer (Uint8Array).
 * @param w       Image width.
 * @param h       Image height.
 * @param bpp     Bytes per pixel (3 for RGB, 4 for RGBA/BGRA).
 * @param rOff    Red channel offset.
 * @param gOff    Green channel offset.
 * @param bOff    Blue channel offset.
 * @returns       Laplacian variance (higher = sharper).
 */
export function computeSharpness(
  pixels: Uint8Array,
  w:      number,
  h:      number,
  bpp:    number,
  rOff:   number,
  gOff:   number,
  bOff:   number,
): number {
  'worklet';

  let sum   = 0;
  let sumSq = 0;
  let count = 0;

  // Skip border pixels (Laplacian needs 1px margin)
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      // Inline grayscale for centre and 4 neighbours
      const cIdx = (y * w + x) * bpp;
      const cGray = (77 * pixels[cIdx + rOff] + 150 * pixels[cIdx + gOff] + 29 * pixels[cIdx + bOff]) >> 8;

      const tIdx = ((y - 1) * w + x) * bpp;
      const tGray = (77 * pixels[tIdx + rOff] + 150 * pixels[tIdx + gOff] + 29 * pixels[tIdx + bOff]) >> 8;

      const bIdx2 = ((y + 1) * w + x) * bpp;
      const bGray = (77 * pixels[bIdx2 + rOff] + 150 * pixels[bIdx2 + gOff] + 29 * pixels[bIdx2 + bOff]) >> 8;

      const lIdx = (y * w + (x - 1)) * bpp;
      const lGray = (77 * pixels[lIdx + rOff] + 150 * pixels[lIdx + gOff] + 29 * pixels[lIdx + bOff]) >> 8;

      const rIdx = (y * w + (x + 1)) * bpp;
      const rGray = (77 * pixels[rIdx + rOff] + 150 * pixels[rIdx + gOff] + 29 * pixels[rIdx + bOff]) >> 8;

      const val = -4 * cGray + tGray + bGray + lGray + rGray;
      sum   += val;
      sumSq += val * val;
      count++;
    }
  }

  if (count === 0) return 0;
  const mean = sum / count;
  return sumSq / count - mean * mean; // variance
}

// Evaluates whether the face crop exceeds the minimum sharpness threshold required for embedding generation.
export function isFaceSharp(
  pixels: Uint8Array,
  w:      number,
  h:      number,
  bpp:    number,
  rOff:   number,
  gOff:   number,
  bOff:   number,
): boolean {
  'worklet';
  return computeSharpness(pixels, w, h, bpp, rOff, gOff, bOff) >= SHARPNESS_THRESHOLD;
}
