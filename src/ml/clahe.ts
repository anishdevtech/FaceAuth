// Implements Contrast Limited Adaptive Histogram Equalization (CLAHE).
// Enhances image contrast locally to improve face recognition performance under
// variable lighting conditions without over-amplifying image noise.
// This is a pure-JS implementation designed to execute safely within Reanimated worklets.

/** Number of tile divisions along each axis (8×8 = 64 tiles). */
const TILE_GRID = 8;

/**
 * Contrast clip limit. Higher = more contrast boost, but more noise.
 * 2.0 is the OpenCV default and works well for face crops.
 */
const CLIP_LIMIT = 2.0;

/**
 * Apply CLAHE to an RGB pixel buffer in-place.
 *
 * @param pixels  Uint8Array of RGB or RGBA pixels (only R/G/B channels are used).
 * @param width   Image width.
 * @param height  Image height.
 * @param bpp     Bytes per pixel (3 for RGB, 4 for RGBA/BGRA).
 * @param rOff    Offset of red channel within each pixel group.
 * @param gOff    Offset of green channel.
 * @param bOff    Offset of blue channel.
 * @param out     Output buffer (same layout as input). May alias `pixels` for in-place.
 */
export function applyCLAHE(
  pixels: Uint8Array,
  width:  number,
  height: number,
  bpp:    number,
  rOff:   number,
  gOff:   number,
  bOff:   number,
  out:    Uint8Array,
): void {
  'worklet';

  const tileW = Math.ceil(width  / TILE_GRID);
  const tileH = Math.ceil(height / TILE_GRID);

  // Phase 1: Construct contrast-limited Cumulative Distribution Function (CDF)
  // lookup tables for each grid tile.

  // Flatten tile LUTs into a single buffer: TILE_GRID * TILE_GRID * 256
  const totalTiles = TILE_GRID * TILE_GRID;
  const luts = new Uint8Array(totalTiles * 256);

  for (let ty = 0; ty < TILE_GRID; ty++) {
    const y0 = ty * tileH;
    const y1 = Math.min(y0 + tileH, height);

    for (let tx = 0; tx < TILE_GRID; tx++) {
      const x0 = tx * tileW;
      const x1 = Math.min(x0 + tileW, width);

      // Build luminance histogram for this tile
      const hist = new Uint32Array(256);
      let count = 0;

      for (let y = y0; y < y1; y++) {
        const rowBase = y * width * bpp;
        for (let x = x0; x < x1; x++) {
          const idx = rowBase + x * bpp;
          // BT.601 luminance (integer approximation for speed)
          const lum = (77 * pixels[idx + rOff] + 150 * pixels[idx + gOff] + 29 * pixels[idx + bOff]) >> 8;
          hist[lum]++;
          count++;
        }
      }

      if (count === 0) continue;

      // Clip histogram
      const clipVal = Math.max(1, Math.floor(CLIP_LIMIT * count / 256));
      let excess = 0;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > clipVal) {
          excess += hist[i] - clipVal;
          hist[i] = clipVal;
        }
      }

      // Redistribute excess evenly
      const redistrib = Math.floor(excess / 256);
      const remainder = excess - redistrib * 256;
      for (let i = 0; i < 256; i++) {
        hist[i] += redistrib + (i < remainder ? 1 : 0);
      }

      // Build CDF → LUT
      const lutOffset = (ty * TILE_GRID + tx) * 256;
      let cdf = 0;
      for (let i = 0; i < 256; i++) {
        cdf += hist[i];
        luts[lutOffset + i] = Math.min(255, Math.round((cdf * 255) / count));
      }
    }
  }

  // Phase 2: Apply the computed LUTs to each pixel by looking up the nearest tile's mapping.

  for (let y = 0; y < height; y++) {
    const ty = Math.min(Math.floor(y / tileH), TILE_GRID - 1);
    const rowBase = y * width * bpp;

    for (let x = 0; x < width; x++) {
      const tx = Math.min(Math.floor(x / tileW), TILE_GRID - 1);
      const idx = rowBase + x * bpp;

      const r = pixels[idx + rOff];
      const g = pixels[idx + gOff];
      const b = pixels[idx + bOff];

      // Original luminance
      const origL = (77 * r + 150 * g + 29 * b) >> 8;

      // CLAHE-mapped luminance
      const lutOffset = (ty * TILE_GRID + tx) * 256;
      const newL = luts[lutOffset + origL];

      // Scale RGB channels by the luminance ratio
      if (origL > 0) {
        const scale = newL / origL;
        out[idx + rOff] = Math.min(255, Math.round(r * scale));
        out[idx + gOff] = Math.min(255, Math.round(g * scale));
        out[idx + bOff] = Math.min(255, Math.round(b * scale));
      } else {
        out[idx + rOff] = newL;
        out[idx + gOff] = newL;
        out[idx + bOff] = newL;
      }

      // Copy alpha if present
      if (bpp === 4) {
        out[idx + 3] = pixels[idx + 3];
      }
    }
  }
}
