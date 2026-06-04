// Handles the generation of anchor boxes and the decoding of model outputs for BlazeFace.
// Transforms raw tensor regressions into bounded face coordinate boxes and extracts 6-point facial landmarks.

import { sigmoid, clamp } from '../utils/mathUtils';

const INPUT_SIZE   = 128;
const NUM_ANCHORS  = 896;

/** Minimum face probability (post-sigmoid) to keep a candidate. */
const SCORE_THRESHOLD = 0.75;

/** IoU threshold for NMS clustering (BlazeFace reference uses 0.3). */
const IOU_THRESHOLD   = 0.3;

const STRIDES:            readonly number[] = [8, 16] as const;
const ANCHORS_PER_STRIDE: readonly number[] = [2,  6] as const;

export interface Anchor {
  cx: number;  // normalised [0, 1]
  cy: number;  // normalised [0, 1]
}

export interface FaceBox {
  xmin:       number;  // normalised [0, 1]
  ymin:       number;
  xmax:       number;
  ymax:       number;
  confidence: number;  // sigmoid probability [0, 1]
  landmarks?: Float32Array;  // 6 keypoints × 2 coords = 12 values
}

/**
 * Generates the 896 prior anchor boxes for the 128×128 BlazeFace input grid.
 *
 * Layout:
 *   stride  8 → 16×16 grid × 2 anchors = 512
 *   stride 16 →  8×8  grid × 6 anchors = 384
 *                                 total = 896 ✓
 *
 * Anchor centres are pixel-centred: `(i + 0.5) / gridSize`, normalised to [0, 1].
 * Pre-computed once at module load and reused across all frames.
 */
export function generateAnchors(): Anchor[] {
  const anchors: Anchor[] = [];
  anchors.length = NUM_ANCHORS;

  let idx = 0;
  for (let s = 0; s < STRIDES.length; s++) {
    const stride     = STRIDES[s];
    const numAnchors = ANCHORS_PER_STRIDE[s];
    const gridSize   = Math.ceil(INPUT_SIZE / stride);

    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        for (let k = 0; k < numAnchors; k++) {
          anchors[idx++] = {
            cx: (x + 0.5) / gridSize,
            cy: (y + 0.5) / gridSize,
          };
        }
      }
    }
  }
  return anchors;
}

/** 896 pre-computed anchors, shared across hooks and worklets. */
export const BLAZEFACE_ANCHORS: readonly Anchor[] = Object.freeze(generateAnchors());

// Computes the Intersection over Union (IoU) of two given bounding boxes.

function iou(a: FaceBox, b: FaceBox): number {
  'worklet';
  const y1 = Math.max(a.ymin, b.ymin);
  const x1 = Math.max(a.xmin, b.xmin);
  const y2 = Math.min(a.ymax, b.ymax);
  const x2 = Math.min(a.xmax, b.xmax);

  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const inter  = interW * interH;
  if (inter === 0) return 0;

  const areaA = (a.ymax - a.ymin) * (a.xmax - a.xmin);
  const areaB = (b.ymax - b.ymin) * (b.xmax - b.xmin);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

// Applies confidence-weighted Non-Maximum Suppression to eliminate overlapping bounding boxes.

/**
 * Confidence-weighted blending NMS, matching the BlazeFace reference
 * implementation. Sorts a copy of the input to avoid mutating the caller's array.
 */
export function nms(boxes: FaceBox[]): FaceBox[] {
  'worklet';
  const sorted     = [...boxes].sort((a, b) => b.confidence - a.confidence);
  const keep:      FaceBox[]  = [];
  const suppressed = new Uint8Array(sorted.length);

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed[i]) continue;

    const cluster: FaceBox[] = [sorted[i]];
    suppressed[i] = 1;

    for (let j = i + 1; j < sorted.length; j++) {
      if (!suppressed[j] && iou(sorted[i], sorted[j]) > IOU_THRESHOLD) {
        cluster.push(sorted[j]);
        suppressed[j] = 1;
      }
    }

    let totalWeight = 0, ymin = 0, xmin = 0, ymax = 0, xmax = 0;
    const landmarks = new Float32Array(12);

    for (const box of cluster) {
      totalWeight += box.confidence;
      ymin += box.ymin * box.confidence;
      xmin += box.xmin * box.confidence;
      ymax += box.ymax * box.confidence;
      xmax += box.xmax * box.confidence;
      
      if (box.landmarks) {
        for (let k = 0; k < 12; k++) {
          landmarks[k] += box.landmarks[k] * box.confidence;
        }
      }
    }

    for (let k = 0; k < 12; k++) {
      landmarks[k] /= totalWeight;
    }

    keep.push({
      ymin: ymin / totalWeight,
      xmin: xmin / totalWeight,
      ymax: ymax / totalWeight,
      xmax: xmax / totalWeight,
      confidence: sorted[i].confidence,
      landmarks: landmarks,
    });
  }

  return keep;
}

// ─── Decoder ──────────────────────────────────────────────────────────────────

/**
 * Decodes raw BlazeFace tensors into `FaceBox` objects.
 *
 * Regressor layout per anchor (16 values):
 *   [0]   dx — cx offset from anchor centre, in INPUT_SIZE units
 *   [1]   dy — cy offset from anchor centre, in INPUT_SIZE units
 *   [2]   dw — box width  in INPUT_SIZE units
 *   [3]   dh — box height in INPUT_SIZE units
 *   [4–15]  6 landmark keypoints (reserved)
 *
 * Score layout per anchor (1 value):
 *   [0]   raw logit — converted with sigmoid() before use
 *
 * @param regressors  Float32Array of shape [896 × 16]
 * @param scores      Float32Array of shape [896 × 1]
 * @param anchors     896 pre-computed anchor centres
 */
export function decodeFaces(
  regressors: Float32Array,
  scores:     Float32Array,
  anchors:    readonly Anchor[],
): FaceBox[] {
  'worklet';
  const candidates: FaceBox[] = [];

  for (let i = 0; i < anchors.length; i++) {
    const prob = sigmoid(scores[i]);
    if (prob < SCORE_THRESHOLD) continue;

    const a    = anchors[i];
    const base = i * 16;

    // TF convention: x-offset at [0], y-offset at [1]
    const cx = a.cx + regressors[base    ] / INPUT_SIZE;
    const cy = a.cy + regressors[base + 1] / INPUT_SIZE;
    const w  =        regressors[base + 2] / INPUT_SIZE;
    const h  =        regressors[base + 3] / INPUT_SIZE;
    
    const landmarks = new Float32Array(12);
    for (let k = 0; k < 6; k++) {
      landmarks[k * 2]     = a.cx + regressors[base + 4 + k * 2] / INPUT_SIZE;
      landmarks[k * 2 + 1] = a.cy + regressors[base + 5 + k * 2] / INPUT_SIZE;
    }

    candidates.push({
      xmin:       clamp(cx - w / 2, 0, 1),
      ymin:       clamp(cy - h / 2, 0, 1),
      xmax:       clamp(cx + w / 2, 0, 1),
      ymax:       clamp(cy + h / 2, 0, 1),
      confidence: prob,
      landmarks:  landmarks,
    });
  }

  return nms(candidates);
}