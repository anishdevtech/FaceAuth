// Provides mathematical operations optimized for execution on the VisionCamera frame-processor thread.
// All functions are decorated with the 'worklet' directive to guarantee isolated execution
// on the UI thread without synchronous bridging to the JavaScript runtime.



/** Maps an unbounded logit to a probability in [0, 1]. */
export function sigmoid(x: number): number {
  'worklet';
  return 1 / (1 + Math.exp(-x));
}

/** Clamps `x` to the closed interval [min, max]. */
export function clamp(x: number, min: number, max: number): number {
  'worklet';
  return Math.min(max, Math.max(min, x));
}



/**
 * Returns a new unit-length Float32Array.
 * If the input is a zero vector (norm < ε), returns zeros rather than NaN.
 */
export function l2Normalize(v: Float32Array): Float32Array {
  'worklet';
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);

  const out = new Float32Array(v.length);
  if (norm < 1e-10) return out;

  const invNorm = 1 / norm;
  for (let i = 0; i < v.length; i++) out[i] = v[i] * invNorm;
  return out;
}

/**
 * Computes the dot product of two uniform-length Float32Arrays.
 * When applied to L2-normalized vectors, the dot product mathematically equates to
 * the cosine similarity. This method is structurally prioritized in the critical inference
 * path to omit redundant norm recomputations.
 */
export function dotProduct(a: Float32Array, b: Float32Array): number {
  'worklet';
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/**
 * General cosine similarity for arbitrary (non-normalised) vectors.
 * Returns a value in [−1, 1].
 *
 * Prefer `dotProduct()` when both vectors are already L2-normalised.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  'worklet';
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}



export interface FaceMatch {
  id:         string;
  name:       string;
  similarity: number;
}

/**
 * Determines the highest-similarity match between a given query embedding and an array of enrolled faces.
 * Requires all input embeddings to be L2-normalized. Employs the dot product directly to calculate cosine
 * similarity for maximum evaluation efficiency.
 *
 * @param queryEmbedding  Unit-length Float32Array from the current frame.
 * @param storedFaces     Enrolled faces with unit-length embeddings.
 * @param threshold       Minimum similarity required to signify a positive match.
 */
export function bestMatch(
  queryEmbedding: Float32Array,
  storedFaces: { id: string; name: string; embedding: Float32Array }[],
  threshold = 0.65,
): FaceMatch | null {
  'worklet';
  let best: FaceMatch | null = null;

  for (let i = 0; i < storedFaces.length; i++) {
    const sim = dotProduct(queryEmbedding, storedFaces[i].embedding);
    if (sim >= threshold && (best === null || sim > best.similarity)) {
      best = { id: storedFaces[i].id, name: storedFaces[i].name, similarity: sim };
    }
  }

  return best;
}