// Manages the persistent storage and retrieval of enrolled facial embeddings.
// Implements an MMKV-backed storage layer with lazy initialization to gracefully handle
// instantiation errors and ensure the module does not fail at load time.

import { createMMKV } from 'react-native-mmkv';


let _storage: ReturnType<typeof createMMKV> | null = null;

function getStorage(): ReturnType<typeof createMMKV> {
  if (!_storage) {
    _storage = createMMKV({ id: 'faceauth-store-v2', encryptionKey: 'faceauth-biometric-aes256' });
  }
  return _storage;
}


const INDEX_KEY  = 'face_index_v2';
const FACE_PREFIX = 'face:';

/**
 * Embedding dimension expected from MobileFaceNet.
 * MobileFaceNet outputs 128-dimensional embeddings; update here if you swap
 * the backbone.
 */
export const EXPECTED_EMBEDDING_DIM = 128;


interface StoredFace {
  id:          string;
  name:        string;
  embedding:   number[];  // Float32Array serialised as a plain number[] for JSON
  enrolledAt:  number;
}


function readIndex(): string[] {
  const raw = getStorage().getString(INDEX_KEY);
  if (!raw) return [];
  try   { return JSON.parse(raw) as string[]; }
  catch { return []; }
}

function writeIndex(ids: string[]): void {
  getStorage().set(INDEX_KEY, JSON.stringify(ids));
}

function readFace(id: string): StoredFace | null {
  const raw = getStorage().getString(`${FACE_PREFIX}${id}`);
  if (!raw) return null;
  try   { return JSON.parse(raw) as StoredFace; }
  catch { return null; }
}

function writeFace(face: StoredFace): void {
  getStorage().set(`${FACE_PREFIX}${face.id}`, JSON.stringify(face));
}

function deleteFaceRecord(id: string): void {
  getStorage().remove(`${FACE_PREFIX}${id}`);
}


export interface EnrolledFace {
  id:          string;
  name:        string;
  embedding:   Float32Array;
  enrolledAt:  number;
}


/**
 * Persists a new face embedding or updates an existing identity.
 * Performs a case-insensitive lookup by name. If an identity exists, their
 * embedding is updated instead of creating a duplicate that would degrade match scoring.
 *
 * @returns The unique identifier of the face record.
 */
export async function saveFace(name: string, embedding: Float32Array): Promise<string> {
  const trimmedName = name.trim();
  const ids         = readIndex();

  const existingId = ids.find(id => {
    const face = readFace(id);
    return face?.name.toLowerCase() === trimmedName.toLowerCase();
  });

  if (existingId) {
    writeFace({
      id:         existingId,
      name:       trimmedName,
      embedding:  Array.from(embedding),
      enrolledAt: Date.now(),
    });
    return existingId;
  }

  const id = `f_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
  writeFace({ id, name: trimmedName, embedding: Array.from(embedding), enrolledAt: Date.now() });
  writeIndex([...ids, id]);
  return id;
}

/**
 * Retrieves all enrolled faces from storage and decodes their embeddings as Float32Arrays.
 * Invalid entries with mismatched embedding dimensions are skipped automatically.
 */
export function getAllFaces(): EnrolledFace[] {
  const ids:   string[]       = readIndex();
  const faces: EnrolledFace[] = [];

  for (const id of ids) {
    const stored = readFace(id);
    if (!stored) continue;

    if (stored.embedding.length !== EXPECTED_EMBEDDING_DIM) {
      console.warn(
        `[faceStore] Skipping "${stored.name}" (${id}): ` +
        `embedding length ${stored.embedding.length} ≠ ${EXPECTED_EMBEDDING_DIM}. ` +
        `Re-enroll this person.`,
      );
      continue;
    }

    faces.push({
      id:         stored.id,
      name:       stored.name,
      embedding:  new Float32Array(stored.embedding),
      enrolledAt: stored.enrolledAt,
    });
  }

  return faces;
}

/** Deletes a single enrolled face by ID. */
export function deleteFace(id: string): void {
  deleteFaceRecord(id);
  writeIndex(readIndex().filter(fid => fid !== id));
}

/** Removes every enrolled face and resets the index. */
export function clearAllFaces(): void {
  const ids = readIndex();
  for (const id of ids) deleteFaceRecord(id);
  getStorage().remove(INDEX_KEY);
}

/** Returns the number of currently enrolled identities. */
export function getFaceCount(): number {
  return readIndex().length;
}

/**
 * Returns `true` if a face with this name is already enrolled.
 * Useful for showing a confirmation before overwriting.
 */
export function isNameEnrolled(name: string): boolean {
  const lower = name.trim().toLowerCase();
  return readIndex().some(id => readFace(id)?.name.toLowerCase() === lower);
}