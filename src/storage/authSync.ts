// Manages offline event logging and synchronization queues for authentication events.
// Authentication attempts (both successful and failed) are persisted to an encrypted MMKV store
// and queued for deferred synchronization when network connectivity is restored.
import { createMMKV } from 'react-native-mmkv';

const syncStore = createMMKV({
  id: 'auth-sync-store',
  encryptionKey: 'faceauth-sync-aes256',
});

const QUEUE_KEY = 'auth_queue';

export interface AuthEvent {
  id: string; // UUID of event
  userId: string | null;
  timestamp: number;
  success: boolean;
  type: 'verify' | 'enroll';
  synced: boolean;
}

// Retrieves the current queue of authentication events from local storage.
function getQueue(): AuthEvent[] {
  const raw = syncStore.getString(QUEUE_KEY);
  if (!raw) return [];
  try {
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

function saveQueue(queue: AuthEvent[]) {
  syncStore.set(QUEUE_KEY, JSON.stringify(queue));
}

/**
 * Logs an authentication or enrollment attempt to the offline synchronization queue.
 * Generates a unique UUID and captures the timestamp of the event.
 */
export function logAuthEvent(userId: string | null, type: 'verify' | 'enroll', success: boolean) {
  const queue = getQueue();
  const event: AuthEvent = {
    id: `evt_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
    userId,
    timestamp: Date.now(),
    success,
    type,
    synced: false,
  };
  queue.push(event);
  saveQueue(queue);
  console.log(`[AuthSync] Logged ${type} event for ${userId || 'unknown'}: success=${success}`);
}

/**
 * Retrieves all events pending synchronization with the remote server.
 */
export function getUnsyncedEvents(): AuthEvent[] {
  return getQueue().filter(e => !e.synced);
}

/**
 * Marks specific event IDs as synchronized and purges them from the local queue to conserve storage.
 */
export function markEventsSynced(eventIds: string[]) {
  let queue = getQueue();
  const idSet = new Set(eventIds);
  
  // Update synced status
  queue = queue.map(e => {
    if (idSet.has(e.id)) {
      return { ...e, synced: true };
    }
    return e;
  });

  // Purge synced events to save space
  queue = queue.filter(e => !e.synced);
  
  saveQueue(queue);
  console.log(`[AuthSync] Purged ${eventIds.length} synced events.`);
}
