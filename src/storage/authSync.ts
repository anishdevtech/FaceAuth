// Manages offline event logging and synchronization queues for authentication events.
// Authentication attempts (both successful and failed) are persisted to an encrypted MMKV store
// and queued for deferred synchronization when network connectivity is restored.
import { createMMKV } from 'react-native-mmkv';

const syncStore = createMMKV({
  id: 'auth-sync-store',
  encryptionKey: 'faceauth-sync-aes256',
});

const HEAD_KEY = 'auth_q_head';
const TAIL_KEY = 'auth_q_tail';
const ITEM_PREFIX = 'auth_q_item_';
const LEGACY_QUEUE_KEY = 'auth_queue';

export interface AuthEvent {
  id: string; // UUID of event
  userId: string | null;
  name: string | null;
  timestamp: number;
  success: boolean;
  type: 'verify' | 'enroll';
  synced: boolean;
  _queueIndex?: number; // Internal tracking
}

function getHead(): number {
  return syncStore.getNumber(HEAD_KEY) || 0;
}

function getTail(): number {
  return syncStore.getNumber(TAIL_KEY) || 0;
}

// Ensure old O(N^2) queue items are migrated to the new O(1) ring buffer on startup
let _authMigrated = false;
function migrateLegacyAuthQueue() {
  if (_authMigrated) return;
  _authMigrated = true;
  
  const raw = syncStore.getString(LEGACY_QUEUE_KEY);
  if (!raw) return;
  
  try {
    const oldEvents = JSON.parse(raw) as AuthEvent[];
    const unsynced = oldEvents.filter(e => !e.synced);
    let tail = getTail();
    
    for (const event of unsynced) {
      syncStore.set(`${ITEM_PREFIX}${tail}`, JSON.stringify({ ...event, _queueIndex: tail }));
      tail++;
    }
    
    syncStore.set(TAIL_KEY, tail);
    console.log(`[AuthSync] Migrated ${unsynced.length} legacy events to O(1) queue.`);
  } catch (err) {
    console.warn('[AuthSync] Failed to migrate legacy auth queue', err);
  } finally {
    // Delete the legacy key so we don't migrate twice
    syncStore.remove(LEGACY_QUEUE_KEY);
  }
}

/**
 * Logs an authentication or enrollment attempt to the offline synchronization queue.
 * Generates a unique UUID and captures the timestamp of the event.
 */
export function logAuthEvent(userId: string | null, name: string | null, type: 'verify' | 'enroll', success: boolean) {
  migrateLegacyAuthQueue();
  
  const tail = getTail();
  const event: AuthEvent = {
    id: `evt_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`,
    userId,
    name,
    timestamp: Date.now(),
    success,
    type,
    synced: false,
    _queueIndex: tail,
  };
  
  syncStore.set(`${ITEM_PREFIX}${tail}`, JSON.stringify(event));
  syncStore.set(TAIL_KEY, tail + 1);
  console.log(`[AuthSync] Logged ${type} event for ${userId || 'unknown'}: success=${success}`);
}

/**
 * Retrieves all events pending synchronization with the remote server.
 */
export function getUnsyncedEvents(): AuthEvent[] {
  migrateLegacyAuthQueue();
  
  const head = getHead();
  const tail = getTail();
  const events: AuthEvent[] = [];
  
  for (let i = head; i < tail; i++) {
    const raw = syncStore.getString(`${ITEM_PREFIX}${i}`);
    if (raw) {
      try {
        events.push(JSON.parse(raw));
      } catch {}
    }
  }
  return events;
}

/**
 * Retrieves all events (both synced and unsynced) from the local storage.
 * Useful for the Attendance logs UI.
 */
export function getAllEvents(): AuthEvent[] {
  return getUnsyncedEvents();
}

/**
 * Marks specific event IDs as synchronized and purges them from the local queue to conserve storage.
 * Runs in O(K) time where K is the number of successfully synced events.
 */
export function markEventsSynced(eventIds: string[]) {
  let head = getHead();
  const tail = getTail();
  const idSet = new Set(eventIds);
  
  for (let i = head; i < tail; i++) {
    const raw = syncStore.getString(`${ITEM_PREFIX}${i}`);
    if (raw) {
      try {
        const evt = JSON.parse(raw) as AuthEvent;
        if (idSet.has(evt.id)) {
          syncStore.remove(`${ITEM_PREFIX}${i}`);
          head = i + 1; // Advance head pointer past synced event
        } else {
          break; // Stop at first unsynced event (SyncManager flushes in order)
        }
      } catch {
        head = i + 1; // Advance past corrupt data
      }
    } else {
      head = i + 1; // Advance past missing data
    }
  }
  
  syncStore.set(HEAD_KEY, head);
  console.log(`[AuthSync] Purged ${eventIds.length} synced events. New head: ${head}`);
}
