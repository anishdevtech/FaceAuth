// Implements a Write-Ahead Queue (WAQ) pattern for synchronization of authentication events.
// Ensures events are persisted locally offline and asynchronously transmitted to remote
// endpoints when network connectivity is restored. Successfully transmitted events are subsequently purged.

import NetInfo from '@react-native-community/netinfo';
import { getUnsyncedEvents, markEventsSynced, type AuthEvent } from '../storage/authSync';

let isSyncing = false;

/**
 * Simulates an asynchronous server upload payload transmission.
 * Must be substituted with appropriate API invocation logic for production deployment.
 */
async function simulateUpload(event: AuthEvent): Promise<boolean> {
  return new Promise((resolve) => {
    setTimeout(() => {
      console.log(`[SyncManager] ✓ Uploaded event ${event.id} (${event.type}: ${event.success ? 'success' : 'fail'})`);
      resolve(true);
    }, 200); // simulate 200ms network round-trip
  });
}

/**
 * Flushes all pending synchronization events from the local persistence queue.
 * Transmits each event sequentially and marks acknowledged events for purging.
 */
export async function flushQueue(): Promise<void> {
  if (isSyncing) return; // prevent concurrent flushes
  isSyncing = true;

  try {
    const events = getUnsyncedEvents();
    if (events.length === 0) {
      console.log('[SyncManager] Queue empty — nothing to flush.');
      return;
    }

    console.log(`[SyncManager] Flushing ${events.length} queued event(s)...`);

    const synced: string[] = [];
    for (const event of events) {
      const ok = await simulateUpload(event);
      if (ok) {
        synced.push(event.id);
      } else {
        // Stop on first failure — retry on next connectivity event
        console.warn(`[SyncManager] Upload failed for ${event.id}, will retry later.`);
        break;
      }
    }

    if (synced.length > 0) {
      markEventsSynced(synced);
      console.log(`[SyncManager] Purged ${synced.length} synced event(s) from local queue.`);
    }
  } catch (err) {
    console.warn('[SyncManager] Flush error:', err);
  } finally {
    isSyncing = false;
  }
}

/**
 * Initializes network connectivity state listeners.
 * Triggers queue synchronization routines upon detection of an active internet connection.
 *
 * @returns Function to terminate the active state listener.
 */
export function startSyncManager(): () => void {
  console.log('[SyncManager] Started — listening for connectivity changes.');

  // Attempt an initial flush in case there are queued events from a previous session
  flushQueue();

  return NetInfo.addEventListener((state) => {
    if (state.isConnected && state.isInternetReachable) {
      console.log('[SyncManager] Network restored — triggering flush.');
      flushQueue();
    }
  });
}
