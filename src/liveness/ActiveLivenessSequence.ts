// Implements the active liveness verification system.
// Generates a randomized sequence of head-movement tasks and evaluates
// facial landmarks against distance thresholds to verify physical presence.
import { clamp } from '../utils/mathUtils';

export type LivenessTask = 'turn_left' | 'turn_right';

export interface SequenceState {
  tasks: LivenessTask[];
  currentTaskIndex: number;
  framesInCurrentTask: number;
  passed: boolean;
  history: number[];
  /** Consecutive frames the current ratio has satisfied the threshold */
  holdFrames: number;
}

export function generateLivenessSequence(): SequenceState {
  'worklet';
  const allTasks: LivenessTask[] = ['turn_left', 'turn_right'];
  const tasks = [allTasks[Math.floor(Math.random() * allTasks.length)]];

  return {
    tasks,
    currentTaskIndex: 0,
    framesInCurrentTask: 0,
    passed: false,
    history: [],
    holdFrames: 0,
  };
}

export function getPromptForTask(task: LivenessTask): string {
  'worklet';
  switch (task) {
    case 'turn_left':  return 'Slowly turn your head LEFT';
    case 'turn_right': return 'Slowly turn your head RIGHT';
  }
}

export function getStepPrompt(state: SequenceState): string {
  'worklet';
  if (state.passed) return 'Liveness Verified!';
  if (state.currentTaskIndex >= state.tasks.length) return 'Verifying...';

  const task = state.tasks[state.currentTaskIndex];
  return getPromptForTask(task);
}

// BlazeFace landmark indices used for orientation tracking.
const REYE = 0; // Right eye
const LEYE = 1; // Left eye
const NOSE = 2; // Nose tip

// Computes the Euclidean distance between two indexed points in a flat coordinate array.
function distArr(arr: Float32Array, i: number, j: number): number {
  'worklet';
  const dx = arr[i * 2] - arr[j * 2];
  const dy = arr[i * 2 + 1] - arr[j * 2 + 1];
  return Math.sqrt(dx * dx + dy * dy);
}

// Evaluates head rotation by comparing the ratio of nose-to-eye distances.
// A left turn decreases the left-side distance relative to the right (ratio < 1.0).
// A right turn decreases the right-side distance relative to the left (ratio > 1.0).
// The target state must be maintained for HOLD_FRAMES_REQUIRED frames to mitigate jitter.
const RATIO_TURN_RIGHT = 1.35; // ratio above this = head turned right
const RATIO_TURN_LEFT  = 0.72; // ratio below this = head turned left
const HOLD_FRAMES_REQUIRED = 2; // consecutive frames required to confirm

export function checkSequenceTask(
  landmarks: Float32Array,
  state: SequenceState,
): boolean {
  'worklet';
  if (state.passed) return true;
  if (!landmarks || landmarks.length < 6) return false;

  const task = state.tasks[state.currentTaskIndex];
  state.framesInCurrentTask++;

  const distL = distArr(landmarks, NOSE, LEYE);
  const distR = distArr(landmarks, NOSE, REYE);

  // Avoid division-by-zero when one eye disappears off-frame
  if (distR < 0.001) return false;
  const ratio = distL / distR;

  // Store ratio in history for debugging (last 10 frames)
  state.history.push(ratio);
  if (state.history.length > 10) state.history.shift();

  let thresholdMet = false;
  if (task === 'turn_left'  && ratio < RATIO_TURN_LEFT)  thresholdMet = true;
  if (task === 'turn_right' && ratio > RATIO_TURN_RIGHT) thresholdMet = true;

  if (thresholdMet) {
    state.holdFrames = (state.holdFrames || 0) + 1;
  } else {
    state.holdFrames = 0; // reset hold counter on any frame that doesn't satisfy
  }

  return state.holdFrames >= HOLD_FRAMES_REQUIRED;
}
