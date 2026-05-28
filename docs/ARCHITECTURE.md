# FaceAuth — Architecture Document

## Overview

FaceAuth is a fully offline facial recognition and liveness detection system built in React Native. All ML inference runs on-device using TFLite models via JSI (JavaScript Interface). No cloud connectivity is required for core authentication.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CAMERA FRAME                             │
│                    (VisionCamera @ 30fps)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CLAHE PREPROCESSING                            │
│         Contrast Limited Adaptive Histogram Equalization         │
│     Purpose: Handle harsh sunlight / deep shadows outdoors       │
│     Implementation: Pure JS, tile-based, ~8ms                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  YUNET FACE DETECTION (TFLite)                   │
│         Model: yunet.tflite (~220KB)                             │
│         Output: Bounding box + 5 facial landmarks                │
│         Latency: ~30ms on mid-range device                       │
│         Delegate: NNAPI (Android HW acceleration)                │
└─────────────┬───────────┬───────────────────────────────────────┘
              │           │
    ┌─────────┘           └──────────┐
    │ No face detected                │ Face found
    │                                 │
    ▼                                 ▼
"Position face                  ┌──────────────┐
 in frame"                      │  BLUR CHECK  │
                                │  (Laplacian  │
                                │   Variance)  │
                                └──────┬───────┘
                                       │
                          ┌────────────┼────────────┐
                          │ Blurry                   │ Sharp
                          ▼                          ▼
                    "Move closer"          ┌──────────────────┐
                                           │ DUAL LIVENESS    │
                                           │ GATE             │
                                           └────────┬─────────┘
                                                    │
                                    ┌───────────────┼───────────────┐
                                    │                               │
                                    ▼                               ▼
                          ┌──────────────────┐            ┌──────────────────┐
                          │  ACTIVE LIVENESS │            │ PASSIVE LIVENESS │
                          │                  │            │                  │
                          │  MediaPipe 468   │            │  Silent-Face     │
                          │  Landmarks:      │            │  TFLite (~2MB)   │
                          │  • EAR (Blink)   │            │                  │
                          │  • MAR (Smile)   │            │  Frequency-domain│
                          │  • Head Turn     │            │  anti-spoofing   │
                          └────────┬─────────┘            └────────┬─────────┘
                                   │                               │
                                   └───────────┬───────────────────┘
                                               │
                                    BOTH must pass
                                               │
                          ┌────────────────────┼───────────────────┐
                          │ FAIL                                   │ PASS
                          ▼                                        ▼
                  "Spoof detected"                    ┌──────────────────────┐
                                                      │ MOBILEFACENET INT8   │
                                                      │ (TFLite ~1.5MB)      │
                                                      │                      │
                                                      │ Output: 128-dim      │
                                                      │ face embedding       │
                                                      │ Latency: ~60ms       │
                                                      └──────────┬───────────┘
                                                                 │
                                                                 ▼
                                                      ┌──────────────────────┐
                                                      │ COSINE SIMILARITY    │
                                                      │ Threshold: 0.6       │
                                                      └──────────┬───────────┘
                                                                 │
                                              ┌──────────────────┼──────────────────┐
                                              │ REGISTER                             │ AUTH
                                              ▼                                      ▼
                                    Average 3 frame               Compare vs stored
                                    embeddings → store            embeddings
                                    in MMKV (AES-256)             (brute-force search)
                                              │                                      │
                                              └──────────────────┬───────────────────┘
                                                                 │
                                                                 ▼
                                                      ┌──────────────────────┐
                                                      │ AUTH RESULT          │
                                                      │ (success/fail +     │
                                                      │  similarity score)   │
                                                      └──────────┬───────────┘
                                                                 │
                                                                 ▼
                                                      ┌──────────────────────┐
                                                      │ LOCAL WRITE-AHEAD    │
                                                      │ QUEUE (MMKV)         │
                                                      │ Event stored locally │
                                                      │ with UUID + timestamp│
                                                      └──────────┬───────────┘
                                                                 │
                                                     [NetInfo detects network]
                                                                 │
                                                                 ▼
                                                      ┌──────────────────────┐
                                                      │ AWS SYNC (Amplify)   │
                                                      │ → DynamoDB           │
                                                      │ → ACK received       │
                                                      │ → Purge local queue  │
                                                      └──────────────────────┘
```

## Model Summary

| Model | File | Size | Purpose | Input | Output | Delegate |
|-------|------|------|---------|-------|--------|----------|
| YuNet | `yunet.tflite` | ~220KB | Face detection + 5 landmarks | RGB 320×320 | Bounding boxes + landmarks | NNAPI |
| MobileFaceNet | `mobile_face_net.tflite` | ~1.5MB | 128-dim face embedding | RGB 112×112 (aligned) | 128-dim unit vector | NNAPI |
| Silent-Face | `silent_face.tflite` | ~2MB | Passive anti-spoofing | RGB 80×80 | [spoof_prob, real_prob] | NNAPI |
| **Total** | | **~3.7MB** | | | | **Under 20MB budget** |

## Key Design Decisions

### 1. Why TFLite + JSI (react-native-fast-tflite)?
- **JSI (JavaScript Interface)** provides direct C++ bridge to native code
- No serialization overhead — frame data stays in native memory
- ~40% faster than bridge-based alternatives
- Critical for real-time 30fps processing

### 2. Why CLAHE Preprocessing?
- Standard histogram equalization amplifies noise in bright regions
- CLAHE divides image into tiles, equalizes each with contrast limit
- Specifically addresses harsh outdoor Indian sunlight + deep shadows
- ~8ms overhead with significant accuracy improvement

### 3. Why Dual Liveness?
- **Active** (EAR/MAR/Head Turn): Cannot be spoofed with printed photo
- **Passive** (Silent-Face frequency domain): Cannot be fooled by high-res prints or screen replays
- Both must pass independently — strongest anti-spoofing combination

### 4. Why MMKV over AsyncStorage?
- Memory-mapped I/O — direct memory access, no serialization round-trip
- ~40x faster than AsyncStorage for reading 128-float vectors
- AES-256 encryption at rest for biometric data
- Write-ahead queue pattern for offline reliability

### 5. Why Never Store Face Images?
- Only 128-dim embeddings stored (~512 bytes vs ~500KB per photo)
- Dramatically reduced attack surface
- Legally compliant with biometric data regulations
- Embedding alone cannot reconstruct original face

## Directory Structure

```
FaceAuth/
├── src/
│   ├── ml/                  # ML pipeline modules
│   │   ├── FaceDetector.ts     # YuNet face detection
│   │   ├── FaceEmbedder.ts     # MobileFaceNet embeddings
│   │   ├── FaceAlignment.ts    # Landmark-based face alignment
│   │   ├── Preprocessing.ts    # CLAHE implementation
│   │   └── QualityCheck.ts     # Laplacian blur detection
│   ├── liveness/            # Anti-spoofing modules
│   │   ├── ActiveLiveness.ts   # EAR/MAR/Head turn checks
│   │   └── PassiveLiveness.ts  # Silent-Face TFLite
│   ├── storage/             # Local data persistence
│   │   └── BiometricStore.ts   # MMKV encrypted embedding storage
│   ├── sync/                # Cloud sync modules
│   │   ├── LocalQueue.ts       # Write-ahead event queue
│   │   └── SyncManager.ts      # NetInfo + Amplify sync
│   ├── screens/             # UI screens
│   │   ├── HomeScreen.tsx
│   │   ├── RegisterScreen.tsx
│   │   ├── AuthScreen.tsx
│   │   └── ResultScreen.tsx
│   ├── components/          # Reusable UI components
│   └── utils/               # Shared utilities
├── docs/
│   ├── ARCHITECTURE.md      # This document
│   └── INTEGRATION.md       # Integration guide
├── android/app/src/main/assets/models/
│   ├── yunet.tflite
│   ├── mobile_face_net.tflite
│   └── silent_face.tflite
└── README.md
```

## Performance Targets (Redmi Note 12 / Realme C55)

| Stage | Target Latency |
|-------|---------------|
| CLAHE Preprocessing | ~8ms |
| YuNet Detection | ~35ms |
| Face Alignment | ~5ms |
| MobileFaceNet Embedding | ~65ms |
| Cosine Similarity Matching | ~2ms |
| **Total Pipeline** | **~115-135ms** |

## Accuracy Targets

| Metric | Target | Definition |
|--------|--------|------------|
| True Accept Rate (TAR) | >94% | Correct person accepted / total correct-person attempts |
| False Accept Rate (FAR) | 0% | Wrong person accepted / total wrong-person attempts |
| False Reject Rate (FRR) | <5% | Correct person rejected / total correct-person attempts |
