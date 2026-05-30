# FaceAuth Project History & Challenges

## 1. Project Overview
FaceAuth is an offline, real-time face authentication mobile application built with React Native. It uses a custom ML pipeline entirely on-device to detect faces, extract embeddings, and compare them against enrolled profiles.

### Architecture
- **Camera Pipeline**: Uses `react-native-vision-camera` (v5) with Frame Processors.
- **ML Engine**: `react-native-fast-tflite` (v3 Nitro modules) running C++ natively via JSI.
- **State & Storage**: `react-native-mmkv` for ultra-fast synchronous on-device storage.
- **Models**:
  - `blaze_face_short_range.tflite` for face bounding box detection.
  - `mobile_face_net.tflite` for generating 128-dimensional face feature embeddings.

---

## 2. Challenges Faced & Solutions

### Challenge A: VisionCamera v5 and Worklets
**Issue**: The migration from React Native VisionCamera v4 to v5 introduced significant architectural changes regarding how Worklets (functions that run on a separate high-performance C++ UI thread) execute. We encountered crashes related to `react-native-worklets` v0.9 (e.g., `methodWrapper TypeError: undefined is not a function`).
**Solution**: 
- Replaced deprecated `runOnJS(callback)(arg)` syntax with the correct `scheduleOnRN(callback, arg)`.
- Avoided passing complex native objects (like TFLite plugin errors) directly into `console.log` on the JS thread, as this triggered an internal binding bug within the `react-native-worklets` console monkey-patch.

### Challenge B: The 46MB Metro HTTP Timeout
**Issue**: When using the standard `require('../../assets/models/mobile_face_net.tflite')` approach, Metro bundler attempts to stream the massive 46MB ML model over local HTTP during development. This caused silent timeouts, resulting in the app hanging indefinitely on the "Loading AI models..." screen.
**Failed Attempt**: We initially tried modifying `react-native-fast-tflite` internally (patching Kotlin files to intercept `file:///android_asset/`). This was rejected as modifying `node_modules` is bad practice.
**Final Solution**: 
- Utilized `react-native-fs` to copy the TFLite models directly from the bundled Android assets to the app's local document directory (`RNFS.DocumentDirectoryPath`) on startup.
- We then pass the local filesystem path (`file:///data/...`) to the ML loader. This completely bypasses Metro's slow HTTP streaming and loads the models natively and instantly in under 1 second.

### Challenge C: UI Overlap on Varied Screen Sizes
**Issue**: The `EnrollScreen` and `VerifyScreen` used hardcoded math (e.g., `SCREEN_H - 180`) for the camera height. On devices with different aspect ratios or navigation bars, this caused the Capture button and Name input sheets to overlap or get pushed off-screen.
**Solution**:
- Removed hardcoded height calculations.
- Implemented robust Flexbox layouts (`flex: 1` on the camera container) to dynamically fill the available space, ensuring a responsive design across all devices.

---

## 3. Current State
- **Face Detection**: Fully working. Draws dynamic bounding boxes over the camera feed.
- **Face Enrollment**: Fully working. Extracts a 128D array from a cropped face and securely stores it alongside a name in MMKV.
- **Face Verification**: Fully working. Continously compares the live camera feed against enrolled faces and computes cosine similarity to confirm identity.
- **UI/UX**: Responsive flexbox layout with custom dark mode theme, micro-animations, and visual feedback for face alignment.
