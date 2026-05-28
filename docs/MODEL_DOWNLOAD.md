# Model Download Instructions

## Models Needed

You need to download these models and place them in:
`android/app/src/main/assets/models/`

### 1. BlazeFace Short Range (Face Detection) ✅ ALREADY DOWNLOADED
- **Size**: ~224KB
- **File**: `blaze_face_short_range.tflite`

### 2. Face Landmarker (468-point landmarks) ✅ ALREADY DOWNLOADED
- **Size**: ~3.6MB
- **File**: `face_landmarker.task`

### 3. MobileFaceNet (Face Embeddings) ⚠️ MANUAL DOWNLOAD NEEDED
- **Size**: ~5MB (will be quantized to ~1.5MB)
- **File**: `mobile_face_net.tflite`

**Download from one of these sources:**

1. **shubham0204/OnDevice-Face-Recognition-Android** (Recommended)
   - URL: https://github.com/shubham0204/OnDevice-Face-Recognition-Android
   - Navigate to `app/src/main/assets/` → download `facenet.tflite`
   - Rename to `mobile_face_net.tflite`

2. **sirius-ai/MobileFaceNet_TF** (Alternative)
   - URL: https://github.com/sirius-ai/MobileFaceNet_TF
   - Navigate to `tflite/` → download `MobileFaceNet.tflite`
   - Rename to `mobile_face_net.tflite`

3. **Google Drive backup** (if GitHub is blocked)
   - Search "MobileFaceNet tflite" on Kaggle or Hugging Face

### 4. Silent-Face Anti-Spoofing ⏳ DAY 4
- Will be converted from PyTorch → ONNX → TFLite on Day 4
- **File**: `silent_face.tflite`

## After downloading, verify:
```bash
ls -lh android/app/src/main/assets/models/
```

Expected output:
```
blaze_face_short_range.tflite   ~224KB
face_landmarker.task            ~3.6MB
mobile_face_net.tflite          ~1.5-5MB
```
