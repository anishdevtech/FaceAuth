// Renders an animated bounding box and facial landmark indicators over the camera preview.
// During liveness checks, it visualizes the BlazeFace keypoints (eyes, nose, mouth)
// connected by indicator lines to provide feedback on facial tracking.
// Note: BlazeFace keypoint indices are: 0=right eye, 1=left eye, 2=nose, 3=mouth, 4=right ear, 5=left ear.
import React, { useEffect, useRef, useMemo } from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import type { FaceBox } from '../ml/blazeface';

interface Props {
  box: FaceBox | null;
  label?: string;
  status?: 'scanning' | 'matched' | 'unknown';
  frameWidth: number;
  frameHeight: number;
  imageWidth?: number;
  imageHeight?: number;
  imageOrientation?: string;
  /** BlazeFace landmark points {x,y} in normalised [0,1] coords. */
  debugMesh?: {x: number; y: number}[];
  /** When true the landmark lines are rendered in blue (liveness mode). */
  livenessMode?: boolean;
}

const STATUS_COLORS = {
  scanning: '#FFFFFF',
  matched:  '#34C759',
  unknown:  '#FF3B30',
};

// Helper component that draws a straight line between two absolute pixel coordinates.
// It utilizes a transformed View to render the connecting segment.

interface LineProps {
  x1: number; y1: number;
  x2: number; y2: number;
  color: string;
  thickness?: number;
}

function LandmarkLine({ x1, y1, x2, y2, color, thickness = 2 }: LineProps) {
  const dx     = x2 - x1;
  const dy     = y2 - y1;
  const length = Math.sqrt(dx * dx + dy * dy);
  const angle  = Math.atan2(dy, dx) * (180 / Math.PI);
  const cx     = (x1 + x2) / 2;
  const cy     = (y1 + y2) / 2;

  return (
    <View
      style={{
        position:        'absolute',
        width:           length,
        height:          thickness,
        backgroundColor: color,
        borderRadius:    thickness / 2,
        opacity:         0.85,
        left:            cx - length / 2,
        top:             cy - thickness / 2,
        transform:       [{ rotate: `${angle}deg` }],
      }}
    />
  );
}

// Defines the connectivity graph for facial landmarks.
// Ear points (indices 4 and 5) are excluded to reduce visual noise.
const CONNECTIONS: [number, number][] = [
  [0, 1], // right eye → left eye
  [0, 2], // right eye → nose
  [1, 2], // left eye  → nose
  [2, 3], // nose      → mouth
  [0, 3], // right eye → mouth  (jaw-line feel)
  [1, 3], // left eye  → mouth
];

// Renders the main overlay including the bounding box and landmarks.
export const FaceOverlay: React.FC<Props> = ({
  box,
  label,
  status = 'scanning',
  frameWidth,
  frameHeight,
  imageWidth = 0,
  imageHeight = 0,
  imageOrientation,
  debugMesh,
  livenessMode = false,
}) => {
  const opacity   = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const left      = useRef(new Animated.Value(0)).current;
  const top       = useRef(new Animated.Value(0)).current;
  const width     = useRef(new Animated.Value(0)).current;
  const height    = useRef(new Animated.Value(0)).current;

  // Pulsating opacity when scanning
  useEffect(() => {
    if (status === 'scanning') {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 0.4, duration: 900, useNativeDriver: false }),
          Animated.timing(pulseAnim, { toValue: 1,   duration: 900, useNativeDriver: false }),
        ]),
      ).start();
    } else {
      pulseAnim.setValue(1);
    }
  }, [status, pulseAnim]);

  // Map normalised image coords → screen pixel coords
  const layoutInfo = useMemo(() => {
    let displayW = frameWidth;
    let displayH = frameHeight;
    let offsetX  = 0;
    let offsetY  = 0;

    if (imageWidth > 0 && imageHeight > 0) {
      const isPortrait = imageOrientation === 'portrait' || imageOrientation === 'portrait-upside-down';
      const logicalW   = isPortrait ? Math.min(imageWidth, imageHeight) : Math.max(imageWidth, imageHeight);
      const logicalH   = isPortrait ? Math.max(imageWidth, imageHeight) : Math.min(imageWidth, imageHeight);

      const frameAspect = frameWidth / frameHeight;
      const imageAspect = logicalW / logicalH;

      if (frameAspect > imageAspect) {
        displayW = frameHeight * imageAspect;
        displayH = frameHeight;
      } else {
        displayW = frameWidth;
        displayH = frameWidth / imageAspect;
      }

      offsetX = (frameWidth  - displayW) / 2;
      offsetY = (frameHeight - displayH) / 2;
    }
    return { displayW, displayH, offsetX, offsetY };
  }, [frameWidth, frameHeight, imageWidth, imageHeight, imageOrientation]);

  // Animate the bounding box
  useEffect(() => {
    if (box) {
      const { displayW, displayH, offsetX, offsetY } = layoutInfo;

      const padX  = (box.xmax - box.xmin) * 0.25;
      const padY  = (box.ymax - box.ymin) * 0.25;
      const pxmin = Math.max(box.xmin - padX, 0);
      const pymin = Math.max(box.ymin - padY, 0);
      const pxmax = Math.min(box.xmax + padX, 1);
      const pymax = Math.min(box.ymax + padY, 1);

      const x = pxmin * displayW + offsetX;
      const y = pymin * displayH + offsetY;
      const w = (pxmax - pxmin) * displayW;
      const h = (pymax - pymin) * displayH;

      Animated.parallel([
        Animated.timing(opacity, { toValue: 1, duration: 150, useNativeDriver: false }),
        Animated.spring(left,   { toValue: x, useNativeDriver: false, damping: 20 }),
        Animated.spring(top,    { toValue: y, useNativeDriver: false, damping: 20 }),
        Animated.spring(width,  { toValue: w, useNativeDriver: false, damping: 20 }),
        Animated.spring(height, { toValue: h, useNativeDriver: false, damping: 20 }),
      ]).start();
    } else {
      Animated.timing(opacity, { toValue: 0, duration: 200, useNativeDriver: false }).start();
    }
  }, [box, layoutInfo, opacity, left, top, width, height]);

  const borderColor   = STATUS_COLORS[status];
  const lineColor     = livenessMode ? '#007AFF' : 'rgba(255,255,255,0.6)';

  // Convert normalised landmark coords to absolute screen pixels
  const screenPts = useMemo(() => {
    if (!debugMesh || debugMesh.length === 0) return [];
    return debugMesh.map(pt => ({
      x: pt.x * layoutInfo.displayW + layoutInfo.offsetX,
      y: pt.y * layoutInfo.displayH + layoutInfo.offsetY,
    }));
  }, [debugMesh, layoutInfo]);

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {/* Bounding box corner accents */}
      <Animated.View
        style={[
          styles.box,
          { opacity: Animated.multiply(opacity, pulseAnim), left, top, width, height },
        ]}
      >
        <View style={[styles.corner, styles.tl, { borderColor }]} />
        <View style={[styles.corner, styles.tr, { borderColor }]} />
        <View style={[styles.corner, styles.bl, { borderColor }]} />
        <View style={[styles.corner, styles.br, { borderColor }]} />

        {/* Label badge */}
        {label ? (
          <View style={[styles.labelContainer, { backgroundColor: borderColor }]}>
            <Text style={styles.labelText} numberOfLines={1}>{label}</Text>
          </View>
        ) : null}
      </Animated.View>

      {/* Facial landmark lines (liveness mode) */}
      {screenPts.length >= 4 && CONNECTIONS.map(([a, b], i) => {
        const ptA = screenPts[a];
        const ptB = screenPts[b];
        if (!ptA || !ptB) return null;
        return (
          <LandmarkLine
            key={i}
            x1={ptA.x} y1={ptA.y}
            x2={ptB.x} y2={ptB.y}
            color={lineColor}
            thickness={livenessMode ? 2.5 : 1.5}
          />
        );
      })}

      {/* Landmark keypoint dots */}
      {screenPts.length >= 4 && screenPts.slice(0, 4).map((pt, i) => (
        <View
          key={`pt-${i}`}
          style={[
            styles.keypoint,
            livenessMode ? styles.keypointLiveness : styles.keypointNormal,
            { left: pt.x - (livenessMode ? 5 : 3), top: pt.y - (livenessMode ? 5 : 3) },
          ]}
        />
      ))}
    </View>
  );
};

// All the visual styles live here.

const CORNER = 36;
const BORDER = 4;
const RADIUS = 20;

const styles = StyleSheet.create({
  box: {
    position: 'absolute',
  },
  corner: {
    position: 'absolute',
    width:    CORNER,
    height:   CORNER,
    borderWidth: BORDER,
  },
  tl: { top: -BORDER, left: -BORDER,     borderRightWidth: 0,  borderBottomWidth: 0, borderTopLeftRadius: RADIUS },
  tr: { top: -BORDER, right: -BORDER,    borderLeftWidth: 0,   borderBottomWidth: 0, borderTopRightRadius: RADIUS },
  bl: { bottom: -BORDER, left: -BORDER,  borderRightWidth: 0,  borderTopWidth: 0,    borderBottomLeftRadius: RADIUS },
  br: { bottom: -BORDER, right: -BORDER, borderLeftWidth: 0,   borderTopWidth: 0,    borderBottomRightRadius: RADIUS },
  labelContainer: {
    position:            'absolute',
    top:                 -28,
    left:                -1.5,
    paddingHorizontal:   10,
    paddingVertical:     4,
    borderTopLeftRadius: 4,
    borderTopRightRadius: 4,
    minWidth: 80,
  },
  labelText: {
    color:       '#000',
    fontSize:    12,
    fontWeight:  '700',
    letterSpacing: 0.5,
  },
  keypoint: {
    position:    'absolute',
    borderRadius: 99,
  },
  keypointNormal: {
    width:           6,
    height:          6,
    backgroundColor: 'rgba(255,255,255,0.7)',
  },
  keypointLiveness: {
    width:           10,
    height:          10,
    backgroundColor: '#007AFF',
    shadowColor:     '#007AFF',
    shadowOffset:    { width: 0, height: 0 },
    shadowOpacity:   0.9,
    shadowRadius:    6,
    elevation:       4,
  },
});
