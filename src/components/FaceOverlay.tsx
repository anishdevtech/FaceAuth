/**
 * FaceOverlay — Animated bounding box drawn over the camera
 *
 * Props:
 *  - box: Normalized FaceBox { x, y, width, height } or null
 *  - label: Text shown above the box ("UNKNOWN" / "John 87%")
 *  - status: controls corner color ('scanning' | 'matched' | 'unknown')
 *  - frameWidth / frameHeight: pixel size of the camera preview area
 */
import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import type { FaceBox } from '../ml/blazeface';

interface Props {
  box: FaceBox | null;
  label?: string;
  status?: 'scanning' | 'matched' | 'unknown';
  frameWidth: number;   // Layout width of the container
  frameHeight: number;  // Layout height of the container
  imageWidth?: number;  // Actual pixel width of the camera frame
  imageHeight?: number; // Actual pixel height of the camera frame
  imageOrientation?: string;
}

const STATUS_COLORS = {
  scanning: '#f0a500',
  matched: '#4ade80',
  unknown: '#ff4d4f',
};

export const FaceOverlay: React.FC<Props> = ({
  box,
  label,
  status = 'scanning',
  frameWidth,
  frameHeight,
  imageWidth = 0,
  imageHeight = 0,
  imageOrientation,
}) => {
  const opacity = useRef(new Animated.Value(0)).current;
  const left   = useRef(new Animated.Value(0)).current;
  const top    = useRef(new Animated.Value(0)).current;
  const width  = useRef(new Animated.Value(0)).current;
  const height = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (box && frameWidth > 0 && frameHeight > 0) {
      // Part 1: Detect and correct the swapped frame dimensions
      const isFrameRotated = imageOrientation === 'landscape-left' || imageOrientation === 'landscape-right';
      const frameW = isFrameRotated ? imageHeight : imageWidth;
      const frameH = isFrameRotated ? imageWidth : imageHeight;

      // Part 2: Calculate the actual rendered size of the camera image on screen
      let displayW = frameWidth;
      let displayH = frameHeight;
      let offsetX = 0;
      let offsetY = 0;

      if (frameW > 0 && frameH > 0) {
        const frameAspect = frameW / frameH;
        const screenAspect = frameWidth / frameHeight;

        displayW = frameAspect > screenAspect
          ? frameWidth
          : frameHeight * frameAspect;

        displayH = frameAspect > screenAspect
          ? frameWidth / frameAspect
          : frameHeight;

        // Black bar offsets
        offsetX = (frameWidth - displayW) / 2;
        offsetY = (frameHeight - displayH) / 2;
      }

      // Part 3: Map BlazeFace normalized [0,1] coordinates to screen pixels
      const x = box.xmin * displayW + offsetX;
      const y = box.ymin * displayH + offsetY;
      const w = (box.xmax - box.xmin) * displayW;
      const h = (box.ymax - box.ymin) * displayH;

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
  }, [box, frameWidth, frameHeight, imageWidth, imageHeight, opacity, left, top, width, height]);

  const borderColor = STATUS_COLORS[status];

  return (
    <Animated.View
      pointerEvents="none"
      style={[styles.box, { opacity, left, top, width, height, borderColor }]}
    >
      {/* Corner accents */}
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
  );
};

const CORNER = 16;
const BORDER = 2.5;

const styles = StyleSheet.create({
  box: {
    position: 'absolute',
    borderWidth: 1.5,
    borderRadius: 4,
  },
  corner: {
    position: 'absolute',
    width: CORNER,
    height: CORNER,
    borderWidth: BORDER,
  },
  tl: { top: -BORDER, left: -BORDER,   borderRightWidth: 0,  borderBottomWidth: 0, borderTopLeftRadius: 4 },
  tr: { top: -BORDER, right: -BORDER,  borderLeftWidth: 0,   borderBottomWidth: 0, borderTopRightRadius: 4 },
  bl: { bottom: -BORDER, left: -BORDER,  borderRightWidth: 0, borderTopWidth: 0,   borderBottomLeftRadius: 4 },
  br: { bottom: -BORDER, right: -BORDER, borderLeftWidth: 0,  borderTopWidth: 0,   borderBottomRightRadius: 4 },
  labelContainer: {
    position: 'absolute',
    top: -28,
    left: -1.5,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderTopLeftRadius: 4,
    borderTopRightRadius: 4,
    minWidth: 80,
  },
  labelText: {
    color: '#000',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
});
