// Renders an animated bounding box and facial landmark indicators over the camera preview.
// Uses strict 60fps react-native-reanimated physics.
import React, { useEffect, useMemo } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Animated, { 
  useSharedValue, 
  useAnimatedStyle, 
  withSpring, 
  withTiming, 
  withRepeat, 
  withSequence 
} from 'react-native-reanimated';
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

}

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
  // Shared values for bounding box position & size
  const left = useSharedValue(0);
  const top = useSharedValue(0);
  const width = useSharedValue(0);
  const height = useSharedValue(0);
  const boxOpacity = useSharedValue(0);

  // Status crossfade opacities
  const whiteOpacity = useSharedValue(1);
  const greenOpacity = useSharedValue(0);
  const redOpacity = useSharedValue(0);
  const scanPulse = useSharedValue(1.0);

  useEffect(() => {
    'worklet';
    whiteOpacity.value = withTiming(status === 'scanning' ? 1 : 0, { duration: 250 });
    greenOpacity.value = withTiming(status === 'matched' ? 1 : 0, { duration: 250 });
    redOpacity.value = withTiming(status === 'unknown' ? 1 : 0, { duration: 250 });

    if (status === 'scanning') {
      scanPulse.value = withRepeat(
        withSequence(
          withTiming(0.4, { duration: 450 }),
          withTiming(1.0, { duration: 450 })
        ),
        -1, false
      );
    } else {
      scanPulse.value = withTiming(1.0, { duration: 200 });
    }
  }, [status, whiteOpacity, greenOpacity, redOpacity, scanPulse]);

  // Layout calculation
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

  // Spring physics when box changes
  useEffect(() => {
    if (box) {
      const { displayW, displayH, offsetX, offsetY } = layoutInfo;

      // Use strict ellipse ratio centered on the detected face
      const faceW = (box.xmax - box.xmin) * displayW;
      const faceH = (box.ymax - box.ymin) * displayH;
      const centerX = box.xmin * displayW + faceW / 2 + offsetX;
      const centerY = box.ymin * displayH + faceH / 2 + offsetY;

      const newW = faceW * 1.4;
      const newH = newW * 1.35; // Taller than wide -> Ellipse
      const newX = centerX - newW / 2;
      const newY = centerY - newH / 2;

      boxOpacity.value = withTiming(1, { duration: 50 });
      left.value = withSpring(newX, { damping: 20, stiffness: 350 });
      top.value = withSpring(newY, { damping: 20, stiffness: 350 });
      width.value = withSpring(newW, { damping: 20, stiffness: 350 });
      height.value = withSpring(newH, { damping: 20, stiffness: 350 });
    } else {
      boxOpacity.value = withTiming(0, { duration: 200 });
    }
  }, [box, layoutInfo, left, top, width, height, boxOpacity]);

  // Animated Styles
  const containerStyle = useAnimatedStyle(() => ({
    opacity: boxOpacity.value,
    transform: [
      { translateX: left.value },
      { translateY: top.value }
    ],
    width: width.value,
    height: height.value,
  }));

  const whiteOvalStyle = useAnimatedStyle(() => ({
    opacity: whiteOpacity.value * scanPulse.value,
    borderColor: '#FFFFFF',
  }));

  const greenOvalStyle = useAnimatedStyle(() => ({
    opacity: greenOpacity.value,
    borderColor: '#34C759',
  }));

  const redOvalStyle = useAnimatedStyle(() => ({
    opacity: redOpacity.value,
    borderColor: '#FF3B30',
  }));

  // Floating label badge style
  const labelOpacity = useAnimatedStyle(() => ({
    opacity: withTiming(label ? 1 : 0, { duration: 120 }),
  }));

  const labelBgStyleWhite = useAnimatedStyle(() => ({ opacity: whiteOpacity.value, backgroundColor: '#FFFFFF' }));
  const labelBgStyleGreen = useAnimatedStyle(() => ({ opacity: greenOpacity.value, backgroundColor: '#34C759' }));
  const labelBgStyleRed   = useAnimatedStyle(() => ({ opacity: redOpacity.value, backgroundColor: '#FF3B30' }));

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      <Animated.View style={[styles.boxContainer, containerStyle]}>
        
        {/* Three stacked absolute ovals for crossfading */}
        <Animated.View style={[styles.oval, whiteOvalStyle]} />
        <Animated.View style={[styles.oval, greenOvalStyle]} />
        <Animated.View style={[styles.oval, redOvalStyle]} />

        {/* Floating Label Badge */}
        <Animated.View style={[styles.labelWrapper, labelOpacity]}>
          <Animated.View style={[styles.labelBg, labelBgStyleWhite]} />
          <Animated.View style={[styles.labelBg, labelBgStyleGreen]} />
          <Animated.View style={[styles.labelBg, labelBgStyleRed]} />
          <Text style={styles.labelText} numberOfLines={1}>{label || ''}</Text>
        </Animated.View>

      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  boxContainer: {
    position: 'absolute',
    alignItems: 'center',
    top: 0,
    left: 0,
  },
  oval: {
    ...StyleSheet.absoluteFill,
    borderWidth: 2.5,
    borderStyle: 'dashed',
    borderRadius: 999,
  },
  labelWrapper: {
    position: 'absolute',
    top: -28,
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 6,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 80,
    overflow: 'hidden',
  },
  labelBg: {
    ...StyleSheet.absoluteFill,
  },
  labelText: {
    fontFamily: 'DMSans-ExtraBold',
    fontSize: 13,
    color: '#000000',
    letterSpacing: 0.2,
    zIndex: 1,
  },
});
