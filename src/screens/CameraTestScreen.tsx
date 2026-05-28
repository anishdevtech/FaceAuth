import React, {useCallback, useEffect, useState} from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Alert,
  Platform,
  Linking,
  StatusBar,
} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
} from 'react-native-vision-camera';

const CameraTestScreen: React.FC = () => {
  const {hasPermission, requestPermission} = useCameraPermission();
  const [isActive, setIsActive] = useState(true);
  const device = useCameraDevice('front');

  useEffect(() => {
    if (!hasPermission) {
      requestPermission().then(granted => {
        if (!granted) {
          Alert.alert(
            'Camera Permission Required',
            'FaceAuth needs camera access for facial recognition. Please grant camera permission in Settings.',
            [
              {text: 'Cancel', style: 'cancel'},
              {text: 'Open Settings', onPress: () => Linking.openSettings()},
            ],
          );
        }
      });
    }
  }, [hasPermission, requestPermission]);

  const toggleCamera = useCallback(() => {
    setIsActive(prev => !prev);
  }, []);

  if (!hasPermission) {
    return (
      <View style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#0a0c0f" />
        <View style={styles.messageContainer}>
          <Text style={styles.icon}>📷</Text>
          <Text style={styles.title}>Camera Permission</Text>
          <Text style={styles.subtitle}>
            FaceAuth needs camera access to perform facial recognition.
          </Text>
          <TouchableOpacity
            style={styles.button}
            onPress={() => requestPermission()}>
            <Text style={styles.buttonText}>Grant Permission</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#0a0c0f" />
        <View style={styles.messageContainer}>
          <Text style={styles.icon}>⚠️</Text>
          <Text style={styles.title}>No Camera Found</Text>
          <Text style={styles.subtitle}>
            Could not find a front-facing camera on this device.
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0a0c0f" />

      {/* Camera Preview */}
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isActive}
      />

      {/* Overlay */}
      <View style={styles.overlay}>
        {/* Top bar */}
        <View style={styles.topBar}>
          <Text style={styles.topBarTitle}>📹 Camera Test</Text>
          <View style={styles.statusBadge}>
            <View
              style={[
                styles.statusDot,
                {backgroundColor: isActive ? '#4ade80' : '#ff4d4f'},
              ]}
            />
            <Text style={styles.statusText}>
              {isActive ? 'LIVE' : 'PAUSED'}
            </Text>
          </View>
        </View>

        {/* Face guide oval */}
        <View style={styles.ovalContainer}>
          <View style={styles.oval} />
          <Text style={styles.guideText}>Position your face in the oval</Text>
        </View>

        {/* Bottom info */}
        <View style={styles.bottomBar}>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Device:</Text>
            <Text style={styles.infoValue}>
              {device.name || 'Front Camera'}
            </Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Platform:</Text>
            <Text style={styles.infoValue}>
              {Platform.OS} {Platform.Version}
            </Text>
          </View>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Status:</Text>
            <Text
              style={[
                styles.infoValue,
                {color: isActive ? '#4ade80' : '#ff4d4f'},
              ]}>
              {isActive ? '✅ Camera Active' : '⏸ Camera Paused'}
            </Text>
          </View>

          <TouchableOpacity
            style={[
              styles.button,
              {backgroundColor: isActive ? '#ff4d4f' : '#4ade80'},
            ]}
            onPress={toggleCamera}>
            <Text style={styles.buttonText}>
              {isActive ? '⏸ Pause Camera' : '▶ Resume Camera'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0c0f',
  },
  messageContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  icon: {
    fontSize: 48,
    marginBottom: 16,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: '#e8eaf0',
    marginBottom: 8,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 15,
    color: '#9da3b4',
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 24,
  },
  overlay: {
    ...StyleSheet.absoluteFill,
    justifyContent: 'space-between',
  },
  topBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingHorizontal: 20,
    paddingBottom: 12,
    backgroundColor: 'rgba(10, 12, 15, 0.7)',
  },
  topBarTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#f0a500',
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(26, 30, 39, 0.8)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#252b38',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  statusText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#e8eaf0',
    letterSpacing: 1,
  },
  ovalContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  oval: {
    width: 220,
    height: 300,
    borderRadius: 110,
    borderWidth: 3,
    borderColor: 'rgba(240, 165, 0, 0.6)',
    borderStyle: 'dashed',
  },
  guideText: {
    marginTop: 16,
    fontSize: 14,
    color: 'rgba(255, 255, 255, 0.7)',
    textAlign: 'center',
  },
  bottomBar: {
    backgroundColor: 'rgba(10, 12, 15, 0.85)',
    paddingHorizontal: 20,
    paddingVertical: 20,
    paddingBottom: Platform.OS === 'ios' ? 36 : 20,
    borderTopWidth: 1,
    borderTopColor: '#252b38',
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 6,
  },
  infoLabel: {
    fontSize: 13,
    color: '#5c6478',
    fontWeight: '600',
  },
  infoValue: {
    fontSize: 13,
    color: '#e8eaf0',
  },
  button: {
    backgroundColor: '#f0a500',
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 16,
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '700',
    color: '#000',
  },
});

export default CameraTestScreen;
