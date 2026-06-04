// Provides a highly customizable, animated confirmation modal.
// Designed to replace the native Alert module for improved UI consistency.
// Accepts customizable icons, titles, subtitles, and destructive actions.
import React, { useEffect } from 'react';
import {
  Modal,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import Reanimated, {
  FadeIn,
  FadeOut,
  SlideInDown,
  SlideOutDown,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
} from 'react-native-reanimated';

interface ConfirmModalProps {
  visible: boolean;
  icon?: string;
  title: string;
  subtitle?: string;
  confirmText?: string;
  cancelText?: string;
  confirmDestructive?: boolean;
  onConfirm: () => void;
  onCancel?: () => void;
}

export const ConfirmModal: React.FC<ConfirmModalProps> = ({
  visible,
  icon = '✓',
  title,
  subtitle,
  confirmText = 'OK',
  cancelText,
  confirmDestructive = false,
  onConfirm,
  onCancel,
}) => {
  const iconScale = useSharedValue(0.5);

  useEffect(() => {
    if (visible) {
      iconScale.value = withSpring(1, { damping: 12, stiffness: 200 });
    } else {
      iconScale.value = 0.5;
    }
  }, [visible, iconScale]);

  const iconStyle = useAnimatedStyle(() => ({
    transform: [{ scale: iconScale.value }],
  }));

  return (
    <Modal
      transparent
      visible={visible}
      animationType="none"
      statusBarTranslucent
      onRequestClose={onCancel ?? onConfirm}
    >
      {/* Backdrop */}
      <Reanimated.View
        entering={FadeIn.duration(200)}
        exiting={FadeOut.duration(200)}
        style={styles.backdrop}
      >
        <TouchableOpacity
          style={StyleSheet.absoluteFill}
          activeOpacity={1}
          onPress={onCancel ?? onConfirm}
        />

        {/* Card */}
        <Reanimated.View
          entering={SlideInDown.springify().damping(18).stiffness(180)}
          exiting={SlideOutDown.duration(200)}
          style={styles.card}
        >
          {/* Icon */}
          <Reanimated.Text style={[styles.icon, iconStyle]}>
            {icon}
          </Reanimated.Text>

          {/* Title */}
          <Text style={styles.title}>{title}</Text>

          {/* Subtitle */}
          {subtitle ? (
            <Text style={styles.subtitle}>{subtitle}</Text>
          ) : null}

          {/* Divider */}
          <View style={styles.divider} />

          {/* Actions */}
          <View style={[styles.actions, cancelText ? styles.actionsRow : null]}>
            {cancelText ? (
              <TouchableOpacity
                style={[styles.btn, styles.cancelBtn]}
                onPress={onCancel}
                activeOpacity={0.7}
              >
                <Text style={styles.cancelBtnText}>{cancelText}</Text>
              </TouchableOpacity>
            ) : null}

            <TouchableOpacity
              style={[
                styles.btn,
                styles.confirmBtn,
                confirmDestructive && styles.destructiveBtn,
              ]}
              onPress={onConfirm}
              activeOpacity={0.7}
            >
              <Text
                style={[
                  styles.confirmBtnText,
                  confirmDestructive && styles.destructiveBtnText,
                ]}
              >
                {confirmText}
              </Text>
            </TouchableOpacity>
          </View>
        </Reanimated.View>
      </Reanimated.View>
    </Modal>
  );
};

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
  },
  card: {
    width: '100%',
    backgroundColor: '#1C1C1E',
    borderRadius: 28,
    padding: 28,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 20 },
    shadowOpacity: 0.5,
    shadowRadius: 30,
    elevation: 20,
  },
  icon: {
    fontSize: 56,
    marginBottom: 16,
  },
  title: {
    color: '#FFFFFF',
    fontSize: 22,
    fontWeight: '800',
    textAlign: 'center',
    letterSpacing: 0.2,
    marginBottom: 8,
  },
  subtitle: {
    color: '#8E8E93',
    fontSize: 15,
    fontWeight: '500',
    textAlign: 'center',
    lineHeight: 22,
    marginBottom: 4,
  },
  divider: {
    width: '100%',
    height: StyleSheet.hairlineWidth,
    backgroundColor: '#2C2C2E',
    marginVertical: 20,
  },
  actions: {
    width: '100%',
  },
  actionsRow: {
    flexDirection: 'row',
    gap: 12,
  },
  btn: {
    paddingVertical: 15,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cancelBtn: {
    flex: 1,
    backgroundColor: '#2C2C2E',
  },
  cancelBtnText: {
    color: '#8E8E93',
    fontSize: 16,
    fontWeight: '600',
  },
  confirmBtn: {
    flex: 1,
    backgroundColor: '#007AFF',
  },
  confirmBtnText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '700',
  },
  destructiveBtn: {
    backgroundColor: 'rgba(255,59,48,0.15)',
    borderWidth: 1,
    borderColor: '#FF3B30',
  },
  destructiveBtnText: {
    color: '#FF3B30',
  },
});
