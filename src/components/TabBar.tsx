import React from 'react';
import { StyleSheet, Text, View, Platform, Pressable } from 'react-native';
import Animated, { useAnimatedStyle, useSharedValue, withSpring, withTiming } from 'react-native-reanimated';

import { LucideIcon } from 'lucide-react-native';
import { useTranslation, TranslationKey } from '../i18n';

export interface Tab {
  key: string;
  labelKey: TranslationKey;
  Icon: LucideIcon;
}

interface Props {
  tabs: Tab[];
  activeIndex: number;
  onPress: (index: number) => void;
}

export const TabBar: React.FC<Props> = ({ tabs, activeIndex, onPress }) => {
  return (
    <View style={styles.container}>
      {tabs.map((tab, index) => {
        const active = index === activeIndex;
        
        return (
          <TabBarItem 
            key={tab.key}
            tab={tab}
            active={active}
            onPress={() => onPress(index)}
          />
        );
      })}
    </View>
  );
};

const TabBarItem: React.FC<{ tab: Tab; active: boolean; onPress: () => void }> = ({ tab, active, onPress }) => {
  const isPressed = useSharedValue(false);
  const { t } = useTranslation();

  const pressStyle = useAnimatedStyle(() => ({
    transform: [{ scale: withSpring(isPressed.value ? 0.92 : 1.0, { damping: 15 }) }]
  }));

  const dotStyle = useAnimatedStyle(() => ({
    opacity: withTiming(active ? 1 : 0, { duration: 180 })
  }));

  const IconComponent = tab.Icon;

  return (
    <Pressable
      style={styles.tab}
      onPressIn={() => isPressed.value = true}
      onPressOut={() => isPressed.value = false}
      onPress={onPress}
    >
      <Animated.View style={[styles.tabContent, pressStyle]}>
        <IconComponent size={22} color={active ? '#f0a500' : '#4a5568'} strokeWidth={2.5} style={{ marginBottom: 3 }} />
        <Text style={[styles.label, active && styles.labelActive]}>{t(tab.labelKey)}</Text>
      </Animated.View>
      <Animated.View style={[styles.activeDot, dotStyle]} />
    </Pressable>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: '#0d1117',
    borderTopWidth: 1,
    borderTopColor: '#1e2433',
    height: Platform.OS === 'ios' ? 83 : 60,
    paddingBottom: Platform.OS === 'ios' ? 20 : 4,
    paddingTop: 8,
    alignItems: 'flex-start',
    elevation: 16,
    shadowColor: '#000000',
    shadowOffset: { width: 0, height: -3 },
    shadowOpacity: 0.35,
    shadowRadius: 12,
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 4,
    position: 'relative',
  },
  tabContent: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  icon: { fontSize: 22, marginBottom: 3, opacity: 0.35 },
  iconActive: { opacity: 1.0, color: '#f0a500' },
  label: {
    fontFamily: 'DMSans-Bold',
    fontSize: 10,
    color: '#4a5568',
    letterSpacing: 1.2,
    textTransform: 'uppercase',
    opacity: 0.35,
    marginTop: 3,
  },
  labelActive: { color: '#f0a500', opacity: 1.0 },
  activeDot: {
    position: 'absolute',
    top: 0,
    width: 28,
    height: 2,
    borderRadius: 1,
    backgroundColor: '#f0a500',
  },
});
