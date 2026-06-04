// Custom bottom tab navigation component.
// Implements a lightweight, state-driven tab bar without relying on heavy third-party navigation libraries.
import React from 'react';
import { StyleSheet, Text, TouchableOpacity, View, Platform } from 'react-native';

export interface Tab {
  key: string;
  label: string;
  icon: string;
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
          <TouchableOpacity
            key={tab.key}
            style={styles.tab}
            onPress={() => onPress(index)}
            activeOpacity={0.7}
          >
            <Text style={[styles.icon, active && styles.iconActive]}>{tab.icon}</Text>
            <Text style={[styles.label, active && styles.labelActive]}>{tab.label}</Text>
            {active && <View style={styles.activeDot} />}
          </TouchableOpacity>
        );
      })}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: '#0d1017',
    borderTopWidth: 1,
    borderTopColor: '#1e2433',
    paddingBottom: Platform.OS === 'ios' ? 20 : 4,
    paddingTop: 8,
    elevation: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 4,
    position: 'relative',
  },
  icon: { fontSize: 22, marginBottom: 3, opacity: 0.4 },
  iconActive: { opacity: 1 },
  label: {
    fontSize: 10,
    color: '#4a5568',
    fontWeight: '600',
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  labelActive: { color: '#f0a500' },
  activeDot: {
    position: 'absolute',
    top: 0,
    width: 32,
    height: 2,
    borderRadius: 1,
    backgroundColor: '#f0a500',
  },
});
