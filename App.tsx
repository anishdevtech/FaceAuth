/**
 * FaceAuth — NHAI Hackathon 7.0 / Datalake 3.0
 *
 * Day 2: Full offline face recognition pipeline
 *   - BlazeFace detection (on-device, <100ms)
 *   - MobileFaceNet embedding (128-D, offline)
 *   - MMKV local storage
 *   - Three-tab layout: Verify | Enroll | Manage
 */
import React, { useState } from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';

import { VerifyScreen } from './src/screens/VerifyScreen';
import { EnrollScreen } from './src/screens/EnrollScreen';
import { ManageScreen } from './src/screens/ManageScreen';
import { TabBar, Tab } from './src/components/TabBar';

const TABS: Tab[] = [
  { key: 'verify',  label: 'Verify',  icon: '🔍' },
  { key: 'enroll',  label: 'Enroll',  icon: '➕' },
  { key: 'manage',  label: 'Manage',  icon: '👥' },
];

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <SafeAreaView style={styles.root}>
      {activeTab === 0 && <VerifyScreen />}
      {activeTab === 1 && <EnrollScreen />}
      {activeTab === 2 && <ManageScreen />}
      <TabBar tabs={TABS} activeIndex={activeTab} onPress={setActiveTab} />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0a0c0f' },
});

export default App;
