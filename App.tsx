// Main application entry point.
// Orchestrates the primary tab-based navigation layout (Verify, Enroll, Manage)
// and initializes global background services such as the SyncManager.
import React, { useEffect, useState } from 'react';
import { StyleSheet } from 'react-native';
import { SafeAreaView, SafeAreaProvider } from 'react-native-safe-area-context';

import { VerifyScreen } from './src/screens/VerifyScreen';
import { EnrollScreen } from './src/screens/EnrollScreen';
import { ManageScreen } from './src/screens/ManageScreen';
import { LogsScreen }   from './src/screens/LogsScreen';
import { TabBar, Tab } from './src/components/TabBar';
import { startSyncManager } from './src/sync/SyncManager';
import { ScanFace, UserPlus, Users, FileText } from 'lucide-react-native';

const TABS: Tab[] = [
  { key: 'verify',  labelKey: 'tab_verify',  Icon: ScanFace },
  { key: 'enroll',  labelKey: 'tab_enroll',  Icon: UserPlus },
  { key: 'manage',  labelKey: 'tab_manage',  Icon: Users },
  { key: 'logs',    labelKey: 'tab_logs',    Icon: FileText },
];

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);

  // Initialize the sync manager immediately on app mount to monitor network connectivity changes.
  useEffect(() => {
    const unsub = startSyncManager();
    return unsub;
  }, []);

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.root}>
        {activeTab === 0 && <VerifyScreen />}
        {activeTab === 1 && <EnrollScreen />}
        {activeTab === 2 && <ManageScreen />}
        {activeTab === 3 && <LogsScreen />}
        <TabBar tabs={TABS} activeIndex={activeTab} onPress={setActiveTab} />
      </SafeAreaView>
    </SafeAreaProvider>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0a0c0f' },
});

export default App;
