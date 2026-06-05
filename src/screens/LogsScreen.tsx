import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, FlatList, RefreshControl, Platform, TouchableOpacity } from 'react-native';
import { getAllEvents, type AuthEvent } from '../storage/authSync';
import { useTranslation } from '../i18n';
import { Settings } from 'lucide-react-native';
import { LanguagePickerModal } from '../components/LanguagePickerModal';

export const LogsScreen: React.FC = () => {
  const [logs, setLogs] = useState<AuthEvent[]>([]);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const { t } = useTranslation();
  const [langModalVisible, setLangModalVisible] = useState(false);

  const loadLogs = useCallback(() => {
    // Sort descending by timestamp
    const all = getAllEvents().sort((a, b) => b.timestamp - a.timestamp);
    setLogs(all);
  }, []);

  // Load initially
  React.useEffect(() => {
    loadLogs();
  }, [loadLogs]);

  const onRefresh = useCallback(() => {
    setIsRefreshing(true);
    loadLogs();
    setTimeout(() => setIsRefreshing(false), 500);
  }, [loadLogs]);

  const renderItem = ({ item }: { item: AuthEvent }) => {
    const date = new Date(item.timestamp);
    const timeString = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const dateString = date.toLocaleDateString();

    const isVerify = item.type === 'verify';
    
    return (
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.cardTitle}>
            {item.name ? item.name : t('unknown_subject')}
          </Text>
          <View style={[styles.badge, { backgroundColor: item.success ? 'rgba(52, 199, 89, 0.1)' : 'rgba(255, 59, 48, 0.1)', borderColor: item.success ? 'rgba(52, 199, 89, 0.3)' : 'rgba(255, 59, 48, 0.3)' }]}>
            <Text style={[styles.badgeText, { color: item.success ? '#34C759' : '#FF3B30' }]}>
              {item.success ? t('logs_success') : t('logs_failed')}
            </Text>
          </View>
        </View>

        <View style={styles.cardDetails}>
          <Text style={styles.detailText}>
            <Text style={{ fontFamily: 'IBMPlexMono-SemiBold', color: '#8b949e' }}>{t('logs_event')} </Text> {isVerify ? t('logs_attendance') : t('logs_enrollment')}
          </Text>
          <Text style={styles.detailText}>
            <Text style={{ fontFamily: 'IBMPlexMono-SemiBold', color: '#8b949e' }}>{t('logs_time')} </Text> {dateString} @ {timeString}
          </Text>
          <Text style={styles.detailText}>
            <Text style={{ fontFamily: 'IBMPlexMono-SemiBold', color: '#8b949e' }}>{t('logs_sync')} </Text> {item.synced ? t('logs_aws_uploaded') : t('logs_pending')}
          </Text>
        </View>
      </View>
    );
  };

  return (
    <View style={styles.root}>
      <View style={styles.header}>
        <View style={{ flex: 1, paddingRight: 10 }}>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
            <Text style={styles.screenTitle}>{t('logs_title')}</Text>
          </View>
          <Text style={styles.subtitle}>{t('logs_total_records', { count: logs.length })}</Text>
        </View>
        <TouchableOpacity onPress={() => setLangModalVisible(true)} style={styles.settingsIcon}>
          <Settings size={20} color="#6b7280" />
        </TouchableOpacity>
      </View>

      <FlatList
        data={logs}
        keyExtractor={item => item.id}
        renderItem={renderItem}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl refreshing={isRefreshing} onRefresh={onRefresh} tintColor="#f0a500" />
        }
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyTitle}>{t('logs_no_logs')}</Text>
            <Text style={styles.emptySubText}>{t('logs_scan_face')}</Text>
          </View>
        }
      />
      <LanguagePickerModal visible={langModalVisible} onClose={() => setLangModalVisible(false)} />
    </View>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0a0c0f' },
  header: {
    paddingTop:        Platform.OS === 'ios' ? 56 : 28,
    paddingHorizontal: 20,
    paddingBottom:     14,
    backgroundColor:   '#0d1117',
    borderBottomWidth: 1,
    borderBottomColor: '#1e2433',
    flexDirection:     'row',
    justifyContent:    'space-between',
    alignItems:        'center',
  },
  screenTitle: {
    fontFamily: 'BarlowCondensed-Bold',
    fontSize: 28,
    letterSpacing: 1.0,
    color: '#e8eaf0',
    textTransform: 'uppercase',
  },
  subtitle: {
    fontFamily: 'IBMPlexMono-Regular',
    fontSize: 10,
    color: '#6b7280',
    marginTop: 2,
    letterSpacing: 1.0,
  },
  settingsIcon: {
    padding: 6,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#1e2433',
  },
  listContent: { paddingBottom: 100 },
  card: {
    backgroundColor: '#0d1117',
    padding: 16,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#1e2433',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  cardTitle: {
    fontFamily: 'DMSans-Bold',
    fontSize: 16,
    color: '#e8eaf0',
    textTransform: 'uppercase',
  },
  badge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    borderWidth: 1,
  },
  badgeText: {
    fontFamily: 'DMSans-Bold',
    fontSize: 10,
    letterSpacing: 1.0,
  },
  cardDetails: {
    gap: 4,
  },
  detailText: {
    fontFamily: 'IBMPlexMono-Regular',
    color: '#6b7280',
    fontSize: 11,
    letterSpacing: 0.5,
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 80,
  },
  emptyTitle: {
    fontFamily: 'BarlowCondensed-Bold',
    color: '#6b7280',
    fontSize: 22,
    marginBottom: 10,
    textTransform: 'uppercase',
  },
  emptySubText: {
    fontFamily: 'DMSans-Regular',
    color: '#4a5568',
    fontSize: 14,
    textAlign: 'center',
    paddingHorizontal: 40,
  },
});
