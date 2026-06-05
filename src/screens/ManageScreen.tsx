// Provides an interface for viewing and managing enrolled facial profiles.
// Reads directly from the local MMKV storage and allows targeted deletion or bulk clearance of identities.
import React, { useCallback, useEffect, useState } from 'react';
import {
  FlatList,
  Platform,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { getAllFaces, deleteFace, clearAllFaces, EnrolledFace } from '../storage/faceStore';
import { ConfirmModal } from '../components/ConfirmModal';
import { useTranslation } from '../i18n';
import { Settings, Trash2 } from 'lucide-react-native';
import { LanguagePickerModal } from '../components/LanguagePickerModal';

const SeparatorComponent = () => <View style={styles.separator} />;

export const ManageScreen: React.FC = () => {
  const [faces, setFaces] = useState<EnrolledFace[]>([]);
  const { t } = useTranslation();
  const [langModalVisible, setLangModalVisible] = useState(false);
  
  const [modal, setModal] = useState<{
    visible: boolean;
    title: string;
    subtitle: string;
    confirmText: string;
    confirmDestructive: boolean;
    icon: string;
    onConfirm: () => void;
  }>({
    visible: false,
    title: '',
    subtitle: '',
    confirmText: '',
    confirmDestructive: false,
    icon: '',
    onConfirm: () => {},
  });

  const hideModal = useCallback(() => setModal(m => ({ ...m, visible: false })), []);

  const reload = useCallback(() => setFaces(getAllFaces()), []);

  useEffect(() => { reload(); }, [reload]);

  const handleDelete = useCallback((face: EnrolledFace) => {
    setModal({
      visible: true,
      title: t('manage_delete_face'),
      subtitle: t('manage_delete_face_sub', { name: face.name }),
      confirmText: 'DELETE',
      confirmDestructive: true,
      icon: '⚠️',
      onConfirm: () => {
        deleteFace(face.id);
        reload();
        hideModal();
      },
    });
  }, [reload, hideModal, t]);

  const handleClearAll = useCallback(() => {
    if (faces.length === 0) return;
    setModal({
      visible: true,
      title: t('manage_clear_all_title'),
      subtitle: t('manage_clear_all_sub'),
      confirmText: t('manage_clear_all'),
      confirmDestructive: true,
      icon: '⚠️',
      onConfirm: () => {
        clearAllFaces();
        reload();
        hideModal();
      },
    });
  }, [faces.length, reload, hideModal, t]);

  const renderItem = ({ item }: { item: EnrolledFace }) => {
    const initial = item.name.charAt(0).toUpperCase();
    const date = new Date(item.enrolledAt).toLocaleDateString('en-IN', {
      day: '2-digit', month: 'short', year: 'numeric',
    });
    return (
      <View style={styles.faceCard}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>{initial}</Text>
        </View>
        <View style={styles.faceInfo}>
          <Text style={styles.faceName}>{item.name}</Text>
          <Text style={styles.faceDate}>{date}</Text>
        </View>
        <TouchableOpacity style={styles.deleteBtn} onPress={() => handleDelete(item)}>
          <Trash2 size={16} color="#FF3B30" />
        </TouchableOpacity>
      </View>
    );
  };

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="#0a0c0f" />

      <View style={styles.header}>
        <View style={{ flex: 1, paddingRight: 10 }}>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
            <Text style={styles.screenTitle}>{t('manage_title')}</Text>
          </View>
          <Text style={styles.subtitle}>
            {faces.length === 1 ? t('manage_identity', { count: faces.length }) : t('manage_identities', { count: faces.length })}
          </Text>
        </View>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 12 }}>
          {faces.length > 0 && (
            <TouchableOpacity style={styles.clearBtn} onPress={handleClearAll}>
              <Text style={styles.clearBtnText}>{t('manage_clear_all')}</Text>
            </TouchableOpacity>
          )}
          <TouchableOpacity onPress={() => setLangModalVisible(true)} style={styles.settingsIcon}>
            <Settings size={20} color="#6b7280" />
          </TouchableOpacity>
        </View>
      </View>

      {faces.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyTitle}>{t('manage_no_profiles')}</Text>
          <Text style={styles.emptySubtitle}>{t('manage_go_enroll')}</Text>
        </View>
      ) : (
        <FlatList
          data={faces}
          keyExtractor={item => item.id}
          renderItem={renderItem}
          contentContainerStyle={styles.list}
          ItemSeparatorComponent={SeparatorComponent}
        />
      )}

      <ConfirmModal
        visible={modal.visible}
        icon={modal.icon}
        title={modal.title}
        subtitle={modal.subtitle}
        confirmText={modal.confirmText}
        cancelText={t('cancel')}
        confirmDestructive={modal.confirmDestructive}
        onConfirm={modal.onConfirm}
        onCancel={hideModal}
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
    color: '#8b949e',
    marginTop: 2,
  },
  settingsIcon: {
    padding: 6,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#1e2433',
  },
  clearBtn: {
    backgroundColor: 'rgba(255, 59, 48, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(255, 59, 48, 0.3)',
    paddingHorizontal: 12, paddingVertical: 6, borderRadius: 6,
  },
  clearBtnText: { fontFamily: 'DMSans-Bold', color: '#FF3B30', fontSize: 11, letterSpacing: 1.0, textTransform: 'uppercase' },
  list:        { paddingVertical: 10 },
  separator:   { height: 0 },
  faceCard:    { 
    flexDirection: 'row', alignItems: 'center', 
    backgroundColor: '#0d1117',
    borderBottomWidth: 1, borderBottomColor: '#1e2433',
    paddingHorizontal: 20, paddingVertical: 16,
  },
  avatar:      {
    width: 48, height: 48, borderRadius: 24,
    backgroundColor: '#161b22',
    borderWidth: 1, borderColor: '#2d3748',
    alignItems: 'center', justifyContent: 'center', marginRight: 16,
  },
  avatarText:  { fontFamily: 'BarlowCondensed-Bold', color: '#e8eaf0', fontSize: 24 },
  faceInfo:    { flex: 1 },
  faceName:    { fontFamily: 'DMSans-SemiBold', color: '#e8eaf0', fontSize: 16, marginBottom: 4 },
  faceDate:    { fontFamily: 'IBMPlexMono-Regular', color: '#6b7280', fontSize: 10, textTransform: 'uppercase' },
  deleteBtn:   { padding: 10, borderRadius: 6, backgroundColor: 'rgba(255, 59, 48, 0.1)' },
  deleteBtnText: { fontSize: 16 },
  empty:       { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 40 },
  emptyTitle:  { fontFamily: 'BarlowCondensed-Bold', color: '#6b7280', fontSize: 22, marginBottom: 10, textTransform: 'uppercase' },
  emptySubtitle: { fontFamily: 'DMSans-Regular', color: '#4a5568', fontSize: 14, textAlign: 'center', lineHeight: 22 },
});
