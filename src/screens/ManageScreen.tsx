// Provides an interface for viewing and managing enrolled facial profiles.
// Reads directly from the local MMKV storage and allows targeted deletion or bulk clearance of identities.
import React, { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  FlatList,
  Platform,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { getAllFaces, deleteFace, clearAllFaces, EnrolledFace } from '../storage/faceStore';

const SeparatorComponent = () => <View style={styles.separator} />;

export const ManageScreen: React.FC = () => {
  const [faces, setFaces] = useState<EnrolledFace[]>([]);

  const reload = useCallback(() => setFaces(getAllFaces()), []);

  useEffect(() => { reload(); }, [reload]);

  const handleDelete = useCallback((face: EnrolledFace) => {
    Alert.alert('Delete face', `Remove "${face.name}" from the system?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete', style: 'destructive',
        onPress: () => { deleteFace(face.id); reload(); },
      },
    ]);
  }, [reload]);

  const handleClearAll = useCallback(() => {
    if (faces.length === 0) return;
    Alert.alert('Clear all', 'Delete ALL enrolled faces?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Clear All', style: 'destructive',
        onPress: () => { clearAllFaces(); reload(); },
      },
    ]);
  }, [faces, reload]);

  const renderItem = ({ item }: { item: EnrolledFace }) => {
    const initial = item.name.charAt(0).toUpperCase();
    const date = new Date(item.enrolledAt).toLocaleDateString('en-IN', {
      day: '2-digit', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
    return (
      <View style={styles.faceCard}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>{initial}</Text>
        </View>
        <View style={styles.faceInfo}>
          <Text style={styles.faceName}>{item.name}</Text>
          <Text style={styles.faceDate}>Enrolled: {date}</Text>
          <Text style={styles.faceVec}>128-D embedding ✓</Text>
        </View>
        <TouchableOpacity style={styles.deleteBtn} onPress={() => handleDelete(item)}>
          <Text style={styles.deleteBtnText}>🗑</Text>
        </TouchableOpacity>
      </View>
    );
  };

  return (
    <View style={styles.root}>
      <StatusBar barStyle="light-content" backgroundColor="#0a0c0f" />

      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>👥 Enrolled Faces</Text>
          <Text style={styles.headerSub}>
            {faces.length} identit{faces.length !== 1 ? 'ies' : 'y'} registered
          </Text>
        </View>
        {faces.length > 0 && (
          <TouchableOpacity style={styles.clearBtn} onPress={handleClearAll}>
            <Text style={styles.clearBtnText}>Clear All</Text>
          </TouchableOpacity>
        )}
      </View>

      {faces.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyIcon}>👤</Text>
          <Text style={styles.emptyTitle}>No faces enrolled</Text>
          <Text style={styles.emptySubtitle}>
            Go to the Enroll tab to add faces to the system.
          </Text>
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
    </View>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#000000' },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: Platform.OS === 'ios' ? 60 : 20,
    paddingHorizontal: 24,
    paddingBottom: 16,
    backgroundColor: '#000000',
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#2C2C2E',
  },
  headerTitle: { fontSize: 28, fontWeight: '800', color: '#FFFFFF', letterSpacing: 0.5 },
  headerSub:   { fontSize: 13, color: '#8E8E93', marginTop: 4, fontWeight: '600' },
  clearBtn: {
    backgroundColor: 'rgba(255, 59, 48, 0.15)',
    paddingHorizontal: 14, paddingVertical: 8, borderRadius: 12,
  },
  clearBtnText: { color: '#FF3B30', fontSize: 13, fontWeight: '700' },
  list:        { padding: 20 },
  separator:   { height: 12, backgroundColor: 'transparent' },
  faceCard:    { 
    flexDirection: 'row', alignItems: 'center', 
    backgroundColor: '#1C1C1E',
    padding: 16, borderRadius: 20,
  },
  avatar:      {
    width: 52, height: 52, borderRadius: 26,
    backgroundColor: '#007AFF',
    alignItems: 'center', justifyContent: 'center', marginRight: 16,
  },
  avatarText:  { color: '#FFFFFF', fontSize: 22, fontWeight: '800' },
  faceInfo:    { flex: 1 },
  faceName:    { color: '#FFFFFF', fontSize: 18, fontWeight: '700', marginBottom: 4 },
  faceDate:    { color: '#8E8E93', fontSize: 12, marginBottom: 2, fontWeight: '500' },
  faceVec:     { color: '#34C759', fontSize: 11, fontWeight: '700' },
  deleteBtn:   { padding: 10, borderRadius: 12, backgroundColor: 'rgba(255, 59, 48, 0.1)' },
  deleteBtnText: { fontSize: 20 },
  empty:       { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 40 },
  emptyIcon:   { fontSize: 64, marginBottom: 20 },
  emptyTitle:  { color: '#FFFFFF', fontSize: 22, fontWeight: '800', marginBottom: 10 },
  emptySubtitle: { color: '#8E8E93', fontSize: 15, textAlign: 'center', lineHeight: 22, fontWeight: '500' },
});
