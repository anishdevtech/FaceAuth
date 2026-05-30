/**
 * ManageScreen — View and delete enrolled faces
 */
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
          ItemSeparatorComponent={() => <View style={styles.separator} />}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: '#0a0c0f' },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: Platform.OS === 'ios' ? 52 : 16,
    paddingHorizontal: 20,
    paddingBottom: 12,
    backgroundColor: '#0d1117',
    borderBottomWidth: 1,
    borderBottomColor: '#1e2433',
  },
  headerTitle: { fontSize: 18, fontWeight: '700', color: '#e8eaf0' },
  headerSub:   { fontSize: 12, color: '#5c6478', marginTop: 2 },
  clearBtn: {
    borderWidth: 1, borderColor: '#ff4d4f',
    paddingHorizontal: 12, paddingVertical: 6, borderRadius: 6,
  },
  clearBtnText: { color: '#ff4d4f', fontSize: 12, fontWeight: '600' },
  list:        { padding: 16 },
  separator:   { height: 1, backgroundColor: '#1e2433' },
  faceCard:    { flexDirection: 'row', alignItems: 'center', paddingVertical: 14, paddingHorizontal: 4 },
  avatar:      {
    width: 48, height: 48, borderRadius: 24,
    backgroundColor: '#f0a500',
    alignItems: 'center', justifyContent: 'center', marginRight: 14,
  },
  avatarText:  { color: '#000', fontSize: 20, fontWeight: '800' },
  faceInfo:    { flex: 1 },
  faceName:    { color: '#e8eaf0', fontSize: 16, fontWeight: '700', marginBottom: 2 },
  faceDate:    { color: '#5c6478', fontSize: 11, marginBottom: 2 },
  faceVec:     { color: '#4ade80', fontSize: 10, fontWeight: '600' },
  deleteBtn:   { padding: 8, borderRadius: 8, backgroundColor: 'rgba(255,77,79,0.1)' },
  deleteBtnText: { fontSize: 20 },
  empty:       { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 40 },
  emptyIcon:   { fontSize: 56, marginBottom: 16 },
  emptyTitle:  { color: '#e8eaf0', fontSize: 20, fontWeight: '700', marginBottom: 8 },
  emptySubtitle: { color: '#5c6478', fontSize: 14, textAlign: 'center', lineHeight: 22 },
});
