import React from 'react';
import { StyleSheet, View, Text, TouchableOpacity, Modal } from 'react-native';
import { useTranslation, LanguageCode } from '../i18n';
import { Check } from 'lucide-react-native';

interface Props {
  visible: boolean;
  onClose: () => void;
}

export const LanguagePickerModal: React.FC<Props> = ({ visible, onClose }) => {
  const { lang, setLanguage } = useTranslation();

  const handleSelect = (code: LanguageCode) => {
    setLanguage(code);
    onClose();
  };

  return (
    <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}>
      <TouchableOpacity style={styles.backdrop} activeOpacity={1} onPress={onClose}>
        <View style={styles.card}>
          <Text style={styles.title}>Select Language</Text>
          <View style={styles.options}>
            <Option label="English" code="en" active={lang === 'en'} onSelect={handleSelect} />
            <Option label="हिन्दी" code="hi" active={lang === 'hi'} onSelect={handleSelect} />
            <Option label="తెలుగు" code="te" active={lang === 'te'} onSelect={handleSelect} />
          </View>
        </View>
      </TouchableOpacity>
    </Modal>
  );
};

const Option = ({ label, code, active, onSelect }: any) => (
  <TouchableOpacity style={[styles.option, active && styles.optionActive]} onPress={() => onSelect(code)}>
    <Text style={[styles.optionText, active && styles.optionTextActive]}>{label}</Text>
    {active && <Check size={20} color="#f0a500" />}
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'center', padding: 20 },
  card: { backgroundColor: '#1C1C1E', borderRadius: 24, padding: 24, borderWidth: 1, borderColor: '#2d3748' },
  title: { color: '#e8eaf0', fontSize: 24, fontFamily: 'BarlowCondensed-Bold', marginBottom: 20, textAlign: 'center', textTransform: 'uppercase' },
  options: { gap: 12 },
  option: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 16, borderRadius: 12, backgroundColor: '#0d1117', borderWidth: 1, borderColor: '#1e2433' },
  optionActive: { borderColor: '#f0a500' },
  optionText: { color: '#6b7280', fontSize: 16, fontFamily: 'DMSans-SemiBold' },
  optionTextActive: { color: '#e8eaf0' },
});
