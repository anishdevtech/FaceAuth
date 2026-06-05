import { useState, useEffect, useCallback } from 'react';
import { createMMKV } from 'react-native-mmkv';
import en from './en';
import hi from './hi';
import te from './te';

export const i18nStorage = createMMKV({ id: 'faceauth-i18n' });

export type LanguageCode = 'en' | 'hi' | 'te';
export type TranslationKey = keyof typeof en;

const translations: Record<LanguageCode, typeof en> = { en, hi, te };

// Global listener for language changes
const listeners = new Set<() => void>();

export const setLanguage = (lang: LanguageCode) => {
  i18nStorage.set('app_language', lang);
  listeners.forEach((listener) => listener());
};

export const getLanguage = (): LanguageCode => {
  const lang = i18nStorage.getString('app_language');
  if (lang === 'hi' || lang === 'te') return lang as LanguageCode;
  return 'en';
};

export const t = (key: TranslationKey, params?: Record<string, string | number>) => {
  const lang = getLanguage();
  let str = translations[lang][key] || translations['en'][key] || key;
  
  if (params) {
    Object.keys(params).forEach(k => {
      str = str.replace(new RegExp(`{{${k}}}`, 'g'), String(params[k]));
    });
  }
  return str;
};

export const useTranslation = () => {
  const [lang, setLang] = useState<LanguageCode>(getLanguage());

  useEffect(() => {
    const listener = () => setLang(getLanguage());
    listeners.add(listener);
    return () => { listeners.delete(listener); };
  }, []);

  const translate = useCallback((key: TranslationKey, params?: Record<string, string | number>) => {
    let str = translations[lang][key] || translations['en'][key] || key;
    if (params) {
      Object.keys(params).forEach(k => {
        str = str.replace(new RegExp(`{{${k}}}`, 'g'), String(params[k]));
      });
    }
    return str;
  }, [lang]);

  return { t: translate, lang, setLanguage };
};
