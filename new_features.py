import numpy as np
import pandas as pd

def add_new_material_classes(df, mapping_file='malzeme_mapping.xlsx'):
    """MATERIALTYPE sütununa göre malzeme sınıflarını Excel'den okuyarak ekler."""

    try:
        # Excel dosyasını oku
        mapping_df = pd.read_excel(mapping_file, dtype=str)

        # Sütun isimleri ne olursa olsun 1. sütunu anahtar (key), 2. sütunu değer (value) yapıyoruz
        # dropna() ile boş olan satırları işleme almaktan kaçınıyoruz
        mapping_df = mapping_df.dropna(subset=[mapping_df.columns[0], mapping_df.columns[1]])

        # İki sütunu bir sözlüğe (dictionary) dönüştür
        material_map = dict(zip(mapping_df.iloc[:, 0], mapping_df.iloc[:, 1]))

    except FileNotFoundError:
        print(f"HATA: '{mapping_file}' bulunamadı! 'MATERIALTYPE_NEW' sütunu sadece 'Diğer' olarak doldurulacak.")
        material_map = {}

    # Yeni sütunu oluşturuyoruz. Eğer Excel'de olmayan bir tip gelirse 'Diğer' olarak işaretler.
    df['MATERIALTYPE_NEW'] = df['MATERIALTYPE'].map(material_map).fillna('Diğer')

    return df

def add_geometric_groups(df, mapping_file='geometri_mapping.xlsx'):
    """DIMENSIONCODE sütununa göre geometrik grupları Excel'den okuyarak ekler ve hacim hesaplar."""

    # --- Adım 1: Excel'den Eşleştirme (Mapping) ve 'DIGER' Kuralı ---
    try:
        mapping_df = pd.read_excel(mapping_file)
        mapping_df = mapping_df.dropna(subset=[mapping_df.columns[0], mapping_df.columns[1]])
        mapping_dict = dict(zip(mapping_df.iloc[:, 0], mapping_df.iloc[:, 1]))
    except FileNotFoundError:
        print(f"HATA: '{mapping_file}' bulunamadı! 'GEOMETRIC_GROUP' sütunu sadece 'DIGER' olarak doldurulacak.")
        mapping_dict = {}

    # Eşleşmeyenleri tekrar 'DIGER' olarak atıyoruz
    df['GEOMETRIC_GROUP'] = df['DIMENSIONCODE'].map(mapping_dict).fillna('DIGER')

    # --- Adım 2: Eksik Veri Doldurma (Gage ve Width için 50 kuralı) ---
    non_round_mask = df['GEOMETRIC_GROUP'] != 'ROUND'

    for col in ['GAGE', 'WIDTH']:
        if col in df.columns:
            df.loc[non_round_mask & ((df[col] == 0) | (df[col].isna())), col] = 50

    # --- Adım 3: Alan Hesaplama ---
    df['CS_AREA'] = 0.0

    # ROUND (Yuvarlak formlar)
    round_mask = df['GEOMETRIC_GROUP'] == 'ROUND'
    if 'OUTERDIAMETER' in df.columns:
        df.loc[round_mask, 'CS_AREA'] = (df['OUTERDIAMETER'] ** 2) * 0.7854

    # DİĞERLERİ (Dikdörtgen formlar ve DIGER olanlar)
    if 'WIDTH' in df.columns and 'GAGE' in df.columns:
        df.loc[non_round_mask, 'CS_AREA'] = df['WIDTH'] * df['GAGE']

    # --- Adım 4: Hacim Hesaplama (Kati Hata Kontrolü) ---
    # 1. Sütun hiç yoksa hata ver
    if 'LENGTH' not in df.columns:
        raise KeyError("KRİTİK HATA: Veri setinde 'LENGTH' sütunu bulunamadı! Hacim hesaplanamıyor.")

    # 2. Sütun var ama içinde boş (NaN) değerler varsa hata ver
    if df['LENGTH'].isna().any():
        raise ValueError(
            "KRİTİK HATA: 'LENGTH' sütununda eksik (NaN) değerler tespit edildi! Lütfen veriyi temizleyin.")

    # Tüm kontrollerden geçerse normal formülü uygula
    df['CALC_VOLUME'] = df['CS_AREA'] * df['LENGTH']

    return df


def add_length_groups(df):
    """GAGE, WIDTH, LENGTH, OUTERDIAMETER arasından en büyük olanı bulup sayısal gruplara (1, 2, 3...) ayırır."""

    dim_cols = ['GAGE', 'WIDTH', 'LENGTH', 'OUTERDIAMETER']

    # 1. Sütun Varlığı Kontrolü
    available_cols = [col for col in dim_cols if col in df.columns]

    if not available_cols:
        raise KeyError(
            "KRİTİK HATA: Boyut analizi için gereken ('GAGE', 'WIDTH', 'LENGTH', 'OUTERDIAMETER') sütunlarının HİÇBİRİ veri setinde yok!")

    # 2. Her satır için EN BÜYÜK (max) değeri bul
    df['MAX_DIMENSION'] = df[available_cols].max(axis=1)

    # 3. Eksik Veri (NaN) Kontrolü
    if df['MAX_DIMENSION'].isna().any():
        hatali_satir_sayisi = df['MAX_DIMENSION'].isna().sum()
        raise ValueError(
            f"KRİTİK HATA: Ön işlemede (impute) sorun var! Tam {hatali_satir_sayisi} satırda boyut verilerinin tamamı eksik (NaN).")

    global_max = df['MAX_DIMENSION'].max()

    # 4. Dinamik 500'lük aralıkları hesapla
    upper_limit = int(np.ceil(global_max / 500.0) * 500)
    if upper_limit == 0:
        upper_limit = 500

    bins = list(range(0, upper_limit + 500, 500))
    bins.append(float('inf'))

    # 5. Veriyi SAYISAL gruplara yerleştir
    df['LENGTH_GROUP_NUM'] = pd.cut(
        df['MAX_DIMENSION'],
        bins=bins,
        labels=False,
        right=True,
        include_lowest=True
    ) + 1

    # 6. Geçici MAX_DIMENSION sütununu temizle
    df = df.drop(columns=['MAX_DIMENSION'])

    return df


# new_features.py dosyasının sonuna veya uygun bir yerine ekle

def add_material_geometric_interaction(df):
    """MATERIALTYPE_NEW ve GEOMETRIC_GROUP sütunlarını birleştirerek etkileşim özelliği yaratır."""

    # 1. Her iki sütunun da var olduğundan emin olalım
    if 'MATERIALTYPE_NEW' in df.columns and 'GEOMETRIC_GROUP' in df.columns:
        # 2. İki metni birleştirip yeni bir kategori oluşturuyoruz (Örn: "Çelik_ROUND")
        df['MAT_GEO_INTERACTION'] = df['MATERIALTYPE_NEW'].astype(str) + "_" + df['GEOMETRIC_GROUP'].astype(str)

        # 3. XGBoost'un tanıması için kategorik tipe çeviriyoruz
        df['MAT_GEO_INTERACTION'] = df['MAT_GEO_INTERACTION'].astype('category')

    return df


def add_time_features(df):
    """Sipariş açılış tarihinden lojistik açıdan kritik zaman özelliklerini türetir."""

    # 1. Tarih formatında olduğundan emin olalım
    if 'PO_CREATIONDATE' in df.columns:
        # Hata vermemesi için boş olmayanları seçelim
        mask = df['PO_CREATIONDATE'].notna()

        # A. Sipariş Hangi Ayda Verildi? (1-12) - Makro Sezonsallık (Çin Yılbaşı, Kış vs.)
        df.loc[mask, 'SIPARIS_AYI'] = df.loc[mask, 'PO_CREATIONDATE'].dt.month

        # B. Sipariş Haftanın Hangi Günü Verildi? (0: Pazartesi ... 4: Cuma, 5: Cmt, 6: Pazar)
        df.loc[mask, 'SIPARIS_GUNU'] = df.loc[mask, 'PO_CREATIONDATE'].dt.dayofweek

        # Boş (Na) kalan tarihleri güvenli bir varsayılan ile (örn: ay 6, gün 0) dolduralım ki XGBoost hata vermesin
        df['SIPARIS_AYI'] = df['SIPARIS_AYI'].fillna(6).astype(int)
        df['SIPARIS_GUNU'] = df['SIPARIS_GUNU'].fillna(0).astype(int)

    else:
        print("UYARI: 'PO_CREATIONDATE' bulunamadı, zaman özellikleri eklenemedi.")

    return df


def add_advanced_structural_features(df):
    """Fiziksel zorluk ve lojistik sınırları temsil eden ileri düzey etkileşimler türetir."""

    # ---------------------------------------------------------
    # 1. EN-BOY ORANI (ASPECT RATIO)
    # Parça ince/uzun mu, yoksa orantılı bir blok mu?
    # ---------------------------------------------------------
    df['ASPECT_RATIO'] = 0.0

    # Yuvarlak parçalar için (Uzunluk / Çap)
    round_mask = (df['GEOMETRIC_GROUP'] == 'ROUND') & (df['OUTERDIAMETER'] > 0)
    if 'OUTERDIAMETER' in df.columns and 'LENGTH' in df.columns:
        df.loc[round_mask, 'ASPECT_RATIO'] = df['LENGTH'] / df['OUTERDIAMETER']

    # Diğer parçalar için (Uzunluk / Genişlik)
    non_round_mask = (df['GEOMETRIC_GROUP'] != 'ROUND') & (df['WIDTH'] > 0)
    if 'WIDTH' in df.columns and 'LENGTH' in df.columns:
        df.loc[non_round_mask, 'ASPECT_RATIO'] = df['LENGTH'] / df['WIDTH']

    # Sonsuz (Inf) veya eksik değerleri temizle (hata vermemesi için)
    df['ASPECT_RATIO'] = df['ASPECT_RATIO'].replace([np.inf, -np.inf], 0).fillna(0)

    # ---------------------------------------------------------
    # 2. ŞEKİL ve BOYUT ETKİLEŞİMİ (Lojistik Zorluk)
    # Örn: "PLATE_1" (Kolay) vs "PLATE_5" (Zor)
    # ---------------------------------------------------------
    if 'GEOMETRIC_GROUP' in df.columns and 'LENGTH_GROUP_NUM' in df.columns:
        df['GEO_SIZE_INTERACTION'] = df['GEOMETRIC_GROUP'].astype(str) + "_" + df['LENGTH_GROUP_NUM'].astype(str)
        df['GEO_SIZE_INTERACTION'] = df['GEO_SIZE_INTERACTION'].astype('category')

    # ---------------------------------------------------------
    # 3. MALZEME ve BOYUT ETKİLEŞİMİ (Tedarik Zorluğu)
    # Örn: "Titanyum_5" (Özel Üretim) vs "Çelik_5" (Standart)
    # ---------------------------------------------------------
    if 'MATERIALTYPE_NEW' in df.columns and 'LENGTH_GROUP_NUM' in df.columns:
        df['MAT_SIZE_INTERACTION'] = df['MATERIALTYPE_NEW'].astype(str) + "_" + df['LENGTH_GROUP_NUM'].astype(str)
        df['MAT_SIZE_INTERACTION'] = df['MAT_SIZE_INTERACTION'].astype('category')

    return df

def add_features(df):
    df = df.copy()

    # Eski Fonksiyonlar
    df = add_new_material_classes(df)
    df = add_geometric_groups(df)
    df = add_length_groups(df)
    df = add_material_geometric_interaction(df)
    df = add_time_features(df)
    df = add_advanced_structural_features(df)

    return df