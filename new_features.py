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
    bins.append(float('inf'))-

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

def add_features(df):
    df = df.copy()

    # Tüm alt fonksiyonları sırayla çalıştır
    df = add_new_material_classes(df)
    df = add_geometric_groups(df)
    df = add_length_groups(df)

    return df