import numpy as np

def add_geometric_features(df):
    """Geometrik aileleri belirler ve hacim/alan hesaplar."""
    has_od = (df['OUTERDIAMETER'] > 0)
    has_gage_or_width = (df['GAGE'] > 0) | (df['WIDTH'] > 0)

    conditions = [
        has_od,
        (~has_od) & has_gage_or_width,
        (~has_od) & (~has_gage_or_width)
    ]
    choices = ['ROUND', 'FLAT', 'LINEAR']
    df['GEO_FAMILY'] = np.select(conditions, choices, default='OTHER')

    # FLAT için 10 kuralı
    is_flat = (df['GEO_FAMILY'] == 'FLAT')
    df.loc[is_flat, 'GAGE'] = df.loc[is_flat, 'GAGE'].replace(0, np.nan).fillna(10)
    df.loc[is_flat, 'WIDTH'] = df.loc[is_flat, 'WIDTH'].replace(0, np.nan).fillna(10)

    # Alan ve Hacim
    df['SECTION_AREA'] = 0.0
    df.loc[df['GEO_FAMILY'] == 'ROUND', 'SECTION_AREA'] = (df['OUTERDIAMETER'] ** 2) * 0.7853
    df.loc[df['GEO_FAMILY'] == 'FLAT', 'SECTION_AREA'] = df['GAGE'] * df['WIDTH']
    df.loc[df['GEO_FAMILY'] == 'LINEAR', 'SECTION_AREA'] = 1.0

    df['VOLUME_INDEX'] = df['SECTION_AREA'] * df['LENGTH']
    return df

def add_new_material_classes(df):
    """MATERIALTYPE sütununa göre malzeme sınıflarını (Malzeme Sınıfı) ekler."""
    material_map = {
        '6AL-4V (AB-1)': 'Titanyum',
        'TI-6AL-4V': 'Titanyum',
        '321': 'Çelik',
        '15-5PH': 'Çelik',
        '17-HPH': 'Çelik',
        'AISI 301': 'Çelik',
        'AISI321': 'Çelik',
        'CRES 304': 'Çelik',
        'PH13-8MO': 'Çelik',
        'AISI 41410': 'Çelik',
        'INCONEL 625': 'Nikel-krom',
        'INCONEL 718': 'Nikel-krom',
        'C63000': 'Nikel-Alüminyum-Bronze',
        'C64200': 'Alüminyum-Bronze',
        '2050': 'Alüminyum',
        '6061': 'Alüminyum',
        '7050': 'Alüminyum',
        '7075': 'Alüminyum',
        '2024-CLAD': 'Alüminyum',
        '5.1129-1': 'Polytetrafluoroethylene (PTFE)'
    }
    
    # Yeni sütunu oluşturuyoruz. Eğer tabloda olmayan bir tip gelirse 'Diğer' olarak işaretler.
    df['MATERIALTYPE_NEW'] = df['MATERIALTYPE'].map(material_map).fillna('Diğer')
    return df


import numpy as np


def add_geometric_groups(df):

    mapping = {
        'EXTRUSION METALLIC': 'EXTRUSION',
        'RECTANGULAR BAR METALLIC': 'RECTANGULAR',
        'SHEET METALLIC': 'SHEET METAL',
        'PLATE METALLIC': 'SHEET METAL',
        'FORGING(RAW)': 'SHEET METAL',
        'MESH METALLIC': 'SHEET METAL',
        'STD. FLAT, SEMI FIN PARTS': 'SHEET METAL',
        'ROUND BAR METALLIC': 'ROUND',
        'ROUND TUBE METALLIC': 'ROUND'
    }
    df['GEOMETRIC_GROUP'] = df['DIMENSIONCODE'].map(mapping)

    # Adım 2: Eksik Veri Doldurma (Gage ve Width için 50 kuralı)
    non_round_mask = df['GEOMETRIC_GROUP'] != 'ROUND'

    for col in ['GAGE', 'WIDTH']:
        df.loc[non_round_mask & ((df[col] == 0) | (df[col].isna())), col] = 50

    # Adım 3: Alan ve Hacim Hesaplama
    # SECTION_AREA -> CS_AREA (Cross-Section Area)
    df['CS_AREA'] = 0.0

    # ROUND
    round_mask = df['GEOMETRIC_GROUP'] == 'ROUND'
    df.loc[round_mask, 'CS_AREA'] = (df['OUTERDIAMETER'] ** 2) * 0.7854

    # DİĞERLERİ
    df.loc[non_round_mask, 'CS_AREA'] = df['WIDTH'] * df['GAGE']

    # VOLUME_INDEX -> CALC_VOLUME (Calculated Volume)
    df['CALC_VOLUME'] = df['CS_AREA'] * df.get('LENGTH', 1)

    return df


def add_condition_change_feature(df):
    """
    STARTCONDITION ve FINALCONDITION sütunlarını karşılaştırır.
    Farklılık varsa 1, aynıysa 0 değerini atar.
    """
    
    df['IS_CONDITION_CHANGED'] = (
            df['STARTCONDITION'].astype(str).str.strip() !=
            df['FINALCONDITION'].astype(str).str.strip()
    ).astype(int)

    return df

def add_features(df):
    df = df.copy()

    # Tüm alt fonksiyonları sırayla çalıştır
    df = add_new_material_classes(df)
    df = add_geometric_groups(df)
    df = add_condition_change_feature(df)

    return df
