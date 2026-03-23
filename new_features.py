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
        '5.1129-1': 'Polytetrafluoroethylene (PTFE)',
        'AERMET 100': 'Çelik',
        '1.7734': 'Çelik',
        'AERO100': 'Çelik',
        '17-7PH': 'Çelik',
        '5052': 'Alüminyum',
        '2024': 'Alüminyum',
        '0.70 - 1.00C': 'Çelik',
        'AISI 304': 'Çelik',
        'SAE 4340': 'Çelik',
        'AISI 321': 'Çelik',
        '17-4PH': 'Çelik',
        'C10100': 'Bakır',
        'AISI 1010': 'Çelik',
        '7075-CLAD': 'Alüminyum',
        'TYPE 1': 'Tungsten',
        'AISI 4340': 'Çelik',
        'CRES 304 (4X4)': 'Çelik',
        'TUNGSTEN': 'Tungsten',
        'CP-2': 'Titanyum',
        '93 PB - 6.5 SB - 0.5 SN': 'Kurşun',
        '32CDV13': 'Çelik',
        'AISI 304 (D:0.023")': 'Çelik',
        'PYROWEAR 53': 'Çelik',
        '1.7734 (F-0225)': 'Çelik',
        'NYLATRON GS': 'Plastik',
        '9310': 'Çelik',
        'TI-3AL-2.5V': 'Titanyum',
        '1': 'Plastik',
        'A-286': 'Nikel Alaşım',
        'SAE 4130': 'Çelik',
        '2124': 'Alüminyum',
        '4340': 'Çelik',
        'C5': 'Çelik',
        'AISI 302': 'Çelik',
        '5.1129.20': 'Teflon PTFE'
    }
    
    # Yeni sütunu oluşturuyoruz. Eğer tabloda olmayan bir tip gelirse 'Diğer' olarak işaretler.
    df['MATERIALTYPE_NEW'] = df['MATERIALTYPE'].map(material_map).fillna('Diğer')
    return df

def add_geometric_groups(df):

    mapping = {
        'EXTRUSION METALLIC': 'EXTRUSION',
        'RECTANGULAR BAR METALLIC': 'PLATE',
        'SHEET METALLIC': 'SHEET',
        'PLATE METALLIC': 'PLATE',
        'FORGING(RAW)': 'PLATE',
        'MESH METALLIC': 'PLATE',
        'STD. FLAT, SEMI FIN PARTS': 'PLATE',
        'ROUND BAR METALLIC': 'ROUND',
        'ROUND TUBE METALLIC': 'ROUND',
        'SHEET&PLATE NON METALLIC': 'PLATE',
        'RECTANGULAR TUBE METALLIC': 'EXTRUSION',
        'SPRING WIRE': 'EXTRUSION',
        'ROD NON METALLIC': 'ROUND',
        'SHIM METALLIC': 'SHEET'
    }
    df['GEOMETRIC_GROUP'] = df['DIMENSIONCODE'].map(mapping).fillna('DIGER')

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