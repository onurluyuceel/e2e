import pandas as pd
import numpy as np

def preprocess_data(df, is_training=True):

    df = df.copy()  # Güvenlik önlemi
    # --- STEP 0: INITIAL DROPS ---
    drop_initially = ['LOAD_ITEM1', 'PO_KEY', 'MAINPART', 'Yönetici Etkisi1', 'LEAD_TIME_START_DATE', 'LEAD_TIME_FINISH_DATE', 'MATERIALSPEC', 'STARTCONDITION', 'FINALCONDITION', 'SDR', 'BLRSZ', 'WOR', 'SCH', 'TAR', 'DET', 'DOC', 'FMP', 'TKM', 'HLD',
                    'CLS', 'ARL', 'MIK', 'OKS', 'INT', 'NPO', 'OSI', 'PKM', 'ITP', 'ADR', 'KPY']
    df = df.drop(columns=[c for c in drop_initially if c in df.columns])

    # --- STEP 1: TYPE CONVERSIONS ---
    string_cols = ['PLANT', 'MAINPART GRUP', 'VENDORFINAL', 'MATERIALTYPE', 'DIMENSIONCODE']
    for col in [c for c in string_cols if c in df.columns]:
        df[col] = df[col].astype(str).replace(['nan', 'None', ''], np.nan)

    date_cols = ['PO_CREATIONDATE', 'İLK_BARKOD_TARİH']
    for col in [c for c in date_cols if c in df.columns]:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    numeric_cols = ['LOAD_ITEM', 'ORDER_MIKTAR', 'Yönetici Etkisi', 'GAGE', 'WIDTH', 'LENGTH', 'OUTERDIAMETER', 'YUZEY_ISLEM', 'IDEAL_LEAD_TIME']
    for col in [c for c in numeric_cols if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- ÖZEL KURAL: 1 EKİM TARİH DÜZENLEMESİ ---
    # 1 Ekim kuralı sadece her iki tarih de varsa (Eğitimde) çalışır
    if 'İLK_BARKOD_TARİH' in df.columns and 'PO_CREATIONDATE' in df.columns:
        threshold_date = pd.Timestamp(2024, 10, 1)
        mask = (df['PO_CREATIONDATE'] < threshold_date) & (df['İLK_BARKOD_TARİH'] >= threshold_date)
        df.loc[mask, 'PO_CREATIONDATE'] = threshold_date

    # --- STEP 3: EKSİK VERİ TEMİZLİĞİ ---
    initial_rows = len(df)
    # Kritik Nokta: Tahminlemede 'İLK_BARKOD_TARİH' ve 'LEAD_TIME' zorunlu değildir
    ignore_cols = ['OUTERDIAMETER', 'GAGE', 'WIDTH', 'İLK_BARKOD_TARİH', 'LEAD_TIME']
    check_cols = [c for c in df.columns if c not in ignore_cols]

    # Zorunlu sütunlardan biri bile eksikse o satırı siler
    df = df.dropna(subset=check_cols, how='any').copy()
    na_deleted_count = initial_rows - len(df)

    # --- STEP 4: LEAD TIME & TRAINING LOGIC ---
    lt_deleted_count = 0
    if is_training:
        # Eğitim modunda bu sütunlar şart! Yoksa hata verir
        if 'İLK_BARKOD_TARİH' in df.columns and 'PO_CREATIONDATE' in df.columns:
            df.loc[:, 'LEAD_TIME'] = (df['İLK_BARKOD_TARİH'] - df['PO_CREATIONDATE']).dt.days
            rows_before_lt_filter = len(df)
            df = df[df['LEAD_TIME'] <= 150].copy()  # 1 yıldan uzun süren "hatalı" verileri sil
            lt_deleted_count = rows_before_lt_filter - len(df)
        else:
            raise KeyError("Eğitim (is_training=True) seçili ama 'İLK_BARKOD_TARİH' sütunu bulunamadı!")

    # --- CHECK ZONE ---
    print("\n" + "=" * 45)
    print("PREPROCESSING SEQUENTIAL REPORT")
    print("=" * 45)

    print("1. FINAL DATA TYPES:")
    print(df.dtypes)
    print("-" * 30)

    print(f"2. ROWS DELETED (Missing Mandatory Data): {na_deleted_count}")
    print(f"3. ROWS DELETED (Lead Time > 365 Days): {lt_deleted_count}")
    print("-" * 30)

    print(f"4. FINAL CLEANED ROW COUNT: {len(df)}")
    print("\nUPDATED MISSING VALUES STATUS (Should be 0 for all):")
    print(df.isna().sum())
    print("=" * 45 + "\n")
    # --- END OF CHECK ZONE ---

    return df
