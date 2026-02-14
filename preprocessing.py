import pandas as pd
import numpy as np

def preprocess_data(df):
    # --- STEP 0: INITIAL DROPS ---
    drop_initially = ['PO_KEY', 'LEAD_TIME_START_DATE', 'LEAD_TIME_FINISH_DATE']
    df = df.drop(columns=[c for c in drop_initially if c in df.columns])

    # --- STEP 1: TYPE CONVERSIONS ---
    string_cols = ['LOAD_ITEM', 'PLANT', 'MAINPART', 'VENDORFINAL',
                   'MATERIALSPEC', 'MATERIALTYPE', 'STARTCONDITION', 'FINALCONDITION', 'DIMENSIONCODE']
    for col in [c for c in string_cols if c in df.columns]:
        df[col] = df[col].astype(str).replace(['nan', 'None', ''], np.nan)

    date_cols = ['PO_CREATIONDATE', 'İLK_BARKOD_TARİH']
    for col in [c for c in date_cols if c in df.columns]:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    numeric_cols = ['SDR', 'BLRSZ', 'WOR', 'SCH', 'TAR', 'DET', 'DOC', 'FMP', 'TKM', 'HLD',
                    'CLS', 'ARL', 'MIK', 'OKS', 'INT', 'NPO', 'OSI', 'PKM', 'ITP', 'ADR',
                    'ORDER_MIKTAR', 'GAGE', 'WIDTH', 'LENGTH', 'OUTERDIAMETER']
    for col in [c for c in numeric_cols if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- ÖZEL KURAL: 1 EKİM TARİH DÜZENLEMESİ ---
    threshold_date = pd.Timestamp(2024, 10, 1)

    # Mantık: PO < 1 Ekim VE Barkod >= 1 Ekim ise PO'yu 1 Ekim yap
    mask = (df['PO_CREATIONDATE'] < threshold_date) & (df['İLK_BARKOD_TARİH'] >= threshold_date)
    df.loc[mask, 'PO_CREATIONDATE'] = threshold_date

    # --- STEP 2: ALL FILTERING & DELETIONS ---
    # A. Mandatory Column Check (Missing Data Deletion)
    initial_rows = len(df)
    ignore_cols = ['OUTERDIAMETER', 'GAGE', 'WIDTH']
    check_cols = [c for c in df.columns if c not in ignore_cols]
    df = df.dropna(subset=check_cols, how='any').copy()
    na_deleted_count = initial_rows - len(df)

    # B. Lead Time Calculation & Filtering (Logical Deletion)
    df.loc[:, 'LEAD_TIME'] = (df['İLK_BARKOD_TARİH'] - df['PO_CREATIONDATE']).dt.days
    rows_before_lt_filter = len(df)
    df = df[df['LEAD_TIME'] <= 365].copy()
    lt_deleted_count = rows_before_lt_filter - len(df)

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
