import pandas as pd
import xgboost as xgb
import json
import os
from preprocessing import preprocess_data
from new_features import add_features


def main():
    yeni_veri_dosyasi = 'Lift-up_tahminleme_verisiv2.xlsx'  # Arkadaşının vereceği 200 satırlık dosya

    if not os.path.exists(yeni_veri_dosyasi):
        print(f"HATA: '{yeni_veri_dosyasi}' adlı Excel dosyası bulunamadı!")
        return

    print("1. Yeni siparişler yükleniyor ve temizleniyor...")
    df_raw = pd.read_excel(yeni_veri_dosyasi)

    # DİKKAT: is_training=False gönderiyoruz ki LEAD_TIME boş diye satırları silmesin!
    df = preprocess_data(df_raw, is_training=False)
    df['VENDORFINAL'] = df['VENDORFINAL'].astype(str)

    print("2. Özellikler (Feature Engineering) türetiliyor...")
    df_featured = add_features(df)

    print("3. Eğitimde kullanılan özellik listesi yükleniyor...")
    # Manuel liste yerine dosyadan okuyoruz!
    with open('trained_features.json', 'r') as f:
        selected_features = json.load(f)

    # Artık X_new sadece bu listedeki sütunları otomatik seçecek
    X_new = df_featured[[c for c in selected_features if c in df_featured.columns]].copy()

    print("3. K-Means Tedarikçi haritası uygulanıyor...")
    with open('vendor_cluster_map.json', 'r') as f:
        cluster_map = json.load(f)

    X_new['VENDOR_GROUP'] = X_new['VENDORFINAL'].map(cluster_map).fillna('Cluster_New')
    X_new.drop(columns=['VENDORFINAL'], inplace=True)

    print("4. Kategorik veri tipleri ayarlanıyor...")
    with open('category_map.json', 'r') as f:
        categories_dict = json.load(f)

    for col, categories in categories_dict.items():
        if col in X_new.columns:
            # Eğitimdeki aynı kategorileri zorluyoruz ki model hata vermesin
            X_new[col] = pd.Categorical(X_new[col], categories=categories)

    print("5. Yapay Zeka modeli yükleniyor ve TAHMİN yapılıyor...")
    model = xgb.XGBRegressor()
    model.load_model('final_xgboost_model.json')

    # Tahmin yap!
    tahminler = model.predict(X_new)

    # Tahminleri orijinal veriye ekle ve yuvarla (Örn: 42.6 gün -> 43 gün)
    df_raw['TAHMIN_EDILEN_SURE (GUN)'] = tahminler
    df_raw['TAHMIN_EDILEN_SURE (GUN)'] = df_raw['TAHMIN_EDILEN_SURE (GUN)'].round(0).astype(int)


    # Sonucu kaydet
    df_raw.to_excel('tahmin_sonuclari.xlsx', index=False)
    print("\n[BAŞARILI] Tahminler yapıldı ve 'tahmin_sonuclari.xlsx' olarak kaydedildi!")


if __name__ == "__main__":
    main()
"""
import pandas as pd
import xgboost as xgb
import json
import os
from preprocessing import preprocess_data
from new_features import add_features


def main():
    # 1. DOSYA İSMİ DATAV3 OLARAK DEĞİŞTİRİLDİ
    yeni_veri_dosyasi = 'datav3.xlsx'
    output_dosyasi = 'datav3_karsilastirma_sonucu.xlsx'  # Çıktı ismini karışmaması için değiştirdik

    if not os.path.exists(yeni_veri_dosyasi):
        print(f"HATA: '{yeni_veri_dosyasi}' adlı Excel dosyası bulunamadı!")
        return

    print(f"1. {yeni_veri_dosyasi} yükleniyor ve temizleniyor...")
    df_raw = pd.read_excel(yeni_veri_dosyasi)

    # DİKKAT: is_training=False gönderiyoruz ki satırları silmesin!
    df = preprocess_data(df_raw, is_training=False)
    df['VENDORFINAL'] = df['VENDORFINAL'].astype(str)

    print("2. Özellikler (Feature Engineering) türetiliyor...")
    df_featured = add_features(df)

    print("3. Eğitimde kullanılan özellik listesi yükleniyor...")
    with open('trained_features.json', 'r') as f:
        selected_features = json.load(f)

    X_new = df_featured[[c for c in selected_features if c in df_featured.columns]].copy()

    print("4. K-Means Tedarikçi haritası uygulanıyor...")
    with open('vendor_cluster_map.json', 'r') as f:
        cluster_map = json.load(f)

    X_new['VENDOR_GROUP'] = X_new['VENDORFINAL'].map(cluster_map).fillna('Cluster_New')
    X_new.drop(columns=['VENDORFINAL'], inplace=True)

    print("5. Kategorik veri tipleri ayarlanıyor...")
    with open('category_map.json', 'r') as f:
        categories_dict = json.load(f)

    for col, categories in categories_dict.items():
        if col in X_new.columns:
            X_new[col] = pd.Categorical(X_new[col], categories=categories)

    print("6. Yapay Zeka modeli yükleniyor ve TAHMİN yapılıyor...")
    model = xgb.XGBRegressor()
    model.load_model('final_xgboost_model.json')

    # Tahmin yap!
    tahminler = model.predict(X_new)

    # --- DÜZELTİLEN BÖLÜM ---
    # Tahminleri df_raw'a DEĞİL, df'e (temizlenmiş veriye) ekliyoruz!
    df['TAHMIN_EDILEN_SURE (GUN)'] = tahminler.round(0).astype(int)

    # datav3'ün içinde tarihler olduğu için gerçek süreyi bulup sapmayı hesaplayabiliriz
    if 'İLK_BARKOD_TARİH' in df.columns and 'PO_CREATIONDATE' in df.columns:
        # Tarihleri datetime formatına çevirip farkı gün olarak al
        barkod = pd.to_datetime(df['İLK_BARKOD_TARİH'], dayfirst=True, errors='coerce')
        po_tarih = pd.to_datetime(df['PO_CREATIONDATE'], dayfirst=True, errors='coerce')

        df['GERCEK_SURE (GUN)'] = (barkod - po_tarih).dt.days

        # Sapma = Tahmin - Gerçek
        df['SAPMA (GUN)'] = df['TAHMIN_EDILEN_SURE (GUN)'] - df['GERCEK_SURE (GUN)']

    # Sonucu kaydet (Yine df_raw değil, df'i kaydediyoruz)
    df.to_excel(output_dosyasi, index=False)
    print(f"\n[BAŞARILI] {len(df)} satır için karşılaştırma yapıldı ve '{output_dosyasi}' olarak kaydedildi!")


if __name__ == "__main__":
    main()
    
"""