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

    # DOĞRU OLAN (Temizlenmiş veriye ekle):
    df['TAHMIN_EDILEN_SURE (GUN)'] = tahminler.round(0).astype(int)
    df.to_excel('tahmin_sonuclari.xlsx', index=False)

    print("\n[BAŞARILI] Tahminler yapıldı ve 'tahmin_sonuclari.xlsx' olarak kaydedildi!")


if __name__ == "__main__":
    main()