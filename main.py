import os
import json
import pandas as pd
from preprocessing import preprocess_data
from new_features import add_features
from model_training import run_cross_validation, optimize_xgboost

def main():

    # 1. Veri Yükleme ve Temizleme (Preprocessing)
    print("Veri yükleniyor ve temizleniyor...")
    df = pd.read_excel('datav3.xlsx')
    df = preprocess_data(df)
    df.to_excel('data_processed.xlsx', index=False)
    print("Temizlenmiş veri 'data_processed.xlsx' adıyla kaydedildi.")

    # Merkezi tip dönüşümü: Veriyi tekrar okumaya gerek kalmadan bellekte devam ediyoruz
    df['VENDORFINAL'] = df['VENDORFINAL'].astype(str)

    # 2. Özellik Mühendisliği (Feature Engineering)
    print("Yeni özellikler türetiliyor...")
    df_featured = add_features(df)
    df_featured.to_excel('data_with_features.xlsx', index=False)
    print("Zenginleştirilmiş veri 'data_with_features.xlsx' adıyla kaydedildi.")

    selected_features = [
        'LOAD_ITEM', 'ORDER_MIKTAR', 'MAINPART', 'VENDORFINAL',
        'MATERIALTYPE_NEW', 'IS_CONDITION_CHANGED', 'GEOMETRIC_GROUP',
        'LENGTH', 'CALC_VOLUME'
    ]

    params_file = 'best_xgb_params.json'

    # Dosya var mı kontrol et
    if os.path.exists(params_file):
        print(f"Kaydedilmiş parametreler '{params_file}' dosyasından okunuyor...")
        with open(params_file, 'r') as f:
            best_xgb_params = json.load(f)

    else:
        print("\nParametre dosyası bulunamadı. Optimizasyon başlatılıyor...")
        best_xgb_params = optimize_xgboost(
            df_featured,
            features_list=selected_features,
            target='LEAD_TIME',
            n_splits=5,
            n_trials=50
        )

        # Bulunan parametreleri JSON olarak bilgisayara kaydet
        with open(params_file, 'w') as f:
            json.dump(best_xgb_params, f, indent=4)
        print(f"En iyi parametreler '{params_file}' adıyla kaydedildi.")

    print("\nKULLANILAN XGBOOST PARAMETRELERİ")
    for key, value in best_xgb_params.items():
        print(f"  {key}: {value}")

    # Modeli parametrelerle çalıştır
    xgb_cv_model = run_cross_validation(
        df_featured,
        features_list=selected_features,
        target='LEAD_TIME',
        n_splits=5,
        xgb_params=best_xgb_params
    )
    """
    # Nihai modeli eğit
    final_prod_model = train_and_save_final_model(
        df_featured,
        features_list=selected_features,
        target='LEAD_TIME',
        xgb_params=best_xgb_params
    )
    """
    print("\n[TAMAMLANDI] Tüm süreç başarıyla sonuçlandı.")

if __name__ == "__main__":
    main()