
import pandas as pd
from preprocessing import preprocess_data
from new_features import add_features
from analysis import run_all_analyses
from catboost_model import train_catboost_model


def main():
    
    # 1. Veri Yükleme ve Temizleme (Preprocessing)
    print("Veri yükleniyor ve temizleniyor...")
    df = pd.read_excel('data.xlsx')
    df = preprocess_data(df)

    # Temizlenmiş veriyi yedek olarak kaydet
    df.to_excel('data_processed.xlsx', index=False)
    print("Temizlenmiş veri 'data_processed.xlsx' adıyla kaydedildi.")
    

    df = pd.read_excel('data_processed.xlsx')

    # Merkezi tip dönüşümü: Veriyi tekrar okumaya gerek kalmadan bellekte devam ediyoruz
    df['VENDORFINAL'] = df['VENDORFINAL'].astype(str)

    # 2. Özellik Mühendisliği (Feature Engineering)
    print("Yeni özellikler türetiliyor...")
    df_featured = add_features(df)

    # Özellik eklenmiş veriyi kaydet
    df_featured.to_excel('data_with_features.xlsx', index=False)
    print("Zenginleştirilmiş veri 'data_with_features.xlsx' adıyla kaydedildi.")

    """
    # 3. Analizler
    print("Analizler başlatılıyor...")
    run_all_analyses(df_featured)
    """

    """
    # 4. Eğitim
    print("CatBoost model eğitimi başlatılıyor...")
    trained_model = train_catboost_model(df_featured)
    """

    print("\n[TAMAMLANDI] Tüm süreç başarıyla sonuçlandı.")

if __name__ == "__main__":
    main()
