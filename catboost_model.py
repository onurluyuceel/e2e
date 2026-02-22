import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def apply_multidim_kmeans_clustering(X_train, X_test, y_train, n_clusters=4):
    """
    Sadece Train verisindeki çoklu özellikleri kullanarak
    Tedarikçileri K-Means ile kümeler. (Veri sızıntısı yapmaz)
    """
    # 1. Profil çıkarma için geçici train seti
    train_temp = X_train.copy()
    train_temp['LEAD_TIME'] = y_train

    # 2. SADECE TRAIN verisi üzerinden Tedarikçi Profillerini Çıkar
    vendor_profiles = train_temp.groupby('VENDORFINAL').agg(
        mean_lead_time=('LEAD_TIME', 'mean'),
        std_lead_time=('LEAD_TIME', 'std'),  # İstikrar (Dalgalanma)
        order_count=('VENDORFINAL', 'count'),  # Frekans (Hacim)
        mean_order_qty=('ORDER_MIKTAR', 'mean')  # Sipariş Boyutu
    ).fillna(0)  # Sadece 1 siparişi olanların standart sapması NaN çıkar, onları 0 yapıyoruz.

    # 3. K-Means öncesi veriyi ÖLÇEKLENDİR (Farklı birimleri eşitlemek için ŞART)
    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(vendor_profiles)

    # 4. K-Means Modelini eğit
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    vendor_profiles['VENDOR_CLUSTER'] = 'Cluster_' + kmeans.fit_predict(scaled_profiles).astype(str)

    # 5. Eşleştirme sözlüğünü oluştur
    cluster_mapping = vendor_profiles['VENDOR_CLUSTER'].to_dict()

    # 6. Kümeleri Train ve Test setlerine uygula (Map)
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()

    X_train_new['VENDOR_GROUP'] = X_train_new['VENDORFINAL'].map(cluster_mapping).fillna('Unknown')
    X_test_new['VENDOR_GROUP'] = X_test_new['VENDORFINAL'].map(cluster_mapping).fillna('Unknown')

    # 7. Orijinal VENDORFINAL sütununu artık silebiliriz
    X_train_new.drop(columns=['VENDORFINAL'], inplace=True)
    X_test_new.drop(columns=['VENDORFINAL'], inplace=True)

    # Kümelerin karakteristiğini görmek için terminale bastırıyoruz
    print("\n--- K-MEANS TEDARİKÇİ PROFİLLERİ (KÜME ÖZETİ) ---")
    print(vendor_profiles.groupby('VENDOR_CLUSTER').mean().round(2))
    print("--------------------------------------------------\n")

    return X_train_new, X_test_new

def train_catboost_model(df, target='LEAD_TIME'):
    # 1. Seçili Özellikler
    features = [
        'LOAD_ITEM', 'ORDER_MIKTAR', 'MAINPART', 'VENDORFINAL',
        'MATERIALTYPE_NEW', 'IS_CONDITION_CHANGED', 'GEOMETRIC_GROUP',
        'LENGTH', 'CALC_VOLUME'
    ]

    # Hata almamak için veriyi kopyalıyoruz
    X = df[features].copy()
    y = df[target]

    # 2. Veri Bölme (Sızıntıyı önlemek için İlk sıraya alındı)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ---> YENİ: ÇOK BOYUTLU PROFİLLEME İLE K-MEANS <---
    X_train, X_test = apply_multidim_kmeans_clustering(X_train, X_test, y_train, n_clusters=4)

    # 3. Kategorik Sütunları Belirleme ve Tip Dönüşümü
    # (Bunu K-Means'ten SONRA yapıyoruz çünkü VENDORFINAL gitti, VENDOR_GROUP geldi)
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # CatBoost için kategorik sütunların sadece 'string' tipinde olması yeterlidir.
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # 4. Model Parametreleri
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=10,
        eval_metric='RMSE',
        early_stopping_rounds=100,
        verbose=500,
        allow_writing_files=False
    )

    # 5. Eğitim
    model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_test, y_test))

    # 6. Metrikler
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    r2_test = r2_score(y_test, test_preds)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_preds))
    mae_train = mean_absolute_error(y_train, train_preds)
    mae_test = mean_absolute_error(y_test, test_preds)

    # Gap (Sapma) Hesabı:
    if mae_train > 0:
        mae_gap_percentage = ((mae_test - mae_train) / mae_train) * 100
    else:
        mae_gap_percentage = 0.0

    print("\n" + "=" * 40)
    print(f"{'[CATBOOST PERFORMANS RAPORU]':^45}")
    print("=" * 40)
    print(f"  {'R2 Score':<20} : {r2_test:.4f}")
    print(f"  {'RMSE':<20} : {rmse_test:.4f}")
    print(f"  {'Test MAE':<20} : {mae_test:.4f}")
    print(f"  {'Train MAE':<20} : {mae_train:.4f}")
    print("=" * 40)
    print(f"  {'MAE FARKI (GAP)':<20} : %{mae_gap_percentage:.2f}")
    print("=" * 40)

    # 7. Özellik Önemi
    print("\n[ÖZELLİK ÖNEMİ]")
    feature_importance = model.get_feature_importance(prettified=True)
    print(feature_importance)

    return model