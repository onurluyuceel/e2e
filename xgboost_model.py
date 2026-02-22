import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler  # ÇOK BOYUTLU K-MEANS İÇİN YENİ EKLENDİ

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
        mean_order_qty=('ORDER_MIKTAR', 'mean'),  # Sipariş Boyutu
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

    # Opsiyonel: Kümelerin karakteristiğini görmek için terminale bastırıyoruz
    print("\n--- K-MEANS TEDARİKÇİ PROFİLLERİ (KÜME ÖZETİ) ---")
    print(vendor_profiles.groupby('VENDOR_CLUSTER').mean().round(2))
    print("--------------------------------------------------\n")

    return X_train_new, X_test_new


def train_xgboost_model(df, target='LEAD_TIME'):
    # 1. Özellik Seçimi
    features = [
        'LOAD_ITEM', 'ORDER_MIKTAR', 'MAINPART', 'VENDORFINAL',
        'MATERIALTYPE_NEW', 'IS_CONDITION_CHANGED', 'GEOMETRIC_GROUP',
        'LENGTH', 'CALC_VOLUME'
    ]

    X = df[features].copy()
    y = df[target]

    # 2. Veri Bölme (İlk sıraya alındı)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ---> YENİ: ÇOK BOYUTLU PROFİLLEME İLE K-MEANS <---
    X_train, X_test = apply_multidim_kmeans_clustering(X_train, X_test, y_train, n_clusters=4)

    # 3. Kategorik Dönüşüm (XGBoost için Şart - Hızlı Yöntem)
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        train_uniques = X_train[col].dropna().unique()
        test_uniques = X_test[col].dropna().unique()
        all_categories = np.union1d(train_uniques, test_uniques)

        cat_dtype = pd.CategoricalDtype(categories=all_categories, ordered=False)
        X_train[col] = X_train[col].astype(cat_dtype)
        X_test[col] = X_test[col].astype(cat_dtype)

    # 4. Model Parametreleri
    model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=4,
        reg_lambda=120,
        reg_alpha=10,
        min_child_weight=10,
        gamma=2,
        subsample=0.6,
        colsample_bytree=0.6,
        objective='reg:squarederror',
        eval_metric='mae',
        early_stopping_rounds=100,
        enable_categorical=True,
        tree_method='hist',
        random_state=42
    )

    # 5. Eğitim
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=500
    )

    # 6. Metrikler
    results = model.evals_result()

    # Son adımdaki MAE değerlerini alıyoruz
    final_train_mae = results['validation_0']['mae'][-1]
    final_test_mae = results['validation_1']['mae'][-1]

    # Yüzdesel fark (Gap) hesaplama
    mae_gap_percentage = ((final_test_mae - final_train_mae) / final_train_mae) * 100

    # Tahminler ve Diğer Metrikler
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # 7. Performans Raporu
    print("\n" + "=" * 40)
    print("[XGBOOST PERFORMANS RAPORU]")
    print("=" * 40)
    print(f"  {'R2 Score':<22} : {r2:.4f}")
    print(f"  {'RMSE':<22} : {rmse:.4f}")
    print(f"  {'Test MAE':<22} : {final_test_mae:.4f} Gün")
    print(f"  {'Train MAE':<22} : {final_train_mae:.4f} Gün")
    print("=" * 40)
    print(f"  {'MAE FARKI (GAP)':<22} : %{mae_gap_percentage:.2f}")
    print("=" * 40)

    # 8. Özellik Önemi (Güncel İsim Listesi İle)
    current_features = X_train.columns.tolist()
    importance_df = pd.DataFrame({
        'feature': current_features,
        'importance': model.feature_importances_ * 100
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    print("\n[ÖZELLİK ÖNEMİ]")
    print(importance_df)

    return model