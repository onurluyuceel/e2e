import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans

def apply_leadtime_kmeans_clustering(X_train, X_test, y_train, n_clusters=4):
    """
    Sadece Train verisindeki Ortalama Lead Time değerlerini kullanarak
    Tedarikçileri K-Means ile kümeler. (Veri sızıntısı yapmaz)
    """
    vendor_lt_mean = y_train.groupby(X_train['VENDORFINAL']).mean().to_frame(name='LEAD_TIME')

    # 2. K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    vendor_lt_mean['VENDOR_CLUSTER'] = 'Cluster_' + kmeans.fit_predict(vendor_lt_mean[['LEAD_TIME']]).astype(str)

    # Eşleştirme sözlüğü
    cluster_mapping = vendor_lt_mean['VENDOR_CLUSTER'].to_dict()

    # 3. Kopyalama
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()

    # 4. Map ve Fillna işlemlerini ZİNCİRLE (Tek Satırda)
    X_train_new['VENDOR_GROUP'] = X_train_new['VENDORFINAL'].map(cluster_mapping).fillna('Unknown')
    X_test_new['VENDOR_GROUP'] = X_test_new['VENDORFINAL'].map(cluster_mapping).fillna('Unknown')

    # 5. Orijinal sütunu düşür
    X_train_new.drop(columns=['VENDORFINAL'], inplace=True)
    X_test_new.drop(columns=['VENDORFINAL'], inplace=True)

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

    # 3. Veri Bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ---> K-Means ile Sadece Lead Time Üzerinden Gruplama <---
    X_train, X_test = apply_leadtime_kmeans_clustering(X_train, X_test, y_train, n_clusters=4)

    # 2. Kategorik Dönüşüm (XGBoost için Şart)
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in cat_cols:
        # Numpy union1d ile Train ve Test'teki benzersiz değerlerin kümesini al (Çok Daha Hızlı)
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

    # 8. Özellik Önemi
    # DİKKAT: Artık features listesi yerine modelin gerçekten eğitildiği
    # güncel sütun isimlerini (X_train.columns.tolist()) kullanıyoruz!
    current_features = X_train.columns.tolist()

    importance_df = pd.DataFrame({
        'feature': current_features,
        'importance': model.feature_importances_ * 100
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    print("\n[ÖZELLİK ÖNEMİ]")
    print(importance_df)

    return model