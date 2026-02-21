import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans

def apply_leadtime_kmeans_clustering(X_train, X_test, y_train, n_clusters=4):
    """
    Sadece Train verisindeki Ortalama Lead Time değerlerini kullanarak
    Tedarikçileri K-Means ile kümeler. (Veri sızıntısı yapmaz)
    """
    # 1. Ekstra DataFrame yaratmadan doğrudan y_train'i X_train'deki sütuna göre grupla
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

def train_catboost_model(df, target='LEAD_TIME'):
    # 1. Seçili Özellikler
    features = [
        'LOAD_ITEM', 'ORDER_MIKTAR', 'MAINPART', 'VENDORFINAL',
        'MATERIALTYPE_NEW', 'IS_CONDITION_CHANGED', 'GEOMETRIC_GROUP',
        'LENGTH', 'CALC_VOLUME'
    ]

    X = df[features]
    y = df[target]

    # 1. Veri Bölme (Bölmeyi öne aldık)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # ---> YENİ: K-Means ile Gruplama <---
    X_train, X_test = apply_leadtime_kmeans_clustering(X_train, X_test, y_train, n_clusters=4)

    # 2. Kategorik Sütunları Belirleme ve Tip Dönüşümü
    # (Bunu K-Means'ten SONRA yapıyoruz çünkü VENDORFINAL gitti, VENDOR_GROUP geldi)
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # CatBoost için kategorik sütunların sadece 'string' tipinde olması yeterlidir.
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # 3. Model Parametreleri

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

    # 4. Eğitim
    model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_test, y_test))

    # 5. Metrikler
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

    # 6. Özellik Önemi
    print("\n[ÖZELLİK ÖNEMİ]")
    feature_importance = model.get_feature_importance(prettified=True)
    print(feature_importance)

    return model