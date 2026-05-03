import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import optuna
import json
from analysis import plot_unified_correlation

def apply_multidim_kmeans_clustering(X_train, X_test, y_train, n_clusters=5, verbose=True):
    """
    Tedarikçi bazlı gruplamayı veri sızıntısı olmadan yapar.
    """
    # 1. Profil çıkarma için geçici train seti
    train_temp = X_train.copy()
    train_temp['LEAD_TIME'] = y_train

    # 2. SADECE TRAIN verisi üzerinden Tedarikçi Profillerini Çıkar
    vendor_profiles = train_temp.groupby('VENDORFINAL').agg(
        mean_lead_time=('LEAD_TIME', 'mean')
    ).fillna(0)

    # 3. K-Means öncesi veriyi ÖLÇEKLENDİR (Farklı birimleri eşitlemek için ŞART)
    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(vendor_profiles)

    # 4. K-Means Modelini eğit
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    vendor_profiles['VENDOR_CLUSTER'] = 'Cluster_' + kmeans.fit_predict(scaled_profiles).astype(str)

    # 5. Eşleştirme sözlüğünü oluştur
    cluster_map = vendor_profiles['VENDOR_CLUSTER'].to_dict()

    # 6. Kümeleri Train ve Test setlerine uygula (Map)
    X_train['VENDOR_GROUP'] = X_train['VENDORFINAL'].map(cluster_map).fillna('Cluster_New')
    X_test['VENDOR_GROUP'] = X_test['VENDORFINAL'].map(cluster_map).fillna('Cluster_New')

    # 7. Orijinal VENDORFINAL sütununu artık silebiliriz
    X_train.drop(columns=['VENDORFINAL'], inplace=True)
    X_test.drop(columns=['VENDORFINAL'], inplace=True)

    return X_train, X_test

def print_cv_summary_report(metrics):
    """
    Çapraz doğrulama (CV) metriklerinin ortalama ve standart sapmalarını
    hesaplayarak terminale şık bir rapor basar.
    """
    print("\n" + "=" * 55)
    print("[XGBOOST CV NİHAİ RAPOR]")
    print("=" * 55)
    print(f"  {'R2 Score (Avg ± Std)':<25} : {np.mean(metrics['test_r2']):.4f} ± {np.std(metrics['test_r2']):.4f}")
    print(f"  {'RMSE (Avg ± Std)':<25} : {np.mean(metrics['test_rmse']):.4f} ± {np.std(metrics['test_rmse']):.4f}")
    print(f"  {'Test MAE (Avg ± Std)':<25} : {np.mean(metrics['test_mae']):.4f} ± {np.std(metrics['test_mae']):.4f}")
    print(f"  {'Train MAE (Avg ± Std)':<25} : {np.mean(metrics['train_mae']):.4f} ± {np.std(metrics['train_mae']):.4f}")
    print("=" * 55)
    print(f"  {'MAE FARKI (GAP) (%)':<25} : %{np.mean(metrics['gap']):.2f} ± %{np.std(metrics['gap']):.2f}")
    print("=" * 55)

def print_feature_importance(fold_importances, feature_names, model_name="XGBOOST"):
    """
    K-Fold'lardan gelen özellik önemlerini alır, ortalama/standart sapmasını hesaplar
    ve terminale formatlı bir şekilde yazdırır.
    """
    print(f"\n" + "=" * 65)
    print(f"[{model_name} - FEATURE IMPORTANCE (En Önemli İlk 20 Özellik)]")
    print("=" * 65)

    # Ortalama ve standart sapmayı hesapla
    avg_importance = np.mean(fold_importances, axis=0)
    std_importance = np.std(fold_importances, axis=0)

    # DataFrame'i oluştur ve büyükten küçüğe sırala
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Mean': avg_importance,
        'Importance_Std': std_importance
    }).sort_values(by='Importance_Mean', ascending=False)

    # En önemli ilk 20 özelliği al
    top_features = importance_df.head(20)

    # Sütun başlıklarını yazdır
    print(f"{'Özellik (Feature)':<25} | {'Ortalama Önem (%)':<20} | {'Standart Sapma (±)'}")
    print("-" * 65)

    # Satırları tek tek formatlayarak yazdır
    for index, row in top_features.iterrows():
        print(f"{row['Feature']:<25} | %{row['Importance_Mean']:<19.2f} | ± %{row['Importance_Std']:.2f}")

    print("=" * 65 + "\n")

def run_cross_validation(df, features_list, target='LEAD_TIME', n_splits=5, xgb_params=None):
    """
    Belirlenen özellikler üzerinden K-Fold Çapraz Doğrulama (Cross-Validation) yapar.
    """
    # 1. ÖZELLİK SEÇİMİ
    X = df[[c for c in features_list if c in df.columns]].copy()
    y = df[target]

    # 2. K-FOLD YAPISININ KURULMASI
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    metrics = {
        'test_mae': [], 'train_mae': [],
        'test_r2': [], 'test_rmse': [], 'gap': []
    }
    fold_importances = []
    feature_names = None

    # 3. CROSS-VALIDATION DÖNGÜSÜ
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # KRİTİK ADIM: K-MEANS GRUPLAMA
        X_train, X_test = apply_multidim_kmeans_clustering(X_train, X_test, y_train, verbose=False)

        # 4. KATEGORİK DEĞİŞKEN DÖNÜŞÜMÜ
        cat_cols = X_train.select_dtypes(include=['object']).columns
        for col in cat_cols:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

        # 5. PARAMETRE ENTEGRASYONU (Sade ve Modern Nesne Yönelimli Yaklaşım)
        if xgb_params is None:
            xgb_params = {}

        model = xgb.XGBRegressor(**xgb_params)

        # Zorunlu teknik ayarları ve erken durdurmayı set_params ile ekle
        model.set_params(
            enable_categorical=True,
            tree_method='hist',
            objective='reg:squarederror',
            eval_metric='mae',
            random_state=42
        )

        if 'early_stopping_rounds' not in xgb_params:
            model.set_params(early_stopping_rounds=100)

        # 6. EĞİTİM (fit_params hatası düzeltildi, doğrudan verbose=False eklendi)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

        # 7. METRİKLERİ HESAPLA VE KAYDET
        results = model.evals_result()
        f_train_mae = results['validation_0']['mae'][-1]
        f_test_mae = results['validation_1']['mae'][-1]
        preds = model.predict(X_test)
        f_r2 = r2_score(y_test, preds)
        f_rmse = np.sqrt(mean_squared_error(y_test, preds))
        f_gap = ((f_test_mae - f_train_mae) / f_train_mae) * 100

        metrics['test_mae'].append(f_test_mae)
        metrics['train_mae'].append(f_train_mae)
        metrics['test_r2'].append(f_r2)
        metrics['test_rmse'].append(f_rmse)
        metrics['gap'].append(f_gap)

        # 8. ÖZELLİK ÖNEMLERİNİ KAYDET
        if fold == 0:
            feature_names = X_train.columns.tolist()

        if hasattr(model, 'feature_importances_'):
            raw_importances = model.feature_importances_
            normalized_importances = (raw_importances / raw_importances.sum()) * 100
            fold_importances.append(normalized_importances)

        # 9. FOLD İLERLEMESİNİ YAZDIR
        print(
            f" Fold {fold + 1} | Test MAE: {f_test_mae:.2f} | Train MAE: {f_train_mae:.2f} | R2: {f_r2:.4f} | RMSE: {f_rmse:.2f} | Gap: %{f_gap:.2f}")

    # ==========================================
    # DÖNGÜ BİTTİ - HARİCİ RAPORLAMA ÇAĞRILARI
    # ==========================================

    # Tüm CV Raporu
    print_cv_summary_report(metrics)

    # Özellik önemleri tablosu (Güvenlik if bloğu eklendi)
    if fold_importances:
        print_feature_importance(fold_importances, feature_names, "XGBOOST")

    # Korelasyon matrisi
    plot_unified_correlation(X_train, y_train, target_name=target)

    return model

def optimize_xgboost(df, features_list, target='LEAD_TIME', n_splits=5, n_trials=50):
    """
    Optuna kullanarak XGBoost hiperparametrelerini K-Fold CV ile optimize eder.
    HIZLANDIRILMIŞ VERSİYON: K-Means önceden hesaplanır.
    """
    # 1. HATA DÜZELTİLDİ: Manuel liste silindi, features_list kullanılıyor.
    X = df[[c for c in features_list if c in df.columns]].copy()
    y = df[target]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    precomputed_folds = []

    print("\nOptuna için veriler bölünüyor ve K-Means kümeleri önceden hesaplanıyor...")

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train, X_test = apply_multidim_kmeans_clustering(X_train, X_test, y_train, verbose=False)

        cat_cols = X_train.select_dtypes(include=['object']).columns
        for col in cat_cols:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

        precomputed_folds.append((X_train, X_test, y_train, y_test))

    print("Ön hesaplama tamamlandı. XGBoost Optimizasyonu başlıyor (Lütfen bekleyin)...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000, step=500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'enable_categorical': True,
            'tree_method': 'hist',
            'random_state': 42,
            'early_stopping_rounds': 50 # ERKEN DURDURMA EKLENDİ
        }

        fold_maes = []
        for X_train, X_test, y_train, y_test in precomputed_folds:
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            preds = model.predict(X_test)
            fold_maes.append(mean_absolute_error(y_test, preds))

        return np.mean(fold_maes)

    # optuna.logging.WARNING satırını siliyoruz veya INFO yapıyoruz
    optuna.logging.set_verbosity(optuna.logging.INFO)  # <-- INFO yaparsan her denemeyi yazar

    study = optuna.create_study(direction='minimize')

    # show_progress_bar=True ekleyerek görsel bir bar da görebilirsin (Opsiyonel)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def train_and_save_final_model(df, features_list, target='LEAD_TIME', xgb_params=None):
    """
    Tüm veri seti ile nihai modeli eğitir.
    Tedarikçi K-Means sözlüğünü ve Modeli bilgisayara kaydeder.
    """
    import json

    X = df[[c for c in features_list if c in df.columns]].copy()
    y = df[target]

    print("\n[CANLIYA ALMA] K-Means Tedarikçi Grupları oluşturuluyor ve kaydediliyor...")
    # 1. K-Means ve Sözlük Kaydı
    train_temp = X.copy()
    train_temp['LEAD_TIME'] = y
    vendor_profiles = train_temp.groupby('VENDORFINAL').agg(mean_lead_time=('LEAD_TIME', 'mean')).fillna(0)

    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(vendor_profiles)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    vendor_profiles['VENDOR_CLUSTER'] = 'Cluster_' + kmeans.fit_predict(scaled_profiles).astype(str)

    cluster_map = vendor_profiles['VENDOR_CLUSTER'].to_dict()

    # Haritayı kaydet
    with open('vendor_cluster_map.json', 'w') as f:
        json.dump(cluster_map, f, indent=4)

    X['VENDOR_GROUP'] = X['VENDORFINAL'].map(cluster_map).fillna('Cluster_New')
    X.drop(columns=['VENDORFINAL'], inplace=True)

    # 2. Kategorik Dönüşüm ve Kategori İsimlerini Kaydetme
    cat_cols = X.select_dtypes(include=['object']).columns
    categories_dict = {}
    for col in cat_cols:
        X[col] = X[col].astype('category')
        categories_dict[col] = list(X[col].cat.categories)

    with open('category_map.json', 'w') as f:
        json.dump(categories_dict, f, indent=4)

        # YENİ: Seçilen özellik listesini JSON olarak kaydet
    with open('trained_features.json', 'w') as f:
        json.dump(features_list, f, indent=4)
    print("[KAYIT] Kullanılan özellik listesi 'trained_features.json' olarak kaydedildi.")

    # 3. Nihai Model Eğitimi
    print("[CANLIYA ALMA] Nihai XGBoost modeli tüm veriyle eğitiliyor...")
    model = xgb.XGBRegressor(**xgb_params)
    model.set_params(enable_categorical=True, tree_method='hist', objective='reg:squarederror')

    model.fit(X, y, verbose=False)

    # Modeli Kaydet
    model.save_model('final_xgboost_model.json')
    print(
        "[CANLIYA ALMA BAŞARILI] 'final_xgboost_model.json', 'vendor_cluster_map.json' ve 'category_map.json' kaydedildi!")

    return model