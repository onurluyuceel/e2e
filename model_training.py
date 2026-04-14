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

    # Kümelerin karakteristiğini görmek için terminale bastırıyoruz
    print("\n--- K-MEANS TEDARİKÇİ PROFİLLERİ (KÜME ÖZETİ) ---")
    print(vendor_profiles.groupby('VENDOR_CLUSTER').mean().round(2))
    print("--------------------------------------------------\n")

    return X_train, X_test

def plot_feature_importance(importance_df, model_name):
    """Özellik önemlerini ortalama ve standart sapma hata çubuklarıyla çizer."""
    plt.figure(figsize=(10, 8))

    # En önemli ilk 20 özelliği al ve ters çevir (en yüksek değer en üstte görünsün)
    top_features = importance_df.head(20).copy()
    top_features = top_features.sort_values(by='Importance_Mean', ascending=True)

    # Matplotlib ile yatay bar çizimi (Hata çubukları ile kusursuz çalışır)
    plt.barh(
        y=top_features['Feature'],
        width=top_features['Importance_Mean'],
        xerr=top_features['Importance_Std'],
        color='teal',
        capsize=5,       # Hata çubuklarının ucundaki minik çizgiler
        edgecolor='black'
    )

    plt.title(f'Feature Importance - {model_name.upper()} (Mean ± Std over K-Folds)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


def run_cross_validation(df, features_list, target='LEAD_TIME', n_splits=5, xgb_params=None):
    """
    Belirlenen özellikler üzerinden K-Fold Çapraz Doğrulama (Cross-Validation) yapar.
    """
    # 1. ÖZELLİK SEÇİMİ (İçerideki manuel liste silindi, doğrudan parametre kullanılıyor)
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
        # verbose=(fold==0) sayesinde kümeleri terminale 5 kere değil, sadece 1 kere yazar.
        X_train, X_test = apply_multidim_kmeans_clustering(X_train, X_test, y_train, verbose=(fold == 0))

        # 4. KATEGORİK DEĞİŞKEN DÖNÜŞÜMÜ
        cat_cols = X_train.select_dtypes(include=['object']).columns
        for col in cat_cols:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

        # 5. PARAMETRE ENTEGRASYONU
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 3000,
                'learning_rate': 0.01,
                'max_depth': 4,
                'reg_lambda': 120,
                'reg_alpha': 10,
                'min_child_weight': 10,
                'gamma': 2,
                'subsample': 0.6,
                'colsample_bytree': 0.6,
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'early_stopping_rounds': 100,
                'enable_categorical': True,
                'tree_method': 'hist',
                'random_state': 42
            }
        else:
            xgb_params['enable_categorical'] = True
            xgb_params['tree_method'] = 'hist'
            xgb_params['objective'] = 'reg:squarederror'
            xgb_params['eval_metric'] = 'mae'
            xgb_params['random_state'] = 42

        model = xgb.XGBRegressor(**xgb_params)

        # Eğitim sırasında her adımı ekrana basıp bilgisayarı yormaması için verbose=False yapıldı
        fit_params = {'verbose': False}
        if 'early_stopping_rounds' not in xgb_params:
            model.set_params(early_stopping_rounds=100)

        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], **fit_params)

        results = model.evals_result()
        f_train_mae = results['validation_0']['mae'][-1]
        f_test_mae = results['validation_1']['mae'][-1]

        # Özellik Önemlerini Kaydetme
        if fold == 0:
            feature_names = X_train.columns.tolist()

        if hasattr(model, 'feature_importances_'):
            raw_importances = model.feature_importances_
            normalized_importances = (raw_importances / raw_importances.sum()) * 100
            fold_importances.append(normalized_importances)

        # Fold Metriklerini Kaydetme
        preds = model.predict(X_test)
        f_r2 = r2_score(y_test, preds)
        f_rmse = np.sqrt(mean_squared_error(y_test, preds))
        f_gap = ((f_test_mae - f_train_mae) / f_train_mae) * 100

        metrics['test_mae'].append(f_test_mae)
        metrics['train_mae'].append(f_train_mae)
        metrics['test_r2'].append(f_r2)
        metrics['test_rmse'].append(f_rmse)
        metrics['gap'].append(f_gap)

        print(
            f" Fold {fold + 1} | Test MAE: {f_test_mae:.2f} | Train MAE: {f_train_mae:.2f} | R2: {f_r2:.4f} | RMSE: {f_rmse:.2f} | Gap: %{f_gap:.2f}")

    # Raporlama Kısmı (Eski model_type çökmeleri giderildi)
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

    # FEATURE IMPORTANCE ÇİZDİRME
    if fold_importances:
        print("\nFeature Importance grafiği hesaplanıyor ve açılıyor...")
        avg_importance = np.mean(fold_importances, axis=0)
        std_importance = np.std(fold_importances, axis=0)

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance_Mean': avg_importance,
            'Importance_Std': std_importance
        }).sort_values(by='Importance_Mean', ascending=False)

        # Artık olmayan model_type yerine doğrudan "XGBOOST" stringi gönderiliyor
        plot_feature_importance(importance_df, "XGBOOST")

    plot_unified_correlation(X_train, y_train, target_name=target)

    return model


def optimize_xgboost(df, features_list, target='LEAD_TIME', n_splits=5, n_trials=50):
    """
    Optuna kullanarak XGBoost hiperparametrelerini K-Fold CV ile optimize eder.
    HIZLANDIRILMIŞ VERSİYON: K-Means önceden hesaplanır.
    """
    features = [
        'LOAD_ITEM', 'ORDER_MIKTAR', 'MAINPART', 'VENDORFINAL',
        'MATERIALTYPE_NEW', 'IS_CONDITION_CHANGED', 'GEOMETRIC_GROUP',
        'LENGTH', 'CALC_VOLUME'
    ]

    X = df[[c for c in features if c in df.columns]].copy()
    y = df[target]

    # =========================================================
    # OPTİMİZASYON 1: FOLD'LARI VE KÜMELERİ ÖNCEDEN HESAPLA
    # K-Means 250 kere değil, sadece 5 kere çalışacak!
    # =========================================================
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    precomputed_folds = []

    print("\nOptuna için veriler bölünüyor ve K-Means kümeleri önceden hesaplanıyor...")

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Veri sızıntısını önlemek için her fold'da K-Means çalışır (Sadece 1 kez)
        X_train, X_test = apply_multidim_kmeans_clustering(X_train, X_test, y_train)

        cat_cols = X_train.select_dtypes(include=['object']).columns
        for col in cat_cols:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

        # Hazırlanan bu seti belleğe (listeye) ekle
        precomputed_folds.append((X_train, X_test, y_train, y_test))

    sys.stdout = old_stdout  # Printleri geri aç
    print("Ön hesaplama tamamlandı. XGBoost Optimizasyonu başlıyor...")

    # =========================================================
    # OPTİMİZASYON 2: YALINLAŞTIRILMIŞ OBJECTIVE FONKSİYONU
    # =========================================================
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
            'random_state': 42
        }

        fold_maes = []

        # Sadece önceden hesaplanmış, temiz listeyi döngüye sok!
        for X_train, X_test, y_train, y_test in precomputed_folds:
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            preds = model.predict(X_test)
            fold_maes.append(mean_absolute_error(y_test, preds))

        return np.mean(fold_maes)

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    sys.stdout = old_stdout

    return study.best_params