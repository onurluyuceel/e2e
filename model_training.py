import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from analysis import plot_unified_correlation

def apply_multidim_kmeans_clustering(X_train, X_test, y_train, n_clusters=5):
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

def run_cross_validation(df, model_type='xgboost', target='LEAD_TIME', n_splits=5):
    """
    Belirlenen özellikler üzerinden K-Fold Çapraz Doğrulama (Cross-Validation) yapar.
    """

    # 1. ÖZELLİK SEÇİMİ (FEATURE SELECTION)
    # Modelin öğrenmesini istediğimiz "en etkili" 9 değişkeni buraya tanımlıyoruz.
    # NOT: VENDORFINAL'ı buraya ekledik çünkü K-Means fonksiyonu bu ismi kullanarak VENDOR_GROUP türetecek.
    features = [
        'VENDORFINAL', 'MATERIALTYPE_NEW', 'MAINPART GRUP',
        'LENGTH', 'LOAD_ITEM',
        'Yönetici Etkisi', 'ORDER_MIKTAR', 'YUZEY_ISLEM'
    ]

    # Sadece seçilen özellikleri ve hedef değişkeni (Lead Time) veriden ayırıyoruz.
    # 'if c in df.columns' kontrolü, olmayan bir sütun çağrılırsa kodun çökmesini önler.
    X = df[[c for c in features if c in df.columns]].copy()
    y = df[target]

    # 2. K-FOLD YAPISININ KURULMASI
    # n_splits=5: Veriyi 5 parçaya böler. Her seferinde 4 parça eğitim, 1 parça test olur.
    # shuffle=True: Veriyi bölmeden önce karıştırır (tarihsel bir sıralama varsa yanlılığı önler).
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Tüm metrikleri saklamak için genişletilmiş sözlük
    metrics = {
        'test_mae': [], 'train_mae': [],
        'test_r2': [], 'test_rmse': [], 'gap': []
    }
    # Feature Importance Değişkenleri
    fold_importances = []
    feature_names = None

    # 3. CROSS-VALIDATION DÖNGÜSÜ
    # Her bir 'fold' (katman) için veriyi %80 Eğitim - %20 Test olarak ayırıp işlemleri başlatıyoruz.
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Veriyi o anki fold indexlerine göre ayırıyoruz
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # --- KRİTİK ADIM: K-MEANS GRUPLAMA ---
        # Veri sızıntısını (Data Leakage) önlemek için K-Means'i her fold içinde sıfırdan hesaplıyoruz.
        # Bu işlem sonucunda VENDORFINAL sütunu silinir, yerine VENDOR_GROUP eklenir.
        X_train, X_test = apply_multidim_kmeans_clustering(X_train, X_test, y_train)

        # 4. MODEL SEÇİMİ VE EĞİTİMİ
        if model_type == 'xgboost':
            # XGBoost kategorik verileri 'category' tipinde bekler.
            # 'object' (metin) tipindeki sütunları bulup dönüştürüyoruz.
            cat_cols = X_train.select_dtypes(include=['object']).columns
            for col in cat_cols:
                # --- KRİTİK DÜZELTME BAŞLANGICI ---
                # 1. Train setindeki kategorileri belirle
                X_train[col] = X_train[col].astype('category')

                # 2. Test setindeki değerleri Train'in kategorilerine zorla.
                # Train'de bulunmayan değerler (örn: Cluster_New) otomatik olarak NaN olur.
                # XGBoost NaN kategorik değerleri "bilinmeyen" olarak işleyebilir.
                X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

            # --- SENİN PARAMETRELERİNİN ENTEGRASYONU ---
            model = xgb.XGBRegressor(
                n_estimators=3000,  # Daha fazla ağaç
                learning_rate=0.1,  # Daha yavaş ve dikkatli öğrenme
                max_depth=7,  # Sığ ağaçlar (Overfitting önler)
                reg_lambda=15,  # L2 Regularization
                reg_alpha=8,  # L1 Regularization
                min_child_weight=6,  # Dallar arası minimum ağırlık
                gamma=2,  # Dallanma için gereken minimum azalma
                subsample=0.85,  # Verinin %60'ını rastgele seç
                colsample_bytree=0.8,  # Özelliklerin %60'ını rastgele seç
                objective='reg:squarederror',
                eval_metric='mae',
                early_stopping_rounds=100,  # Gelişme durursa eğitimi kes
                enable_categorical=True,  # Kategorik desteği
                tree_method='hist',  # Hızlı eğitim metodu
                random_state=42
                )
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=500)

            results = model.evals_result()
            f_train_mae = results['validation_0']['mae'][-1]
            f_test_mae = results['validation_1']['mae'][-1]

        elif model_type == 'catboost':
            # CatBoost kategorik sütunların listesini açıkça bilmek ister.
            cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            # CatBoost Regressor Ayarları
            # allow_writing_files=False: Eğitim sırasında log dosyası oluşturmasını engeller (hız için).
            model = CatBoostRegressor(
                iterations=3000,
                learning_rate=0.03,
                depth=4,
                l2_leaf_reg=10,
                eval_metric='MAE',
                early_stopping_rounds=100,
                verbose=500,
                allow_writing_files=False
            )
            model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_test, y_test))

            # Metrik tutarlılığı için doğrudan MAE hesaplıyoruz
            f_train_mae = mean_absolute_error(y_train, model.predict(X_train))
            f_test_mae = mean_absolute_error(y_test, model.predict(X_test))

        # Özellik Önemlerini Kaydetme
        if fold == 0:
            feature_names = X_train.columns.tolist()

        if hasattr(model, 'feature_importances_'):
            raw_importances = model.feature_importances_
            # Modellerin farklı formatlarını eşitlemek için toplamı 100'e (Yüzdeye) zorluyoruz
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

        # Fold içindeki metrikleri yazdıran güncellenmiş satır:
        print(f" Fold {fold + 1} | Test MAE: {f_test_mae:.2f} | Train MAE: {f_train_mae:.2f} | R2: {f_r2:.4f} | RMSE: {f_rmse:.2f} | Gap: %{f_gap:.2f}")

    # Raporlama Kısmı
    print("\n" + "=" * 55)
    print(f"[{model_type.upper()} CV NİHAİ RAPOR]")
    print("=" * 55)
    print(f"  {'R2 Score (Avg ± Std)':<25} : {np.mean(metrics['test_r2']):.4f} ± {np.std(metrics['test_r2']):.4f}")
    print(f"  {'RMSE (Avg ± Std)':<25} : {np.mean(metrics['test_rmse']):.4f} ± {np.std(metrics['test_rmse']):.4f}")
    print(
        f"  {'Test MAE (Avg ± Std)':<25} : {np.mean(metrics['test_mae']):.4f} ± {np.std(metrics['test_mae']):.4f}")
    print(
        f"  {'Train MAE (Avg ± Std)':<25} : {np.mean(metrics['train_mae']):.4f} ± {np.std(metrics['train_mae']):.4f}")
    print("=" * 55)
    print(f"  {'MAE FARKI (GAP) (%)':<25} : %{np.mean(metrics['gap']):.2f} ± %{np.std(metrics['gap']):.2f}")
    print("=" * 55)

    # FEATURE IMPORTANCE HESAPLAMA VE ÇİZDİRME
    if fold_importances:
        print("\nFeature Importance grafiği hesaplanıyor ve açılıyor...")

        avg_importance = np.mean(fold_importances, axis=0)
        std_importance = np.std(fold_importances, axis=0)

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance_Mean': avg_importance,
            'Importance_Std': std_importance
        }).sort_values(by='Importance_Mean', ascending=False)

    # Çizim fonksiyonunu çağırıyoruz
    plot_feature_importance(importance_df, model_type)
    # ====================================================
    # BÜTÜNCÜL KORELASYON HARİTASINI ÇAĞIR
    # ====================================================
    # K-Means ile üretilen VENDOR_GROUP'u görebilmek için
    # son fold'daki eğitim setini (X_train, y_train) gönderiyoruz.
    plot_unified_correlation(X_train, y_train, target_name=target)
    # ====================================================

    return model