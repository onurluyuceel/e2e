import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import pandas as pd

def cramers_v(x, y):
    """İki kategorik değişken arasındaki Cramer's V korelasyonunu hesaplar."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    # Bias Correction (Yanlılık Düzeltmesi)
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)

    denominator = min((k_corr - 1), (r_corr - 1))
    if denominator == 0:
        return 0
    return np.sqrt(phi2_corr / denominator)

def correlation_ratio(categories, measurements):
    """Kategorik ve Sayısal değişkenler arasındaki ilişkiyi (ETA) hesaplar."""

    # --- DÜZELTME: Pandas index uyuşmazlığını önlemek için saf numpy dizisine çeviriyoruz ---
    categories = np.array(categories)
    measurements = np.array(measurements)
    # ----------------------------------------------------------------------------------------

    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)

    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)

    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))

    if numerator == 0 or denominator == 0:
        return 0.0
    return np.sqrt(numerator / denominator)

def plot_unified_correlation(X, y, target_name='LEAD_TIME'):
    """Tüm seçilmiş özelliklerin (Kategorik + Sayısal) bütüncül ısı haritasını çizer."""
    df_plot = X.copy()
    df_plot[target_name] = y

    cols = df_plot.columns
    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    print("\n[ANALİZ] Bütüncül Korelasyon Matrisi hesaplanıyor (Spearman, Cramer's V, ETA)...")

    for i in range(len(cols)):
        for j in range(len(cols)):
            col1 = cols[i]
            col2 = cols[j]

            if i == j:
                corr_matrix.loc[col1, col2] = 1.0
                continue

            valid_idx = df_plot[[col1, col2]].dropna().index
            s1 = df_plot.loc[valid_idx, col1]
            s2 = df_plot.loc[valid_idx, col2]

            if len(s1) == 0:
                corr_matrix.loc[col1, col2] = 0.0
                continue

            is_num1 = pd.api.types.is_numeric_dtype(s1)
            is_num2 = pd.api.types.is_numeric_dtype(s2)

            # 1. Sayısal vs Sayısal (Spearman)
            if is_num1 and is_num2:
                corr = s1.corr(s2, method='spearman')
            # 2. Kategorik vs Kategorik (Cramer's V)
            elif not is_num1 and not is_num2:
                corr = cramers_v(s1, s2)
            # 3. Sayısal vs Kategorik (Correlation Ratio)
            else:
                corr = correlation_ratio(s2, s1) if is_num1 else correlation_ratio(s1, s2)

            corr_matrix.loc[col1, col2] = corr
    # =========================================================================
    # 2. YENİ EKLENEN: TERMİNALE YAZILI RAPOR BASMA KISMI
    # =========================================================================
    # Matrisi düzleştirip tabloya çeviriyoruz
    corr_pairs = corr_matrix.unstack().reset_index()
    corr_pairs.columns = ['Feature_1', 'Feature_2', 'Correlation']

    # Kendisiyle olanları (Örn: PLANT vs PLANT) ve tekrarları (A-B ve B-A) engelliyoruz
    corr_pairs = corr_pairs[corr_pairs['Feature_1'] < corr_pairs['Feature_2']].copy()
    corr_pairs['Abs_Corr'] = corr_pairs['Correlation'].abs()

    # Sadece 0.50'den büyük güçlü ilişkileri filtrele ve büyükten küçüğe sırala
    high_corr = corr_pairs[corr_pairs['Abs_Corr'] >= 0.50].sort_values(by='Abs_Corr', ascending=False)

    print("\n" + "=" * 70)
    print("[KORELASYON RAPORU] Birbiriyle En Çok Örtüşen Özellikler (> %50)")
    print("=" * 70)

    if high_corr.empty:
        print("  %50'nin üzerinde güçlü bir ilişki bulunamadı. Mükemmel!")
    else:
        print(f"  {'Özellik 1':<25} | {'Özellik 2':<25} | {'İlişki Gücü'}")
        print("  " + "-" * 66)
        for _, row in high_corr.iterrows():
            print(f"  {row['Feature_1']:<25} | {row['Feature_2']:<25} | {row['Correlation']:.3f}")
    print("=" * 70 + "\n")
    # 1. MASKE OLUŞTURMA: Mutlak değeri 0.50'den KÜÇÜK olanları "True" (yani gizle) yapıyoruz.
    # (abs kullanıyoruz çünkü -0.60 gibi güçlü ters ilişkilerin de görünmesini isteriz)
    mask = abs(corr_matrix) < 0.50

    plt.figure(figsize=(14, 12))

    # 2. MASKEYİ UYGULAMA: 'mask=mask' parametresini ekliyoruz
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='YlGnBu', center=0,
                vmin=0, vmax=1, square=True, linewidths=.5,
                mask=mask)  # <--- YENİ EKLENEN KISIM

    plt.title("Karma Özellikler Korelasyon Haritası (Sadece %50 Üzeri İlişkiler)", pad=20, size=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
