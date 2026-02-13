import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import pandas as pd

def numeric_feature_correlation(df):

    # İstemeyen statüler
    exclude_cols = ['SDR', 'BLRSZ', 'WOR', 'SCH', 'TAR', 'DET', 'DOC', 'FMP',
                    'TKM', 'HLD', 'CLS', 'ARL', 'MIK', 'OKS', 'INT', 'NPO',
                    'OSI', 'PKM', 'ITP', 'ADR', 'KPY']

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0, square=True)
    plt.title("Sayısal Özelliklerin Korelasyonu")
    plt.tight_layout()
    plt.show()

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


def categorical_feature_correlation(df):
    """Kategorik özellikler için Cramer's V heatmap oluşturur."""
    # Analiz edilecek kategorik sütunları belirle
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not cat_cols:
        print("Analiz edilecek kategorik sütun bulunamadı.")
        return

    n = len(cat_cols)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), columns=cat_cols, index=cat_cols)

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
            else:
                corr_matrix.iloc[i, j] = cramers_v(df[cat_cols[i]], df[cat_cols[j]])

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='YlGnBu', square=True)
    plt.title("Kategorik Özelliklerin Korelasyonu (Cramer's V)")
    plt.tight_layout()
    plt.show()

def run_all_analyses(df):

    numeric_feature_correlation(df)
    categorical_feature_correlation(df)
