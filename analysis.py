import seaborn as sns
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

def run_all_analyses(df):

    numeric_feature_correlation(df)
