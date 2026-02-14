import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def train_xgboost_model(df, target='LEAD_TIME'):
    # 1. Özellik Seçimi
    features = [
        'LOAD_ITEM', 'ORDER_MIKTAR', 'MAINPART', 'VENDORFINAL',
        'MATERIALTYPE_NEW', 'IS_CONDITION_CHANGED', 'GEOMETRIC_GROUP',
        'LENGTH', 'CALC_VOLUME'
    ]

    X = df[features].copy()
    y = df[target]

    # 2. Kategorik Dönüşüm (XGBoost için Şart)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')

    # 3. Veri Bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print("\n[ÖZELLİK ÖNEMİ]")
    print(importance_df)

    return model