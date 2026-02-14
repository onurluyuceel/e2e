import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_catboost_model(df, target='LEAD_TIME'):
    # 1. Seçili Özellikler
    features = [
        'LOAD_ITEM', 'ORDER_MIKTAR', 'MAINPART', 'VENDORFINAL',
        'MATERIALTYPE_NEW', 'IS_CONDITION_CHANGED', 'GEOMETRIC_GROUP',
        'LENGTH', 'CALC_VOLUME'
    ]

    X = df[features]
    y = df[target]

    # Kategorik sütunları otomatik belirle
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 2. Veri Bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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