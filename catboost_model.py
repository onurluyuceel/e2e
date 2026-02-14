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
        eval_metric='R2',
        early_stopping_rounds=100,
        verbose=500,
        allow_writing_files=False
    )

    # 4. Eğitim
    model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_test, y_test))

    # 5. Metrikler
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    print(f"\n[PERFORMANS]")
    print(f"R2 Score : {r2:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"MAE      : {mae:.4f}")

    # 6. Özellik Önemi
    print("\n[ÖZELLİK ÖNEMİ]")
    print(model.get_feature_importance(prettified=True))

    return model