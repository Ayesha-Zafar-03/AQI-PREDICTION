import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from dotenv import load_dotenv

load_dotenv()

def load_features_from_local(path="data/processed/features.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"The file '{path}' was loaded successfully but contains no data.")
    return df

def evaluate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

class StackedAQIModel:
    """
    Wrapper class for stacked ensemble model
    Makes it compatible with app.py's single model.predict() call
    """
    def __init__(self, rf_model, lr_model, meta_model):
        self.rf_model = rf_model
        self.lr_model = lr_model
        self.meta_model = meta_model
    
    def predict(self, X):
        """Predict using stacked ensemble"""
        rf_pred = self.rf_model.predict(X).reshape(-1, 1)
        lr_pred = self.lr_model.predict(X).reshape(-1, 1)
        X_meta = np.hstack((rf_pred, lr_pred))
        return self.meta_model.predict(X_meta)

def train_model(df, model_dir=None, use_simple_features=False):
    """
    Train AQI prediction model with option for simple or full features
    
    Args:
        df: DataFrame with features
        model_dir: Directory to save models
        use_simple_features: If True, use only basic features (compatible with app.py)
    """
    
    if use_simple_features:
        # Simple features that app.py can provide
        feature_cols = [
            'pm2_5_lag1', 'pm2_5_lag2', 'pm2_5_lag3',
            'pm2_5_roll7', 'pm2_5_roll14',
            'dayofweek', 'day', 'month'
        ]
        print("\n🔹 Using SIMPLE feature set (8 features) - compatible with app.py")
    else:
        # Full feature set
        feature_cols = [
            'pm2_5', 'pm10', 'no2', 'o3', 'so2', 'co', 'aqi',
            'dayofweek', 'day', 'month',
            'pm2_5_lag1', 'pm2_5_lag2', 'pm2_5_lag3',
            'pm2_5_roll7', 'pm2_5_roll14'
        ]
        print("\n🔹 Using FULL feature set (15 features)")
    
    target_col = 'pm2_5_next_1'

    # Check which features are available
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        print(f"✅ Available features: {available_features}")
        feature_cols = available_features
    
    df_clean = df.dropna(subset=feature_cols + [target_col])

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    print(f"\n📊 Dataset info:")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {len(X)}")
    print(f"   NaNs in features: {np.isnan(X).sum()}")
    print(f"   NaNs in target: {np.isnan(y).sum()}")

    # Split once for final testing and stacking
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n🔄 Performing 5-Fold Cross-Validation on base models...\n")

    rf_cv_metrics = []
    lr_cv_metrics = []

    for fold, (train_index, test_index) in enumerate(kf.split(X_train_full)):
        X_train, X_test = X_train_full[train_index], X_train_full[test_index]
        y_train, y_test = y_train_full[train_index], y_train_full[test_index]

        # Train base models with fixed random_state
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,  # ✅ Fixed seed for reproducibility
            n_jobs=-1
        )
        lr_model = LinearRegression()

        rf_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)

        # Predict
        rf_pred = rf_model.predict(X_test)
        lr_pred = lr_model.predict(X_test)

        # Evaluate
        rf_metrics = evaluate_metrics(y_test, rf_pred)
        lr_metrics = evaluate_metrics(y_test, lr_pred)

        rf_cv_metrics.append(rf_metrics)
        lr_cv_metrics.append(lr_metrics)

        print(f"Fold {fold + 1}:")
        print(f"  RF - MSE: {rf_metrics[0]:.4f}, RMSE: {rf_metrics[1]:.4f}, MAE: {rf_metrics[2]:.4f}, R²: {rf_metrics[3]:.4f}")
        print(f"  LR - MSE: {lr_metrics[0]:.4f}, RMSE: {lr_metrics[1]:.4f}, MAE: {lr_metrics[2]:.4f}, R²: {lr_metrics[3]:.4f}")
        print("")

    # Average CV results
    def avg_metrics(metrics_list):
        return tuple(np.mean(metrics_list, axis=0))

    rf_avg = avg_metrics(rf_cv_metrics)
    lr_avg = avg_metrics(lr_cv_metrics)

    print("📈 Average CV Metrics:")
    print(f"Random Forest - MSE: {rf_avg[0]:.4f}, RMSE: {rf_avg[1]:.4f}, MAE: {rf_avg[2]:.4f}, R²: {rf_avg[3]:.4f}")
    print(f"Linear Reg.   - MSE: {lr_avg[0]:.4f}, RMSE: {lr_avg[1]:.4f}, MAE: {lr_avg[2]:.4f}, R²: {lr_avg[3]:.4f}")

    print("\n🎯 Training base models on full training data...")

    # Train base models on full training set with FIXED random_state
    rf_model_final = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,  # ✅ Fixed seed
        n_jobs=-1
    )
    lr_model_final = LinearRegression()

    rf_model_final.fit(X_train_full, y_train_full)
    lr_model_final.fit(X_train_full, y_train_full)

    # Base predictions for meta-model training
    rf_train_pred = rf_model_final.predict(X_train_full).reshape(-1, 1)
    lr_train_pred = lr_model_final.predict(X_train_full).reshape(-1, 1)
    X_meta_train = np.hstack((rf_train_pred, lr_train_pred))

    # Meta-model training
    meta_model = LinearRegression()
    meta_model.fit(X_meta_train, y_train_full)

    # Prepare meta-model validation features
    rf_val_pred = rf_model_final.predict(X_val).reshape(-1, 1)
    lr_val_pred = lr_model_final.predict(X_val).reshape(-1, 1)
    X_meta_val = np.hstack((rf_val_pred, lr_val_pred))

    # Final predictions with meta-model
    final_pred = meta_model.predict(X_meta_val)

    # Evaluate all on validation set
    print("\n📊 Final Evaluation on Validation Set:")
    rf_val_metrics = evaluate_metrics(y_val, rf_val_pred.flatten())
    lr_val_metrics = evaluate_metrics(y_val, lr_val_pred.flatten())
    meta_val_metrics = evaluate_metrics(y_val, final_pred)

    print(f"Random Forest - MSE: {rf_val_metrics[0]:.4f}, RMSE: {rf_val_metrics[1]:.4f}, MAE: {rf_val_metrics[2]:.4f}, R²: {rf_val_metrics[3]:.4f}")
    print(f"Linear Reg.   - MSE: {lr_val_metrics[0]:.4f}, RMSE: {lr_val_metrics[1]:.4f}, MAE: {lr_val_metrics[2]:.4f}, R²: {lr_val_metrics[3]:.4f}")
    print(f"Stacked Model - MSE: {meta_val_metrics[0]:.4f}, RMSE: {meta_val_metrics[1]:.4f}, MAE: {meta_val_metrics[2]:.4f}, R²: {meta_val_metrics[3]:.4f}")

    # Create stacked model wrapper
    stacked_model = StackedAQIModel(rf_model_final, lr_model_final, meta_model)

    # Save models
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        
        # Save individual models (for inspection)
        joblib.dump(rf_model_final, os.path.join(model_dir, "random_forest.pkl"))
        joblib.dump(lr_model_final, os.path.join(model_dir, "linear_regression.pkl"))
        joblib.dump(meta_model, os.path.join(model_dir, "stacked_model.pkl"))
        
        # ✅ IMPORTANT: Save the wrapper as rf_model.pkl (what app.py expects)
        joblib.dump(stacked_model, os.path.join(model_dir, "rf_model.pkl"))
        
        # Save feature names for reference
        feature_info = {
            'features': feature_cols,
            'n_features': len(feature_cols),
            'simple_features': use_simple_features
        }
        joblib.dump(feature_info, os.path.join(model_dir, "feature_info.pkl"))
        
        print(f"\n✅ Saved models to directory: {model_dir}")
        print(f"   - random_forest.pkl (individual RF model)")
        print(f"   - linear_regression.pkl (individual LR model)")
        print(f"   - stacked_model.pkl (meta model)")
        print(f"   - rf_model.pkl (✅ MAIN MODEL for app.py)")
        print(f"   - feature_info.pkl (feature metadata)")

    return stacked_model, {
        "train_samples": len(X_train_full),
        "val_samples": len(X_val),
        "features": feature_cols,
        "rf_val_r2": rf_val_metrics[3],
        "lr_val_r2": lr_val_metrics[3],
        "stacked_val_r2": meta_val_metrics[3]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/features.csv", help="Path to features CSV")
    parser.add_argument("--model_dir", default="models/aqi_model", help="Directory to save trained models")
    parser.add_argument("--simple", action="store_true", help="Use simple feature set (compatible with app.py)")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 AQI Model Training Pipeline")
    print("=" * 60)

    df = load_features_from_local(args.input)
    model, metrics = train_model(df, model_dir=args.model_dir, use_simple_features=args.simple)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    print(f"📊 Final Metrics:")
    print(f"   Train samples: {metrics['train_samples']}")
    print(f"   Val samples: {metrics['val_samples']}")
    print(f"   RF R²: {metrics['rf_val_r2']:.4f}")
    print(f"   LR R²: {metrics['lr_val_r2']:.4f}")
    print(f"   Stacked R²: {metrics['stacked_val_r2']:.4f}")
    print(f"\n🎯 Model ready for deployment!")
    print(f"   Location: {args.model_dir}/rf_model.pkl")
    print("=" * 60)
