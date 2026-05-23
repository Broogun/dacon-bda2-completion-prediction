"""
Ensemble Training - CatBoost + LightGBM

실행:
    python run_ensemble.py

출력:
    - models/catboost_model.pkl
    - models/lightgbm_model.pkl
    - outputs/oof/catboost_oof.npy
    - outputs/oof/lightgbm_oof.npy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import yaml

from src.config import *
from src.features import create_features, get_categorical_features, get_numerical_features

print("="*60)
print("Ensemble Training - CatBoost + LightGBM")
print("="*60)

# Load configs
with open(CONFIG_DIR / "catboost.yaml") as f:
    cat_config = yaml.safe_load(f)

with open(CONFIG_DIR / "lightgbm.yaml") as f:
    lgb_config = yaml.safe_load(f)

# Load data
print("\n[1/5] Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

y = train[TARGET_COL].values

# Feature engineering
print("\n[2/5] Feature engineering...")
train_fe = create_features(train)
test_fe = create_features(test)

X_train = train_fe.drop(columns=[ID_COL, TARGET_COL], errors='ignore')
X_test = test_fe.drop(columns=[ID_COL], errors='ignore')

cat_features = get_categorical_features(X_train)
num_features = get_numerical_features(X_train)

print(f"Features: {X_train.shape[1]} (cat: {len(cat_features)}, num: {len(num_features)})")

# Train ensemble
print("\n[3/5] Training ensemble...")

oof_cat_all = np.zeros(len(X_train))
oof_lgbm_all = np.zeros(len(X_train))
test_cat_all = np.zeros(len(X_test))
test_lgbm_all = np.zeros(len(X_test))

for seed_idx, seed in enumerate(RANDOM_SEEDS):
    print(f"\n[Seed {seed_idx+1}/{len(RANDOM_SEEDS)}] Seed: {seed}")
    
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_train, y)):
        print(f"  Fold {fold_idx+1}/{N_FOLDS}...", end=" ")
        
        # CatBoost
        cat_model = CatBoostClassifier(
            **cat_config['model_params'],
            random_seed=seed,
            cat_features=cat_features
        )
        cat_model.fit(
            X_train.iloc[tr_idx],
            y[tr_idx],
            eval_set=(X_train.iloc[va_idx], y[va_idx]),
            use_best_model=True
        )
        oof_cat_all[va_idx] = cat_model.predict_proba(X_train.iloc[va_idx])[:, 1]
        
        # LightGBM (numerical only)
        X_num = X_train[num_features]
        dtrain = lgb.Dataset(X_num.iloc[tr_idx], y[tr_idx])
        dval = lgb.Dataset(X_num.iloc[va_idx], y[va_idx])
        
        lgbm_model = lgb.train(
            lgb_config['model_params'],
            dtrain,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        oof_lgbm_all[va_idx] = lgbm_model.predict(X_num.iloc[va_idx])
        
        print("✓")

# Average OOF
oof_cat = oof_cat_all
oof_lgbm = oof_lgbm_all

print(f"\nOOF CatBoost range: [{oof_cat.min():.3f}, {oof_cat.max():.3f}]")
print(f"OOF LightGBM range: [{oof_lgbm.min():.3f}, {oof_lgbm.max():.3f}]")

# Save OOF
print("\n[4/5] Saving OOF predictions...")
np.save(CATBOOST_OOF, oof_cat)
np.save(LIGHTGBM_OOF, oof_lgbm)
print(f"✅ OOF saved")

# Train final models on full data
print("\n[5/5] Training final models on full data...")

# CatBoost
final_cat = CatBoostClassifier(
    **cat_config['model_params'],
    random_seed=RANDOM_SEEDS[0],
    cat_features=cat_features
)
final_cat.fit(X_train, y, verbose=False)
test_cat = final_cat.predict_proba(X_test)[:, 1]

# LightGBM
X_num_full = X_train[num_features]
dtrain_full = lgb.Dataset(X_num_full, y)
final_lgbm = lgb.train(
    lgb_config['model_params'],
    dtrain_full,
    num_boost_round=1000
)
test_lgbm = final_lgbm.predict(X_test[num_features])

# Save models
joblib.dump(final_cat, CATBOOST_MODEL)
joblib.dump(final_lgbm, LIGHTGBM_MODEL)
print(f"✅ Models saved")

# Save test predictions
np.save(OUTPUT_DIR / "catboost_test.npy", test_cat)
np.save(OUTPUT_DIR / "lightgbm_test.npy", test_lgbm)
print(f"✅ Test predictions saved")

print("\n" + "="*60)
print("✅ Ensemble Training Complete!")
print("="*60)
print(f"\nNext step: python run_stacking.py")
