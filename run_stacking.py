"""
Stacking Meta-model + Final Prediction

실행:
    python run_stacking.py

출력:
    - models/meta_model.pkl
    - outputs/submission.csv
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import yaml

from src.config import *

print("="*60)
print("Stacking Meta-model + Final Prediction")
print("="*60)

# Load config
with open(CONFIG_DIR / "meta.yaml") as f:
    config = yaml.safe_load(f)

# Load OOF predictions
print("\n[1/5] Loading OOF predictions...")
oof_cat = np.load(CATBOOST_OOF)
oof_lgbm = np.load(LIGHTGBM_OOF)
oof_rule = np.load(RULE_OOF)

print(f"CatBoost OOF: {oof_cat.shape}")
print(f"LightGBM OOF: {oof_lgbm.shape}")
print(f"Rule OOF: {oof_rule.shape}")

# Load test predictions
print("\n[2/5] Loading test predictions...")
test_cat = np.load(OUTPUT_DIR / "catboost_test.npy")
test_lgbm = np.load(OUTPUT_DIR / "lightgbm_test.npy")
test_rule = np.load(OUTPUT_DIR / "rule_test.npy")

# Load train data for target
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
y = train[TARGET_COL].values

# Create meta features
print("\n[3/5] Training meta-model...")

meta_X = pd.DataFrame({
    'CAT': oof_cat,
    'LGBM': oof_lgbm,
    'RULE': oof_rule
})

meta_test = pd.DataFrame({
    'CAT': test_cat,
    'LGBM': test_lgbm,
    'RULE': test_rule
})

# Train meta-model
meta_model = LogisticRegression(**config['model_params'])
meta_model.fit(meta_X, y)

print("\nMeta-model Coefficients:")
print(f"  CAT:  {meta_model.coef_[0][0]:.4f}")
print(f"  LGBM: {meta_model.coef_[0][1]:.4f}")
print(f"  RULE: {meta_model.coef_[0][2]:.4f}")

# Predict
probabilities = meta_model.predict_proba(meta_test)[:, 1]
print(f"\nProbability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")

# Hard kill (optional)
if config['hard_kill']['enabled']:
    print("\n[4/5] Applying hard kill rules...")
    
    # Law major
    kill_mask = (test['major_field'] == '법학')
    if 'major1_1' in test.columns:
        kill_mask |= test['major1_1'].str.contains('법', na=False)
    
    # Business worker
    kill_mask |= (
        (test['major_field'] == '경영학') &
        (test['job'].isin(['직장인', '취준생']))
    )
    
    probabilities[kill_mask] = 0.0
    print(f"Killed: {kill_mask.sum()} students")

# Flip-rate selection
print("\n[5/5] Flip-rate selection...")
k = config['flip_rate']['target_count']

threshold = np.sort(probabilities)[::-1][k - 1]
predictions = (probabilities >= threshold).astype(int)

print(f"Target: {k} students")
print(f"Selected: {predictions.sum()} students")
print(f"Selection rate: {predictions.mean():.2%}")

# Save
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET_COL: predictions
})
submission.to_csv(SUBMISSION_PATH, index=False)

# Save meta-model
joblib.dump(meta_model, META_MODEL)

print(f"\nSubmission saved: {SUBMISSION_PATH}")
print(f"Meta-model saved: {META_MODEL}")

print("\n" + "="*60)
print("Stacking Complete!")
print("="*60)
print(f"\nSubmission file ready: {SUBMISSION_PATH}")
print(f"Selected: {predictions.sum()}/{len(predictions)} students")
