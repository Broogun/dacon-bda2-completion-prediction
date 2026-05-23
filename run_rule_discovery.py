"""
Rule Discovery - Decision Tree 기반 자동 Rule 생성

실행:
    python run_rule_discovery.py

출력:
    - models/rule_model.pkl
    - outputs/oof/rule_oof.npy
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
import joblib
import yaml

from src.config import *
from src.features import create_features

print("="*60)
print("Rule Discovery - Decision Tree")
print("="*60)

# Load config
with open(CONFIG_DIR / "decision_tree.yaml") as f:
    config = yaml.safe_load(f)

# Load data
print("\n[1/5] Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
print(f"Train: {train.shape}, Test: {test.shape}")

# Target
y = train[TARGET_COL].values

# Feature engineering
print("\n[2/5] Creating rule features...")

def create_rule_features(df):
    """Rule용 Feature 생성"""
    features = pd.DataFrame(index=df.index)
    
    # Persona combinations
    features['natural_science_jobseeker'] = (
        (df['major_field'] == '자연과학') & (df['job'] == '취준생')
    ).astype(int)
    
    features['humanities_student'] = (
        (df['major_field'] == '인문학') & (df['job'] == '대학생')
    ).astype(int)
    
    # Re-registration
    features['re_registration'] = (df['re_registration'] == '예').astype(int)
    
    # Time investment
    features['time_high'] = (df['time_input'] >= 3.0).astype(int)
    features['time_4h'] = (df['time_input'] == 4.0).astype(int)
    
    # Keywords
    text = df['whyBDA'].fillna('') + ' ' + df['incumbents_lecture_scale_reason'].fillna('')
    features['keyword_opportunity'] = text.str.contains('기회', na=False).astype(int)
    features['keyword_together'] = text.str.contains('함께', na=False).astype(int)
    
    # Interaction
    features['re_reg_time_high'] = (features['re_registration'] & features['time_high']).astype(int)
    
    return features

train_features = create_rule_features(train)
test_features = create_rule_features(test)

print(f"Rule features: {train_features.shape[1]}")
print(f"Sample features: {list(train_features.columns[:5])}")

# Train Decision Tree
print("\n[3/5] Training Decision Tree...")
dt = DecisionTreeClassifier(**config['model_params'])
dt.fit(train_features, y)

# Export rules
rules = export_text(dt, feature_names=train_features.columns.tolist())
print("\nDiscovered Rules:")
print(rules)

# Feature importance
importance = pd.DataFrame({
    'feature': train_features.columns,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Important Features:")
print(importance.head().to_string(index=False))

# Get rule scores (probabilities)
print("\n[4/5] Computing rule scores...")
train_rule_score = dt.predict_proba(train_features)[:, 1]
test_rule_score = dt.predict_proba(test_features)[:, 1]

print(f"Train rule score range: [{train_rule_score.min():.3f}, {train_rule_score.max():.3f}]")
print(f"Test rule score range: [{test_rule_score.min():.3f}, {test_rule_score.max():.3f}]")

# Save
print("\n[5/5] Saving...")

# Save model
joblib.dump({
    'model': dt,
    'feature_names': train_features.columns.tolist(),
    'config': config
}, RULE_MODEL)
print(f"✅ Model saved: {RULE_MODEL}")

# Save OOF
np.save(RULE_OOF, train_rule_score)
print(f"✅ OOF saved: {RULE_OOF}")

# Save test predictions
np.save(OUTPUT_DIR / "rule_test.npy", test_rule_score)
print(f"✅ Test predictions saved")

print("\n" + "="*60)
print("✅ Rule Discovery Complete!")
print("="*60)
print(f"\nNext step: python run_ensemble.py")
