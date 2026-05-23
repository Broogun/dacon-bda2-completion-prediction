# BDA 수료 예측 - 빠른 시작 가이드

## 설치

```bash
# 가상환경 생성 및 패키지 설치
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Mac/Linux

pip install -r requirements.txt
```

---

## 프로젝트 구조

```
BDA-Completion-Prediction/
├── README.md               # 전체 설명
├── PROBLEM.md              # 문제 정의
├── EXPERIMENTS.md          # 실험 기록
│
├── configs/
│   ├── catboost.yaml
│   ├── lightgbm.yaml
│   ├── decision_tree.yaml
│   └── meta.yaml
│
├── src/
│   ├── config.py
│   └── features.py
│
├── notebooks/
│   ├── 00_problem_definition.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_train_test_drift.ipynb
│   ├── 03_lift_analysis.ipynb
│   └── 04_final_model.ipynb  # 최종 모델 구현 ← 분석 흐름 확인 시 여기서 시작
│
├── run_rule_discovery.py   # 1단계
├── run_ensemble.py         # 2단계
├── run_stacking.py         # 3단계
│
├── data/                   # train.csv, test.csv 위치
├── models/                 # 저장된 모델
└── outputs/
    ├── submission.csv
    └── oof/

# 원본 실험 코드 (아카이브)
├── test1.ipynb
└── test2.ipynb
```

---

## 실행 방법

### 데이터 준비
```
data/ 폴더에 train.csv, test.csv 복사
```

### 학습 및 예측 (순서대로)

```bash
# 1단계: Rule Discovery
python run_rule_discovery.py
# → models/rule_model.pkl, outputs/oof/rule_oof.npy

# 2단계: Ensemble (CatBoost + LightGBM, 5-Seed x 5-Fold)
python run_ensemble.py
# → models/catboost_model.pkl, models/lightgbm_model.pkl
# → outputs/oof/catboost_oof.npy, outputs/oof/lightgbm_oof.npy

# 3단계: Stacking + 최종 예측
python run_stacking.py
# → outputs/submission.csv
```

예상 소요 시간: 약 11분 (Rule 1분 + Ensemble 10분 + Stacking 10초)

---

## 설정 변경

**configs/meta.yaml** — 선발 인원:
```yaml
flip_rate:
  target_count: 375   # 변경 가능 (실험 결과 375~376이 최적)
```

**configs/catboost.yaml** — 모델 파라미터:
```yaml
model_params:
  iterations: 2000
  learning_rate: 0.03
  depth: 6
```

---

## 문제 해결

```bash
# OOF 파일 없음 오류 → 순서대로 실행
python run_rule_discovery.py
python run_ensemble.py
python run_stacking.py

# ModuleNotFoundError → 가상환경 활성화 확인 후 재설치
pip install -r requirements.txt
```

---

## 예상 결과

```
Rule Score:    [0.2, 0.8]
CatBoost OOF:  [0.15, 0.85]
LightGBM OOF:  [0.20, 0.80]

Selected: 375 / 814명
예상 Public F1: ~0.44
```

---

참고: README.md, PROBLEM.md, EXPERIMENTS.md
