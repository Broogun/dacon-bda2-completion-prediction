"""
Feature Engineering
"""

import pandas as pd
import numpy as np


def get_refined_score(row) -> int:
    """
    도메인 지식 기반 점수 시스템.

    03_lift_analysis에서 발견한 패턴을 Rule로 변환.
    - 안정 변수(L1<0.2) 우선 사용
    - 부정 키워드는 긍정 맥락 존재 시 감점 없음 (중복 감점 방지)

    Returns:
        int: Rule score (음수~양수)
    """
    t1 = str(row['whyBDA']) if not pd.isna(row['whyBDA']) else ""
    t2 = str(row['incumbents_lecture_scale_reason']) if not pd.isna(row['incumbents_lecture_scale_reason']) else ""
    full_text = t1 + " " + t2
    score = 0

    major = str(row['major_field'])
    job   = str(row['job'])
    desired_job = str(row['desired_job'])

    # 페르소나 (Lift>1.3이나 Support 부족 → 도메인 판단)
    if "자연과학" in major and "취준생" in job:
        score += 3
    if "인문학" in major and "대학생" in job:
        score += 2
    if "데이터 분석가" in desired_job and "온라인" in str(row['incumbents_lecture_type']):
        score += 2

    # 긍정 키워드 (합산 텍스트 기준)
    if "함께" in full_text:
        score += 1
    if "기회" in full_text:
        score += 3

    # 부정 키워드 — 맥락 고려하여 중복 감점 방지
    if "경험" in full_text:
        score -= 2
    positive_context = ["기회", "함께", "도움", "필요", "배우", "성장", "발전"]
    has_positive = any(word in full_text for word in positive_context)
    if ("어려워" in full_text or "혼자" in full_text) and not has_positive:
        score -= 2

    # 방어 규칙
    inflow = str(row['inflow_route'])
    if "인스타그램" in inflow and (len(t1) + len(t2)) < 30:
        score -= 3

    # 안정 신호 (L1<0.2, 4조건 통과)
    if row['time_input'] == 4.0:
        score += 2
    if row['re_registration'] == '예':
        score += 2

    # school1 보정 (L1=0.718 — 불안정, 실험적 적용)
    school = str(row['school1'])
    if school in ['45', '38', '52']:
        score += 2
    if school in ['81', '54', '19']:
        score -= 5

    return score


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering.

    최종 모델에서 사용하는 피처 생성.
    """
    df = df.copy()

    # Rule-based score (04_final_model 핵심 피처)
    df['refined_score'] = df.apply(get_refined_score, axis=1)

    # 텍스트 응답 길이 합산 (tie-breaking 및 방어 규칙용)
    df['len_total'] = (
        df['whyBDA'].fillna('').apply(len)
        + df['incumbents_lecture_scale_reason'].fillna('').apply(len)
    )

    # 전공 관련
    if 'major1_2' in df.columns:
        df['has_double_major'] = df['major1_2'].notna().astype(int)
    if 'major_data' in df.columns:
        df['is_non_major'] = (~df['major_data']).astype(int)
    if 'major type' in df.columns:
        df['is_double_major_type'] = df['major type'].str.contains('복수|다중|이중', na=False).astype(int)

    # 이전 수강 이력
    prev_cols = [col for col in df.columns if col.startswith('previous_class')]
    if prev_cols:
        df['prev_fill_cnt'] = df[prev_cols].notna().sum(axis=1)
        df['has_prev_class'] = (df['prev_fill_cnt'] > 0).astype(int)

    # 결측치 요약
    df['missing_cnt'] = df.isna().sum(axis=1)

    return df


def get_kill_indices(df: pd.DataFrame) -> list:
    """
    Hard Kill Rule 적용 대상 인덱스 반환.

    법학 관련 / 경영학×직장인·취준생 → 수료율 0% 패턴
    주의: L1=0.421 (보통), Support 부족 — 제한적 신뢰
    """
    kill = []
    kill.extend(df[
        df['major1_1'].str.contains('법', na=False) |
        (df['major_field'] == '법학')
    ].index.tolist())
    kill.extend(df[
        (df['major_field'] == '경영학') &
        (df['job'].isin(['직장인', '취준생']))
    ].index.tolist())
    return list(set(kill))


def get_categorical_features(df: pd.DataFrame) -> list:
    """범주형 변수 목록 반환 (ID, Target 제외)"""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return [c for c in cat_cols if c not in ['ID', 'completed']]


def get_numerical_features(df: pd.DataFrame) -> list:
    """수치형 변수 목록 반환 (ID, Target 제외)"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in ['ID', 'completed']]
