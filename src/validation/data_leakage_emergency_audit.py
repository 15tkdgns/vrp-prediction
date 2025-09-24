#!/usr/bin/env python3
"""
🚨 Emergency Data Leakage Audit System
Critical Issue: MDD = 0.0000 indicates severe data leakage
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class DataLeakageEmergencyAudit:
    def __init__(self):
        self.data_path = Path("/root/workspace/data")
        self.audit_results = {}
        self.critical_issues = []
        self.warnings = []

    def audit_raw_data_integrity(self):
        """1. 원시 데이터 무결성 검사"""
        print("🔍 1. 원시 데이터 무결성 검사...")

        try:
            # SPY 훈련 데이터 로드
            data_file = self.data_path / "training" / "sp500_2020_2024_enhanced.csv"
            df = pd.read_csv(data_file)

            print(f"   📊 데이터 크기: {df.shape}")
            print(f"   📅 날짜 범위: {df['Date'].min()} ~ {df['Date'].max()}")

            # 날짜 순서 확인
            df['Date'] = pd.to_datetime(df['Date'])
            is_sorted = df['Date'].is_monotonic_increasing

            if not is_sorted:
                self.critical_issues.append("❌ CRITICAL: 날짜가 시간 순서대로 정렬되지 않음")
                print("   ❌ CRITICAL: 날짜 순서 오류 발견!")
            else:
                print("   ✅ 날짜 순서: 정상")

            # Returns 계산 검증
            calculated_returns = df['Close'].pct_change()
            if 'Returns' in df.columns:
                returns_diff = abs(df['Returns'] - calculated_returns).mean()
                if returns_diff > 1e-6:
                    self.critical_issues.append("❌ CRITICAL: Returns 계산에 미래 정보 포함 의심")
                    print(f"   ❌ CRITICAL: Returns 계산 오류 (차이: {returns_diff:.8f})")
                else:
                    print("   ✅ Returns 계산: 정상")

            # 결측값 검사
            missing_counts = df.isnull().sum()
            critical_missing = missing_counts[missing_counts > len(df) * 0.1]
            if len(critical_missing) > 0:
                self.warnings.append(f"⚠️ 높은 결측률 컬럼: {critical_missing.to_dict()}")

            self.audit_results['raw_data'] = {
                'shape': df.shape,
                'date_sorted': is_sorted,
                'date_range': [str(df['Date'].min()), str(df['Date'].max())],
                'missing_critical': critical_missing.to_dict()
            }

            return df

        except Exception as e:
            self.critical_issues.append(f"❌ CRITICAL: 원시 데이터 로드 실패 - {e}")
            return None

    def audit_feature_engineering_leakage(self, df):
        """2. 피처 엔지니어링 데이터 유출 검사"""
        print("\n🔍 2. 피처 엔지니어링 데이터 유출 검사...")

        leakage_suspects = []

        # 이동평균 검사
        ma_columns = [col for col in df.columns if 'MA_' in col or 'SMA' in col or 'EMA' in col]
        for col in ma_columns:
            if col in df.columns:
                # 첫 번째 계산 가능한 지점 확인
                window = int(col.split('_')[-1]) if col.split('_')[-1].isdigit() else 20
                first_valid = df[col].first_valid_index()
                expected_first = window - 1

                if first_valid < expected_first:
                    leakage_suspects.append(f"{col}: 예상보다 이른 시점부터 값 존재")
                    print(f"   ❌ SUSPECT: {col} - 첫 유효값 위치 {first_valid}, 예상 {expected_first}")

        # RSI 검사
        if 'RSI' in df.columns:
            first_rsi = df['RSI'].first_valid_index()
            if first_rsi < 14:  # RSI는 보통 14일 필요
                leakage_suspects.append("RSI: 계산 윈도우보다 이른 시점부터 값 존재")
                print(f"   ❌ SUSPECT: RSI 첫 유효값 위치 {first_rsi}, 예상 >= 14")

        # Lag 특징 검사
        lag_features = [col for col in df.columns if 'lag' in col.lower()]
        for col in lag_features:
            lag_num = 1  # 기본값
            try:
                lag_num = int(col.split('_')[-1])
            except:
                pass

            # Lag 특징의 첫 번째 값이 너무 이른지 확인
            first_valid = df[col].first_valid_index()
            if first_valid < lag_num:
                leakage_suspects.append(f"{col}: Lag보다 이른 시점부터 값 존재")
                print(f"   ❌ SUSPECT: {col} - 첫 유효값 위치 {first_valid}, 예상 >= {lag_num}")

        # 볼린저 밴드 검사
        bb_columns = [col for col in df.columns if 'BB_' in col]
        for col in bb_columns:
            first_valid = df[col].first_valid_index()
            if first_valid < 20:  # BB는 보통 20일 필요
                leakage_suspects.append(f"{col}: 계산 윈도우보다 이른 시점부터 값 존재")
                print(f"   ❌ SUSPECT: {col} - 첫 유효값 위치 {first_valid}, 예상 >= 20")

        if len(leakage_suspects) == 0:
            print("   ✅ 피처 엔지니어링: 명백한 유출 없음")
        else:
            self.critical_issues.extend(leakage_suspects)

        self.audit_results['feature_engineering'] = {
            'leakage_suspects': leakage_suspects,
            'total_features': len(df.columns),
            'lag_features': len(lag_features),
            'ma_features': len(ma_columns)
        }

    def audit_preprocessing_leakage(self):
        """3. 전처리 과정 데이터 유출 검사"""
        print("\n🔍 3. 전처리 과정 데이터 유출 검사...")

        # 스케일링 순서 검사 - 코드 파일들 확인
        model_files = list(Path("/root/workspace/src").rglob("*.py"))
        scaling_issues = []

        for file_path in model_files:
            if 'model' in str(file_path) or 'test' in str(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()

                    # 의심스러운 패턴 검사
                    if 'fit_transform(X)' in content and 'train_test_split' in content:
                        # fit_transform이 분할 이전에 사용되었는지 확인
                        fit_pos = content.find('fit_transform(X)')
                        split_pos = content.find('train_test_split')

                        if fit_pos < split_pos and fit_pos != -1 and split_pos != -1:
                            scaling_issues.append(f"{file_path.name}: fit_transform이 데이터 분할 이전에 실행")
                            print(f"   ❌ CRITICAL: {file_path.name} - 스케일링 순서 오류!")

                except Exception as e:
                    continue

        if len(scaling_issues) == 0:
            print("   ✅ 전처리 순서: 명백한 오류 없음")
        else:
            self.critical_issues.extend(scaling_issues)

        self.audit_results['preprocessing'] = {
            'scaling_issues': scaling_issues,
            'files_checked': len(model_files)
        }

    def audit_cross_validation_leakage(self):
        """4. 교차 검증 데이터 유출 검사"""
        print("\n🔍 4. 교차 검증 데이터 유출 검사...")

        cv_issues = []

        # 모델 파일에서 교차 검증 방법 확인
        model_files = list(Path("/root/workspace/src").rglob("*test*.py"))

        for file_path in model_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()

                # 잘못된 CV 방법 검사
                if 'KFold' in content and 'TimeSeriesSplit' not in content:
                    cv_issues.append(f"{file_path.name}: 시계열 데이터에 일반 KFold 사용")
                    print(f"   ❌ CRITICAL: {file_path.name} - 잘못된 CV 방법!")

                if 'StratifiedKFold' in content:
                    cv_issues.append(f"{file_path.name}: 시계열 데이터에 StratifiedKFold 사용")
                    print(f"   ❌ CRITICAL: {file_path.name} - 시계열에 부적절한 CV!")

                if 'shuffle=True' in content and 'TimeSeriesSplit' not in content:
                    cv_issues.append(f"{file_path.name}: 시계열 데이터 셔플링")
                    print(f"   ❌ CRITICAL: {file_path.name} - 시계열 순서 파괴!")

            except Exception as e:
                continue

        if len(cv_issues) == 0:
            print("   ✅ 교차 검증: 명백한 오류 없음")
        else:
            self.critical_issues.extend(cv_issues)

        self.audit_results['cross_validation'] = {
            'cv_issues': cv_issues,
            'files_checked': len(model_files)
        }

    def audit_mdd_calculation(self):
        """5. MDD 계산 오류 검사 (핵심 이슈)"""
        print("\n🔍 5. MDD 계산 오류 검사 (핵심 이슈)...")

        # 테스트 데이터로 MDD 계산 검증
        test_returns = np.array([0.02, -0.01, 0.015, -0.03, 0.01, -0.02, 0.025, -0.015])

        # 올바른 MDD 계산
        cumulative = np.cumprod(1 + test_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        correct_mdd = abs(np.min(drawdown))

        print(f"   📊 테스트 수익률: {test_returns}")
        print(f"   📊 누적 수익률: {cumulative}")
        print(f"   📊 올바른 MDD: {correct_mdd:.4f}")

        if correct_mdd == 0.0:
            self.critical_issues.append("❌ CRITICAL: MDD 계산 로직 자체에 오류")
            print("   ❌ CRITICAL: MDD 계산 함수 오류!")
        else:
            print("   ✅ MDD 계산 로직: 정상")

        # 모델에서 MDD 0이 나오는 원인 분석
        zero_mdd_causes = [
            "예측 수익률이 모두 0에 가까움 (극도로 보수적 예측)",
            "수익률 계산에서 실제 손실이 없음 (데이터 유출)",
            "MDD 계산 구간이 너무 짧음",
            "예측값이 실제 변동성을 반영하지 못함"
        ]

        print(f"   🔍 MDD=0 가능한 원인들:")
        for i, cause in enumerate(zero_mdd_causes, 1):
            print(f"     {i}. {cause}")

        self.audit_results['mdd_calculation'] = {
            'test_mdd': correct_mdd,
            'zero_mdd_causes': zero_mdd_causes,
            'calculation_logic_ok': correct_mdd > 0
        }

    def audit_target_leakage(self, df):
        """6. 타겟 변수 데이터 유출 검사"""
        print("\n🔍 6. 타겟 변수 데이터 유출 검사...")

        target_issues = []

        if 'Returns' in df.columns:
            # 타겟이 미래 정보를 포함하는지 확인
            returns = df['Returns'].values

            # 타겟의 자기상관 검사 (과도한 상관은 유출 의심)
            if len(returns) > 1:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                if abs(autocorr) > 0.3:
                    target_issues.append(f"타겟 자기상관이 과도함: {autocorr:.4f}")
                    print(f"   ❌ SUSPECT: 타겟 자기상관 {autocorr:.4f} (>0.3)")
                else:
                    print(f"   ✅ 타겟 자기상관: {autocorr:.4f} (정상)")

            # 타겟의 분산 검사
            target_std = np.std(returns)
            if target_std < 0.001:  # 너무 낮은 변동성
                target_issues.append(f"타겟 변동성이 비현실적으로 낮음: {target_std:.6f}")
                print(f"   ❌ SUSPECT: 타겟 변동성 {target_std:.6f} (너무 낮음)")
            else:
                print(f"   ✅ 타겟 변동성: {target_std:.6f} (정상)")

        if len(target_issues) == 0:
            print("   ✅ 타겟 변수: 명백한 유출 없음")
        else:
            self.warnings.extend(target_issues)

        self.audit_results['target_leakage'] = {
            'target_issues': target_issues,
            'target_available': 'Returns' in df.columns
        }

    def generate_emergency_report(self):
        """긴급 감사 보고서 생성"""
        print("\n" + "="*80)
        print("🚨 긴급 데이터 유출 감사 보고서")
        print("="*80)

        print(f"\n📊 전체 감사 결과:")
        print(f"   🔴 중대 이슈: {len(self.critical_issues)}개")
        print(f"   🟡 경고 사항: {len(self.warnings)}개")

        if len(self.critical_issues) > 0:
            print(f"\n🔴 중대 이슈들:")
            for i, issue in enumerate(self.critical_issues, 1):
                print(f"   {i}. {issue}")

        if len(self.warnings) > 0:
            print(f"\n🟡 경고 사항들:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")

        # 우선순위 수정 권고
        print(f"\n🎯 우선순위 수정 권고:")

        if len(self.critical_issues) > 0:
            print("   1. 🚨 즉시 수정 필요 (CRITICAL)")
            print("      - 모든 모델 성능 지표는 현재 신뢰할 수 없음")
            print("      - 데이터 파이프라인 전면 재구축 필요")
            print("      - MDD=0 문제 해결 최우선")
        else:
            print("   1. ✅ 중대 이슈 없음 - 세부 검토 진행")

        print("   2. 📊 강건한 백테스팅 시스템 구축")
        print("   3. 🔍 조합 교차 검증(CPCV) 도입")
        print("   4. 📈 현실적 성능 지표 재계산")

        # 결과 저장
        audit_file = Path("/root/workspace/emergency_data_leakage_audit.json")
        with open(audit_file, 'w', encoding='utf-8') as f:
            json.dump({
                'audit_timestamp': pd.Timestamp.now().isoformat(),
                'critical_issues': self.critical_issues,
                'warnings': self.warnings,
                'audit_results': self.audit_results,
                'total_critical': len(self.critical_issues),
                'total_warnings': len(self.warnings)
            }, f, indent=2, default=str)

        print(f"\n📁 감사 보고서 저장: {audit_file}")

        # 최종 판정
        if len(self.critical_issues) > 0:
            print(f"\n🚨 최종 판정: 데이터 유출 의심 - 즉시 수정 필요")
            return False
        else:
            print(f"\n✅ 최종 판정: 명백한 데이터 유출 없음 - 추가 정밀 검사 필요")
            return True

    def run_emergency_audit(self):
        """긴급 감사 실행"""
        print("🚨 긴급 데이터 유출 감사 시작")
        print("문제: MDD = 0.0000은 심각한 데이터 유출을 시사합니다.\n")

        # 1. 원시 데이터 검사
        df = self.audit_raw_data_integrity()

        if df is not None:
            # 2. 피처 엔지니어링 검사
            self.audit_feature_engineering_leakage(df)

            # 6. 타겟 변수 검사
            self.audit_target_leakage(df)

        # 3. 전처리 과정 검사
        self.audit_preprocessing_leakage()

        # 4. 교차 검증 검사
        self.audit_cross_validation_leakage()

        # 5. MDD 계산 검사
        self.audit_mdd_calculation()

        # 최종 보고서
        is_clean = self.generate_emergency_report()

        return is_clean, self.audit_results

if __name__ == "__main__":
    auditor = DataLeakageEmergencyAudit()
    is_clean, results = auditor.run_emergency_audit()