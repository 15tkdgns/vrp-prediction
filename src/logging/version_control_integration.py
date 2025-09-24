#!/usr/bin/env python3
"""
🔧 버전 관리 통합 시스템
학술 논문을 위한 Git 기반 실험 버전 추적 도구

주요 기능:
- Git 커밋 자동 추적
- 실험별 브랜치 관리
- 코드 변경 이력 추적
- 버전 간 실험 결과 비교
"""

import subprocess
import json
import hashlib
import datetime
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class GitInfo:
    """Git 정보"""
    commit_hash: str
    branch: str
    commit_message: str
    author: str
    commit_date: str
    changed_files: List[str]
    is_dirty: bool
    remote_url: Optional[str] = None

@dataclass
class VersionedExperiment:
    """버전 추적된 실험"""
    experiment_id: str
    git_info: GitInfo
    experiment_data: Dict[str, Any]
    code_hash: str
    dependencies_hash: str

class VersionControlIntegrator:
    """
    Git 기반 실험 버전 관리 시스템

    실험과 코드 변경을 연결하여 완전한 재현성 보장
    """

    def __init__(self, repo_path: str = ".", auto_commit: bool = False):
        """
        초기화

        Args:
            repo_path: Git 저장소 경로
            auto_commit: 실험 전 자동 커밋 여부
        """
        self.repo_path = Path(repo_path)
        self.auto_commit = auto_commit

        # Git 저장소 확인
        if not self._is_git_repo():
            raise ValueError(f"{repo_path}는 Git 저장소가 아닙니다.")

    def _is_git_repo(self) -> bool:
        """Git 저장소 여부 확인"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def get_current_git_info(self) -> GitInfo:
        """현재 Git 상태 정보 수집"""
        try:
            # 현재 커밋 해시
            commit_hash = self._run_git_command(['rev-parse', 'HEAD']).strip()

            # 현재 브랜치
            branch = self._run_git_command(['rev-parse', '--abbrev-ref', 'HEAD']).strip()

            # 커밋 메시지
            commit_message = self._run_git_command(['log', '-1', '--pretty=format:%s']).strip()

            # 작성자
            author = self._run_git_command(['log', '-1', '--pretty=format:%an <%ae>']).strip()

            # 커밋 날짜
            commit_date = self._run_git_command(['log', '-1', '--pretty=format:%ci']).strip()

            # 변경된 파일들 (마지막 커밋)
            changed_files = self._run_git_command([
                'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash
            ]).strip().split('\n') if commit_hash else []

            # Working directory가 dirty한지 확인
            is_dirty = self._check_dirty_status()

            # 원격 저장소 URL
            remote_url = self._get_remote_url()

            return GitInfo(
                commit_hash=commit_hash,
                branch=branch,
                commit_message=commit_message,
                author=author,
                commit_date=commit_date,
                changed_files=[f for f in changed_files if f],
                is_dirty=is_dirty,
                remote_url=remote_url
            )

        except Exception as e:
            print(f"Git 정보 수집 오류: {str(e)}")
            # 기본값 반환
            return GitInfo(
                commit_hash="unknown",
                branch="unknown",
                commit_message="unknown",
                author="unknown",
                commit_date="unknown",
                changed_files=[],
                is_dirty=True
            )

    def _run_git_command(self, args: List[str]) -> str:
        """Git 명령 실행"""
        result = subprocess.run(
            ['git'] + args,
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Git 명령 실패: {' '.join(['git'] + args)}\n{result.stderr}")

        return result.stdout

    def _check_dirty_status(self) -> bool:
        """Working directory의 dirty 상태 확인"""
        try:
            # Staged changes
            staged = self._run_git_command(['diff', '--cached', '--name-only']).strip()

            # Unstaged changes
            unstaged = self._run_git_command(['diff', '--name-only']).strip()

            # Untracked files
            untracked = self._run_git_command(['ls-files', '--others', '--exclude-standard']).strip()

            return bool(staged or unstaged or untracked)

        except Exception:
            return True

    def _get_remote_url(self) -> Optional[str]:
        """원격 저장소 URL 가져오기"""
        try:
            return self._run_git_command(['config', '--get', 'remote.origin.url']).strip()
        except Exception:
            return None

    def create_experiment_branch(self, experiment_name: str) -> str:
        """실험용 브랜치 생성"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"experiment/{experiment_name}_{timestamp}"

        try:
            # 브랜치 생성 및 체크아웃
            self._run_git_command(['checkout', '-b', branch_name])
            print(f"🌿 실험 브랜치 생성: {branch_name}")
            return branch_name

        except Exception as e:
            print(f"브랜치 생성 실패: {str(e)}")
            return self.get_current_git_info().branch

    def commit_experiment_changes(self, experiment_id: str,
                                description: str,
                                files_to_add: Optional[List[str]] = None) -> str:
        """실험 관련 변경사항 커밋"""
        try:
            # 변경된 파일들 스테이징
            if files_to_add:
                for file_path in files_to_add:
                    self._run_git_command(['add', file_path])
            else:
                # 모든 변경사항 스테이징 (주의해서 사용)
                self._run_git_command(['add', '.'])

            # 커밋 메시지 생성
            commit_message = f"Experiment {experiment_id}: {description}\n\n🧪 Generated by experiment tracking system"

            # 커밋 실행
            self._run_git_command(['commit', '-m', commit_message])

            # 새 커밋 해시 반환
            new_commit = self._run_git_command(['rev-parse', 'HEAD']).strip()
            print(f"💾 실험 변경사항 커밋: {new_commit[:8]}")
            return new_commit

        except Exception as e:
            print(f"커밋 실패: {str(e)}")
            return ""

    def calculate_code_hash(self, include_patterns: Optional[List[str]] = None) -> str:
        """코드 해시 계산"""
        if include_patterns is None:
            include_patterns = ['*.py', '*.yaml', '*.yml', '*.json']

        file_hashes = []

        for pattern in include_patterns:
            try:
                # Git ls-files로 버전 관리되는 파일만 포함
                files = self._run_git_command(['ls-files', pattern]).strip().split('\n')

                for file_path in files:
                    if file_path and os.path.exists(file_path):
                        try:
                            with open(file_path, 'rb') as f:
                                file_content = f.read()
                                file_hash = hashlib.sha256(file_content).hexdigest()
                                file_hashes.append(f"{file_path}:{file_hash}")
                        except Exception:
                            continue

            except Exception:
                continue

        # 전체 코드 해시 생성
        combined_hash = hashlib.sha256('\n'.join(sorted(file_hashes)).encode()).hexdigest()
        return combined_hash

    def calculate_dependencies_hash(self, requirements_files: Optional[List[str]] = None) -> str:
        """의존성 해시 계산"""
        if requirements_files is None:
            requirements_files = ['requirements.txt', 'pyproject.toml', 'environment.yml']

        dependency_content = []

        for req_file in requirements_files:
            req_path = self.repo_path / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        dependency_content.append(f"{req_file}:\n{content}")
                except Exception:
                    continue

        # 설치된 패키지 정보도 포함
        try:
            import pkg_resources
            installed_packages = []
            for pkg in pkg_resources.working_set:
                installed_packages.append(f"{pkg.project_name}=={pkg.version}")

            dependency_content.append("installed_packages:\n" + '\n'.join(sorted(installed_packages)))
        except Exception:
            pass

        # 의존성 해시 생성
        combined_content = '\n---\n'.join(dependency_content)
        return hashlib.sha256(combined_content.encode()).hexdigest()

    def track_experiment(self, experiment_id: str, experiment_data: Dict[str, Any]) -> VersionedExperiment:
        """실험 버전 추적"""
        print(f"🔍 실험 버전 추적: {experiment_id}")

        # 자동 커밋 (옵션)
        if self.auto_commit and self._check_dirty_status():
            print("  변경사항 자동 커밋 중...")
            self.commit_experiment_changes(
                experiment_id,
                "Auto-commit before experiment execution"
            )

        # Git 정보 수집
        git_info = self.get_current_git_info()

        # 코드 및 의존성 해시 계산
        code_hash = self.calculate_code_hash()
        dependencies_hash = self.calculate_dependencies_hash()

        # Dirty 상태 경고
        if git_info.is_dirty:
            print("  ⚠️ 경고: Working directory에 커밋되지 않은 변경사항이 있습니다.")
            print("    실험의 완전한 재현성을 위해 변경사항을 커밋하는 것을 권장합니다.")

        versioned_experiment = VersionedExperiment(
            experiment_id=experiment_id,
            git_info=git_info,
            experiment_data=experiment_data.copy(),
            code_hash=code_hash,
            dependencies_hash=dependencies_hash
        )

        return versioned_experiment

    def compare_experiment_versions(self, version1: VersionedExperiment,
                                  version2: VersionedExperiment) -> Dict[str, Any]:
        """실험 버전 간 비교"""
        comparison = {
            'same_commit': version1.git_info.commit_hash == version2.git_info.commit_hash,
            'same_branch': version1.git_info.branch == version2.git_info.branch,
            'same_code': version1.code_hash == version2.code_hash,
            'same_dependencies': version1.dependencies_hash == version2.dependencies_hash,
            'commit_diff': {
                'version1': {
                    'commit': version1.git_info.commit_hash[:8],
                    'branch': version1.git_info.branch,
                    'date': version1.git_info.commit_date
                },
                'version2': {
                    'commit': version2.git_info.commit_hash[:8],
                    'branch': version2.git_info.branch,
                    'date': version2.git_info.commit_date
                }
            }
        }

        # 성능 차이 계산 (가능한 경우)
        if 'metrics' in version1.experiment_data and 'metrics' in version2.experiment_data:
            metrics1 = version1.experiment_data['metrics']
            metrics2 = version2.experiment_data['metrics']

            performance_diff = {}
            for metric in metrics1.keys():
                if metric in metrics2 and isinstance(metrics1[metric], (int, float)):
                    diff = metrics2[metric] - metrics1[metric]
                    performance_diff[metric] = {
                        'version1': metrics1[metric],
                        'version2': metrics2[metric],
                        'difference': diff,
                        'relative_change': diff / metrics1[metric] * 100 if metrics1[metric] != 0 else 0
                    }

            comparison['performance_diff'] = performance_diff

        return comparison

    def get_experiment_lineage(self, experiment_id: str) -> List[Dict[str, Any]]:
        """실험 계보 추적 (Git 히스토리 기반)"""
        try:
            # Git log에서 실험 관련 커밋들 찾기
            log_output = self._run_git_command([
                'log', '--oneline', '--grep=f"Experiment.*{experiment_id}"'
            ])

            lineage = []
            for line in log_output.strip().split('\n'):
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) >= 2:
                        commit_hash = parts[0]
                        commit_message = parts[1]

                        # 커밋 상세 정보
                        commit_info = self._get_commit_info(commit_hash)
                        lineage.append({
                            'commit_hash': commit_hash,
                            'commit_message': commit_message,
                            'commit_info': commit_info
                        })

            return lineage

        except Exception as e:
            print(f"실험 계보 추적 오류: {str(e)}")
            return []

    def _get_commit_info(self, commit_hash: str) -> Dict[str, str]:
        """특정 커밋의 상세 정보"""
        try:
            author = self._run_git_command(['show', '-s', '--format=%an <%ae>', commit_hash]).strip()
            date = self._run_git_command(['show', '-s', '--format=%ci', commit_hash]).strip()
            message = self._run_git_command(['show', '-s', '--format=%B', commit_hash]).strip()

            return {
                'author': author,
                'date': date,
                'message': message
            }
        except Exception:
            return {}

    def generate_version_report(self, versioned_experiment: VersionedExperiment) -> str:
        """버전 추적 보고서 생성"""
        git_info = versioned_experiment.git_info
        experiment_data = versioned_experiment.experiment_data

        report = [
            f"🔧 실험 버전 추적 보고서: {versioned_experiment.experiment_id}",
            "=" * 60,
            ""
        ]

        # Git 정보
        report.extend([
            "📚 Git 버전 정보:",
            f"   커밋 해시: {git_info.commit_hash[:8]}...{git_info.commit_hash[-8:]}",
            f"   브랜치: {git_info.branch}",
            f"   커밋 메시지: {git_info.commit_message}",
            f"   작성자: {git_info.author}",
            f"   커밋 날짜: {git_info.commit_date}",
            f"   Working Directory 상태: {'Clean' if not git_info.is_dirty else 'Dirty (⚠️ 경고)'}",
            ""
        ])

        if git_info.remote_url:
            report.append(f"   원격 저장소: {git_info.remote_url}")
            report.append("")

        # 변경된 파일들
        if git_info.changed_files:
            report.extend([
                "📝 마지막 커밋에서 변경된 파일들:",
            ])
            for file_name in git_info.changed_files[:10]:  # 최대 10개만 표시
                report.append(f"   - {file_name}")

            if len(git_info.changed_files) > 10:
                report.append(f"   ... 및 {len(git_info.changed_files) - 10}개 파일 더")
            report.append("")

        # 코드 및 의존성 해시
        report.extend([
            "🔐 무결성 해시:",
            f"   코드 해시: {versioned_experiment.code_hash[:16]}...",
            f"   의존성 해시: {versioned_experiment.dependencies_hash[:16]}...",
            ""
        ])

        # 실험 정보
        if 'config' in experiment_data:
            config = experiment_data['config']
            report.extend([
                "🧪 실험 구성:",
                f"   실험명: {config.get('experiment_name', 'N/A')}",
                f"   모델 유형: {config.get('model_type', 'N/A')}",
                f"   랜덤 시드: {config.get('random_seed', 'N/A')}",
                ""
            ])

        # 재현성 가이드
        report.extend([
            "🔄 재현성 가이드:",
            f"   1. git checkout {git_info.commit_hash}",
            "   2. 동일한 Python 환경 및 의존성 설치",
            "   3. 동일한 랜덤 시드로 실험 재실행",
            ""
        ])

        if git_info.is_dirty:
            report.extend([
                "⚠️ 재현성 경고:",
                "   Working directory에 커밋되지 않은 변경사항이 있습니다.",
                "   완전한 재현을 위해서는 해당 변경사항을 복원해야 합니다.",
                ""
            ])

        return "\n".join(report)

def main():
    """테스트 및 예제 실행"""
    print("🔧 버전 관리 통합 시스템 테스트")
    print("=" * 60)

    try:
        # 버전 관리 통합기 초기화
        integrator = VersionControlIntegrator(auto_commit=False)

        # 1. 현재 Git 상태 확인
        print("1. 현재 Git 상태 확인")
        git_info = integrator.get_current_git_info()
        print(f"   현재 브랜치: {git_info.branch}")
        print(f"   커밋: {git_info.commit_hash[:8]}")
        print(f"   상태: {'Clean' if not git_info.is_dirty else 'Dirty'}")
        print()

        # 2. 코드 및 의존성 해시 계산
        print("2. 코드 및 의존성 해시 계산")
        code_hash = integrator.calculate_code_hash()
        dep_hash = integrator.calculate_dependencies_hash()
        print(f"   코드 해시: {code_hash[:16]}...")
        print(f"   의존성 해시: {dep_hash[:16]}...")
        print()

        # 3. 테스트 실험 추적
        print("3. 실험 버전 추적 테스트")
        test_experiment_data = {
            'config': {
                'experiment_name': 'Version_Control_Test',
                'model_type': 'TestModel',
                'random_seed': 42
            },
            'metrics': {
                'mae': 0.123,
                'r2': 0.456
            }
        }

        versioned_exp = integrator.track_experiment(
            "test_version_control_001",
            test_experiment_data
        )

        print(f"   실험 ID: {versioned_exp.experiment_id}")
        print(f"   Git 커밋: {versioned_exp.git_info.commit_hash[:8]}")
        print()

        # 4. 버전 보고서 생성
        print("4. 버전 추적 보고서")
        print("-" * 50)
        report = integrator.generate_version_report(versioned_exp)
        print(report)

        # 5. 실험 계보 조회 (시뮬레이션)
        print("5. 실험 계보 조회")
        lineage = integrator.get_experiment_lineage("test_version_control")
        if lineage:
            for entry in lineage:
                print(f"   커밋: {entry['commit_hash']} - {entry['commit_message']}")
        else:
            print("   관련 커밋이 없습니다.")

    except ValueError as e:
        print(f"오류: {str(e)}")
        print("Git 저장소가 아닌 경우 초기화:")
        print("  git init")
        print("  git add .")
        print("  git commit -m 'Initial commit'")

    except Exception as e:
        print(f"예상치 못한 오류: {str(e)}")

if __name__ == "__main__":
    main()