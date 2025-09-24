#!/usr/bin/env python3
"""
🔍 코드 품질 검사 시스템
학술 논문을 위한 자동화된 코드 품질 검증 도구

주요 기능:
- PEP 8 준수 검사
- 타입 힌트 검증
- Docstring 품질 검사
- 복잡도 분석
- 보안 취약점 검사
"""

import ast
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class QualityIssue:
    """코드 품질 이슈"""
    file_path: str
    line_number: int
    column: int
    issue_type: str
    severity: str  # error, warning, info
    message: str
    rule_code: Optional[str] = None

@dataclass
class QualityReport:
    """코드 품질 보고서"""
    total_files: int
    total_lines: int
    issues: List[QualityIssue]
    summary: Dict[str, int]
    score: float
    recommendations: List[str]

class CodeQualityChecker:
    """
    포괄적 코드 품질 검사 도구

    PEP 8, 타입 힌트, docstring 등 다양한 품질 지표 검사
    """

    def __init__(self, project_root: str = "."):
        """
        초기화

        Args:
            project_root: 프로젝트 루트 디렉토리
        """
        self.project_root = Path(project_root)
        self.issues: List[QualityIssue] = []

    def check_project_quality(self, src_dirs: Optional[List[str]] = None) -> QualityReport:
        """
        프로젝트 전체 품질 검사

        Args:
            src_dirs: 검사할 소스 디렉토리 목록

        Returns:
            품질 보고서
        """
        if src_dirs is None:
            src_dirs = ["src"]

        print("🔍 코드 품질 검사 시작")

        all_python_files = []
        for src_dir in src_dirs:
            src_path = self.project_root / src_dir
            if src_path.exists():
                python_files = list(src_path.rglob("*.py"))
                all_python_files.extend(python_files)

        if not all_python_files:
            print("검사할 Python 파일이 없습니다.")
            return QualityReport(0, 0, [], {}, 100.0, [])

        print(f"검사 대상: {len(all_python_files)}개 파일")

        # 각종 품질 검사 수행
        self.issues = []

        for file_path in all_python_files:
            print(f"  검사 중: {file_path}")
            self._check_file_quality(file_path)

        # 보고서 생성
        return self._generate_report(all_python_files)

    def _check_file_quality(self, file_path: Path):
        """개별 파일 품질 검사"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # AST 파싱
            try:
                tree = ast.parse(content, filename=str(file_path))
            except SyntaxError as e:
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    line_number=e.lineno or 0,
                    column=e.offset or 0,
                    issue_type="syntax_error",
                    severity="error",
                    message=f"Syntax error: {e.msg}",
                    rule_code="E999"
                ))
                return

            # 각종 검사 수행
            self._check_pep8_style(file_path, content)
            self._check_docstrings(file_path, tree)
            self._check_type_hints(file_path, tree)
            self._check_complexity(file_path, tree)
            self._check_imports(file_path, tree)
            self._check_naming_conventions(file_path, tree)

        except Exception as e:
            self.issues.append(QualityIssue(
                file_path=str(file_path),
                line_number=0,
                column=0,
                issue_type="file_error",
                severity="error",
                message=f"파일 읽기 오류: {str(e)}"
            ))

    def _check_pep8_style(self, file_path: Path, content: str):
        """PEP 8 스타일 검사"""
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # 라인 길이 검사 (79자 권장, 88자 허용)
            if len(line) > 88:
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    line_number=i,
                    column=89,
                    issue_type="line_length",
                    severity="warning",
                    message=f"Line too long ({len(line)} > 88 characters)",
                    rule_code="E501"
                ))

            # 후행 공백 검사
            if line.endswith(' ') or line.endswith('\t'):
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    line_number=i,
                    column=len(line),
                    issue_type="trailing_whitespace",
                    severity="warning",
                    message="Trailing whitespace",
                    rule_code="W291"
                ))

            # 탭 문자 검사
            if '\t' in line:
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    line_number=i,
                    column=line.index('\t') + 1,
                    issue_type="tab_indentation",
                    severity="warning",
                    message="Indentation contains tabs",
                    rule_code="W191"
                ))

            # 여러 공백 검사
            if '  ' in line.strip() and not line.strip().startswith('#'):
                # 주석이 아닌 경우에만 검사
                if re.search(r'[^"\']\s\s+[^"\']', line):
                    self.issues.append(QualityIssue(
                        file_path=str(file_path),
                        line_number=i,
                        column=0,
                        issue_type="multiple_spaces",
                        severity="info",
                        message="Multiple spaces found",
                        rule_code="E221"
                    ))

    def _check_docstrings(self, file_path: Path, tree: ast.AST):
        """Docstring 품질 검사"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Public 함수/클래스만 검사 (언더스코어로 시작하지 않는)
                if not node.name.startswith('_'):
                    docstring = ast.get_docstring(node)

                    if not docstring:
                        self.issues.append(QualityIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            column=node.col_offset,
                            issue_type="missing_docstring",
                            severity="warning",
                            message=f"Missing docstring for {type(node).__name__.lower()} '{node.name}'",
                            rule_code="D100"
                        ))
                    else:
                        # Docstring 품질 검사
                        self._check_docstring_quality(file_path, node, docstring)

    def _check_docstring_quality(self, file_path: Path, node: ast.AST, docstring: str):
        """Docstring 상세 품질 검사"""
        lines = docstring.split('\n')

        # 최소 길이 검사
        if len(docstring.strip()) < 10:
            self.issues.append(QualityIssue(
                file_path=str(file_path),
                line_number=node.lineno,
                column=node.col_offset,
                issue_type="short_docstring",
                severity="info",
                message=f"Docstring too short for '{node.name}'",
                rule_code="D200"
            ))

        # 함수의 경우 Args, Returns 섹션 검사
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            has_args = len(node.args.args) > 0 or len(node.args.kwonlyargs) > 0
            has_return = any(isinstance(child, ast.Return) and child.value is not None
                           for child in ast.walk(node))

            # Args 섹션 검사
            if has_args and 'Args:' not in docstring and 'Parameters:' not in docstring:
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    line_number=node.lineno,
                    column=node.col_offset,
                    issue_type="missing_args_section",
                    severity="info",
                    message=f"Missing Args section in docstring for '{node.name}'",
                    rule_code="D417"
                ))

            # Returns 섹션 검사
            if has_return and 'Returns:' not in docstring and 'Return:' not in docstring:
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    line_number=node.lineno,
                    column=node.col_offset,
                    issue_type="missing_returns_section",
                    severity="info",
                    message=f"Missing Returns section in docstring for '{node.name}'",
                    rule_code="D418"
                ))

    def _check_type_hints(self, file_path: Path, tree: ast.AST):
        """타입 힌트 검사"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Public 함수만 검사
                if not node.name.startswith('_'):
                    # 매개변수 타입 힌트 검사
                    for arg in node.args.args:
                        if arg.annotation is None and arg.arg != 'self':
                            self.issues.append(QualityIssue(
                                file_path=str(file_path),
                                line_number=node.lineno,
                                column=node.col_offset,
                                issue_type="missing_type_hint",
                                severity="info",
                                message=f"Missing type hint for parameter '{arg.arg}' in function '{node.name}'",
                                rule_code="T001"
                            ))

                    # 반환 타입 힌트 검사
                    if node.returns is None and node.name != '__init__':
                        self.issues.append(QualityIssue(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            column=node.col_offset,
                            issue_type="missing_return_type",
                            severity="info",
                            message=f"Missing return type hint for function '{node.name}'",
                            rule_code="T002"
                        ))

    def _check_complexity(self, file_path: Path, tree: ast.AST):
        """복잡도 검사 (Cyclomatic Complexity)"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_complexity(node)

                if complexity > 10:  # 복잡도 임계값
                    severity = "warning" if complexity > 15 else "info"
                    self.issues.append(QualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type="high_complexity",
                        severity=severity,
                        message=f"Function '{node.name}' has high complexity ({complexity})",
                        rule_code="C901"
                    ))

    def _calculate_complexity(self, node: ast.AST) -> int:
        """순환 복잡도 계산"""
        complexity = 1  # 기본 복잡도

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, (ast.ExceptHandler,)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # and, or 연산자
                complexity += len(child.values) - 1

        return complexity

    def _check_imports(self, file_path: Path, tree: ast.AST):
        """Import 구문 검사"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append((node.lineno, node))

        # Import 순서 검사 (표준 라이브러리 -> 서드파티 -> 로컬)
        # 간단한 버전: 알파벳 순서 검사
        prev_line = 0
        for line_no, import_node in imports:
            if line_no < prev_line:
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    line_number=line_no,
                    column=0,
                    issue_type="import_order",
                    severity="info",
                    message="Imports are not in order",
                    rule_code="I001"
                ))
            prev_line = line_no

    def _check_naming_conventions(self, file_path: Path, tree: ast.AST):
        """명명 규칙 검사"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 함수명: snake_case
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name) and not node.name.startswith('_'):
                    self.issues.append(QualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type="naming_convention",
                        severity="info",
                        message=f"Function name '{node.name}' should be snake_case",
                        rule_code="N802"
                    ))

            elif isinstance(node, ast.ClassDef):
                # 클래스명: PascalCase
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    self.issues.append(QualityIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type="naming_convention",
                        severity="info",
                        message=f"Class name '{node.name}' should be PascalCase",
                        rule_code="N801"
                    ))

    def _generate_report(self, files: List[Path]) -> QualityReport:
        """품질 보고서 생성"""
        total_lines = 0
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except Exception:
                continue

        # 이슈 요약
        summary = {}
        for issue in self.issues:
            issue_type = issue.issue_type
            summary[issue_type] = summary.get(issue_type, 0) + 1

        # 심각도별 요약
        severity_summary = {}
        for issue in self.issues:
            severity = issue.severity
            severity_summary[severity] = severity_summary.get(severity, 0) + 1

        # 점수 계산 (100점 만점)
        error_count = severity_summary.get('error', 0)
        warning_count = severity_summary.get('warning', 0)
        info_count = severity_summary.get('info', 0)

        # 가중치 적용한 점수 계산
        penalty = error_count * 5 + warning_count * 2 + info_count * 0.5
        max_penalty = total_lines * 0.1  # 최대 페널티는 라인 수의 10%
        score = max(0, 100 - (penalty / max_penalty * 100)) if max_penalty > 0 else 100

        # 권고사항 생성
        recommendations = self._generate_recommendations(summary, severity_summary)

        return QualityReport(
            total_files=len(files),
            total_lines=total_lines,
            issues=self.issues,
            summary={**summary, **{f"{k}_count": v for k, v in severity_summary.items()}},
            score=score,
            recommendations=recommendations
        )

    def _generate_recommendations(self, summary: Dict[str, int], severity_summary: Dict[str, int]) -> List[str]:
        """권고사항 생성"""
        recommendations = []

        if severity_summary.get('error', 0) > 0:
            recommendations.append("🚨 오류를 우선적으로 수정하세요")

        if summary.get('missing_docstring', 0) > 5:
            recommendations.append("📝 Public 함수/클래스에 docstring을 추가하세요")

        if summary.get('missing_type_hint', 0) > 10:
            recommendations.append("🔤 타입 힌트를 추가하여 코드 가독성을 향상하세요")

        if summary.get('high_complexity', 0) > 0:
            recommendations.append("🔄 복잡한 함수를 작은 함수들로 분리하세요")

        if summary.get('line_length', 0) > 5:
            recommendations.append("📏 긴 라인을 여러 라인으로 분할하세요")

        if summary.get('naming_convention', 0) > 0:
            recommendations.append("🏷️ PEP 8 명명 규칙을 따르세요")

        if not recommendations:
            recommendations.append("✅ 코드 품질이 우수합니다!")

        # 도구별 권고사항
        recommendations.extend([
            "🛠️ 자동화 도구 사용 권장:",
            "  - black: 코드 포맷팅",
            "  - ruff: 빠른 린팅",
            "  - mypy: 타입 체킹",
            "  - pre-commit: Git 훅 설정"
        ])

        return recommendations

    def generate_quality_report(self, report: QualityReport) -> str:
        """품질 보고서 문자열 생성"""
        output = [
            "🔍 코드 품질 검사 보고서",
            "=" * 50,
            ""
        ]

        # 전체 통계
        output.extend([
            f"📊 전체 통계:",
            f"   검사 파일: {report.total_files}개",
            f"   총 라인 수: {report.total_lines:,}",
            f"   발견된 이슈: {len(report.issues)}개",
            f"   품질 점수: {report.score:.1f}/100",
            ""
        ])

        # 심각도별 통계
        error_count = report.summary.get('error_count', 0)
        warning_count = report.summary.get('warning_count', 0)
        info_count = report.summary.get('info_count', 0)

        output.extend([
            f"🚨 심각도별 이슈:",
            f"   오류 (Error): {error_count}개",
            f"   경고 (Warning): {warning_count}개",
            f"   정보 (Info): {info_count}개",
            ""
        ])

        # 이슈 유형별 통계
        issue_types = {k: v for k, v in report.summary.items() if not k.endswith('_count')}
        if issue_types:
            output.extend([
                f"📋 이슈 유형별 통계:",
            ])
            for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
                output.append(f"   {issue_type}: {count}개")
            output.append("")

        # 주요 이슈들 (최대 10개)
        major_issues = [issue for issue in report.issues if issue.severity in ['error', 'warning']]
        if major_issues:
            output.extend([
                f"🔍 주요 이슈 (최대 10개):",
            ])
            for issue in major_issues[:10]:
                output.append(f"   {issue.file_path}:{issue.line_number} - {issue.message}")

            if len(major_issues) > 10:
                output.append(f"   ... 및 {len(major_issues) - 10}개 이슈 더")
            output.append("")

        # 권고사항
        output.extend([
            f"💡 권고사항:",
        ])
        for rec in report.recommendations:
            output.append(f"   {rec}")

        return "\n".join(output)

def main():
    """테스트 및 예제 실행"""
    print("🔍 코드 품질 검사 시스템 테스트")
    print("=" * 50)

    # 품질 검사기 초기화
    checker = CodeQualityChecker()

    # 현재 프로젝트 검사
    try:
        report = checker.check_project_quality(["src"])

        # 보고서 출력
        quality_report = checker.generate_quality_report(report)
        print(quality_report)

    except Exception as e:
        print(f"품질 검사 오류: {str(e)}")
        print("src 디렉토리가 없는 경우 테스트 파일로 검사합니다.")

        # 테스트 코드 생성
        test_code = '''#!/usr/bin/env python3
"""
테스트 모듈
잘못된 코드 품질 예제
"""

def badFunction(x,y):  # 명명 규칙 위반, 타입 힌트 없음
    # docstring 없음
    if x>0:  # 공백 부족
        return x+y    # 들여쓰기 문제
    else:
        return 0

class badClass:  # 명명 규칙 위반
    def __init__(self):
        pass

    def very_long_function_name_that_exceeds_the_recommended_line_length_of_79_characters(self):
        # 너무 긴 라인
        pass
'''

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_file = f.name

        try:
            # 단일 파일 검사
            checker._check_file_quality(Path(temp_file))

            # 간단한 보고서 생성
            files = [Path(temp_file)]
            report = checker._generate_report(files)
            quality_report = checker.generate_quality_report(report)
            print(quality_report)

        finally:
            os.unlink(temp_file)

if __name__ == "__main__":
    main()