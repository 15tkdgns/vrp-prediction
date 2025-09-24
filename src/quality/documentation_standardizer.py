#!/usr/bin/env python3
"""
📚 문서화 표준화 시스템
학술 논문을 위한 자동화된 문서화 생성 및 검증 도구

주요 기능:
- NumPy/Google 스타일 docstring 자동 생성
- 기존 docstring 품질 검증
- API 문서 자동 생성
- 타입 힌트 기반 문서화
"""

import ast
import re
import inspect
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FunctionInfo:
    """함수 정보"""
    name: str
    args: List[Tuple[str, str, Optional[str]]]  # (name, type, default)
    return_type: Optional[str]
    docstring: Optional[str]
    is_method: bool
    is_private: bool
    complexity: int

@dataclass
class ClassInfo:
    """클래스 정보"""
    name: str
    docstring: Optional[str]
    methods: List[FunctionInfo]
    attributes: List[Tuple[str, str]]  # (name, type)
    base_classes: List[str]

class DocumentationStandardizer:
    """
    학술 표준 문서화 도구

    NumPy 스타일 docstring 생성 및 검증
    """

    def __init__(self, style: str = "numpy"):
        """
        초기화

        Args:
            style: 문서화 스타일 ("numpy" 또는 "google")
        """
        self.style = style
        if style not in ["numpy", "google"]:
            raise ValueError("지원하는 스타일: 'numpy', 'google'")

    def analyze_python_file(self, file_path: str) -> Tuple[List[FunctionInfo], List[ClassInfo]]:
        """
        Python 파일 분석

        Args:
            file_path: 분석할 파일 경로

        Returns:
            함수 정보와 클래스 정보 튜플
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"구문 오류: {e}")
            return [], []

        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self._analyze_function(node)
                functions.append(func_info)
            elif isinstance(node, ast.ClassDef):
                class_info = self._analyze_class(node)
                classes.append(class_info)

        return functions, classes

    def _analyze_function(self, node: ast.FunctionDef) -> FunctionInfo:
        """함수 노드 분석"""
        # 인자 분석
        args = []

        # 일반 인자
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = self._get_type_annotation(arg.annotation)
            args.append((arg_name, arg_type, None))

        # 기본값이 있는 인자
        defaults = node.args.defaults
        if defaults:
            # 뒤에서부터 기본값 적용
            for i, default in enumerate(defaults):
                arg_idx = len(node.args.args) - len(defaults) + i
                if arg_idx < len(args):
                    name, arg_type, _ = args[arg_idx]
                    default_value = self._get_default_value(default)
                    args[arg_idx] = (name, arg_type, default_value)

        # 키워드 전용 인자
        for kw_arg in node.args.kwonlyargs:
            arg_name = kw_arg.arg
            arg_type = self._get_type_annotation(kw_arg.annotation)
            args.append((arg_name, arg_type, None))

        # 반환 타입
        return_type = self._get_type_annotation(node.returns)

        # Docstring
        docstring = ast.get_docstring(node)

        # 복잡도 계산
        complexity = self._calculate_complexity(node)

        return FunctionInfo(
            name=node.name,
            args=args,
            return_type=return_type,
            docstring=docstring,
            is_method=False,  # 추후 클래스 분석에서 설정
            is_private=node.name.startswith('_'),
            complexity=complexity
        )

    def _analyze_class(self, node: ast.ClassDef) -> ClassInfo:
        """클래스 노드 분석"""
        methods = []
        attributes = []

        # 메소드 분석
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                method_info = self._analyze_function(child)
                method_info.is_method = True
                methods.append(method_info)
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                # 타입 어노테이션이 있는 속성
                attr_name = child.target.id
                attr_type = self._get_type_annotation(child.annotation)
                attributes.append((attr_name, attr_type))

        # 베이스 클래스
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(f"{base.value.id}.{base.attr}")

        # Docstring
        docstring = ast.get_docstring(node)

        return ClassInfo(
            name=node.name,
            docstring=docstring,
            methods=methods,
            attributes=attributes,
            base_classes=base_classes
        )

    def _get_type_annotation(self, annotation: Optional[ast.AST]) -> Optional[str]:
        """타입 어노테이션을 문자열로 변환"""
        if annotation is None:
            return None

        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Attribute):
            return f"{annotation.value.id}.{annotation.attr}"
        elif isinstance(annotation, ast.Subscript):
            # List[int], Dict[str, int] 등
            value = self._get_type_annotation(annotation.value)
            slice_value = self._get_type_annotation(annotation.slice)
            return f"{value}[{slice_value}]"
        elif isinstance(annotation, ast.Tuple):
            # Union, Tuple 등
            elements = [self._get_type_annotation(elt) for elt in annotation.elts]
            return f"({', '.join(filter(None, elements))})"
        else:
            # 복잡한 경우 ast.unparse 사용 (Python 3.9+)
            try:
                return ast.unparse(annotation)
            except AttributeError:
                return "Any"

    def _get_default_value(self, default: ast.AST) -> str:
        """기본값을 문자열로 변환"""
        if isinstance(default, ast.Constant):
            if isinstance(default.value, str):
                return f"'{default.value}'"
            return str(default.value)
        elif isinstance(default, ast.Name):
            return default.id
        elif isinstance(default, ast.Attribute):
            return f"{default.value.id}.{default.attr}"
        else:
            try:
                return ast.unparse(default)
            except AttributeError:
                return "..."

    def _calculate_complexity(self, node: ast.AST) -> int:
        """함수의 순환 복잡도 계산"""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def generate_function_docstring(self, func_info: FunctionInfo) -> str:
        """
        함수의 표준 docstring 생성

        Args:
            func_info: 함수 정보

        Returns:
            생성된 docstring
        """
        if self.style == "numpy":
            return self._generate_numpy_function_docstring(func_info)
        else:
            return self._generate_google_function_docstring(func_info)

    def _generate_numpy_function_docstring(self, func_info: FunctionInfo) -> str:
        """NumPy 스타일 함수 docstring 생성"""
        lines = ['"""']

        # 한 줄 요약 (함수명 기반으로 생성)
        summary = self._generate_function_summary(func_info.name)
        lines.append(summary)

        # 상세 설명이 필요한 경우
        if func_info.complexity > 5 or len(func_info.args) > 3:
            lines.append("")
            lines.append("상세 설명이 필요합니다.")

        # Parameters 섹션
        if func_info.args:
            non_self_args = [arg for arg in func_info.args if arg[0] != 'self']
            if non_self_args:
                lines.append("")
                lines.append("Parameters")
                lines.append("----------")

                for arg_name, arg_type, default in non_self_args:
                    if arg_type:
                        if default is not None:
                            lines.append(f"{arg_name} : {arg_type}, optional")
                            lines.append(f"    설명이 필요합니다. 기본값: {default}")
                        else:
                            lines.append(f"{arg_name} : {arg_type}")
                            lines.append(f"    설명이 필요합니다.")
                    else:
                        lines.append(f"{arg_name}")
                        lines.append(f"    타입과 설명이 필요합니다.")

        # Returns 섹션
        if func_info.return_type and func_info.name != "__init__":
            lines.append("")
            lines.append("Returns")
            lines.append("-------")
            lines.append(f"{func_info.return_type}")
            lines.append("    반환값 설명이 필요합니다.")

        # 복잡한 함수의 경우 Examples 섹션 추가
        if func_info.complexity > 7:
            lines.append("")
            lines.append("Examples")
            lines.append("--------")
            lines.append(">>> # 사용 예제가 필요합니다.")
            lines.append(f">>> result = {func_info.name}()")

        lines.append('"""')
        return "\n".join(lines)

    def _generate_google_function_docstring(self, func_info: FunctionInfo) -> str:
        """Google 스타일 함수 docstring 생성"""
        lines = ['"""']

        # 한 줄 요약
        summary = self._generate_function_summary(func_info.name)
        lines.append(summary)

        # Args 섹션
        if func_info.args:
            non_self_args = [arg for arg in func_info.args if arg[0] != 'self']
            if non_self_args:
                lines.append("")
                lines.append("Args:")

                for arg_name, arg_type, default in non_self_args:
                    if arg_type:
                        if default is not None:
                            lines.append(f"    {arg_name} ({arg_type}, optional): 설명이 필요합니다. "
                                        f"기본값: {default}")
                        else:
                            lines.append(f"    {arg_name} ({arg_type}): 설명이 필요합니다.")
                    else:
                        lines.append(f"    {arg_name}: 타입과 설명이 필요합니다.")

        # Returns 섹션
        if func_info.return_type and func_info.name != "__init__":
            lines.append("")
            lines.append("Returns:")
            lines.append(f"    {func_info.return_type}: 반환값 설명이 필요합니다.")

        lines.append('"""')
        return "\n".join(lines)

    def _generate_function_summary(self, func_name: str) -> str:
        """함수명 기반 요약 생성"""
        # 일반적인 패턴 매칭
        if func_name.startswith('get_'):
            return f"{func_name[4:].replace('_', ' ')} 정보를 가져옵니다."
        elif func_name.startswith('set_'):
            return f"{func_name[4:].replace('_', ' ')} 값을 설정합니다."
        elif func_name.startswith('is_'):
            return f"{func_name[3:].replace('_', ' ')} 여부를 확인합니다."
        elif func_name.startswith('create_'):
            return f"{func_name[7:].replace('_', ' ')}을(를) 생성합니다."
        elif func_name.startswith('calculate_'):
            return f"{func_name[10:].replace('_', ' ')}을(를) 계산합니다."
        elif func_name.startswith('generate_'):
            return f"{func_name[9:].replace('_', ' ')}을(를) 생성합니다."
        elif func_name.startswith('validate_'):
            return f"{func_name[9:].replace('_', ' ')}을(를) 검증합니다."
        elif func_name.startswith('__init__'):
            return "클래스를 초기화합니다."
        else:
            return f"{func_name.replace('_', ' ')} 기능을 수행합니다."

    def generate_class_docstring(self, class_info: ClassInfo) -> str:
        """
        클래스의 표준 docstring 생성

        Args:
            class_info: 클래스 정보

        Returns:
            생성된 docstring
        """
        lines = ['"""']

        # 클래스 요약
        summary = f"{class_info.name} 클래스"
        lines.append(summary)
        lines.append("")
        lines.append("상세 설명이 필요합니다.")

        # Attributes 섹션 (공개 속성만)
        public_attrs = [(name, attr_type) for name, attr_type in class_info.attributes
                       if not name.startswith('_')]

        if public_attrs:
            lines.append("")
            if self.style == "numpy":
                lines.append("Attributes")
                lines.append("----------")
                for attr_name, attr_type in public_attrs:
                    lines.append(f"{attr_name} : {attr_type if attr_type else 'Any'}")
                    lines.append(f"    속성 설명이 필요합니다.")
            else:  # Google style
                lines.append("Attributes:")
                for attr_name, attr_type in public_attrs:
                    type_info = f" ({attr_type})" if attr_type else ""
                    lines.append(f"    {attr_name}{type_info}: 속성 설명이 필요합니다.")

        # 주요 메소드 목록 (공개 메소드만)
        public_methods = [method for method in class_info.methods
                         if not method.is_private and method.name != '__init__']

        if public_methods:
            lines.append("")
            lines.append("주요 메소드:")
            for method in public_methods[:5]:  # 최대 5개만 표시
                lines.append(f"    {method.name}: {self._generate_function_summary(method.name)}")

        lines.append('"""')
        return "\n".join(lines)

    def validate_existing_docstring(self, docstring: str, func_info: FunctionInfo) -> List[str]:
        """
        기존 docstring 품질 검증

        Args:
            docstring: 검증할 docstring
            func_info: 함수 정보

        Returns:
            발견된 문제점 목록
        """
        issues = []

        if not docstring:
            issues.append("Docstring이 없습니다.")
            return issues

        lines = docstring.strip().split('\n')

        # 최소 길이 검사
        if len(docstring.strip()) < 10:
            issues.append("Docstring이 너무 짧습니다.")

        # 첫 줄 요약 검사
        if not lines[0].strip():
            issues.append("첫 줄에 요약이 없습니다.")

        # Args/Parameters 섹션 검사
        has_args = any(arg[0] != 'self' for arg in func_info.args)
        if has_args:
            if self.style == "numpy":
                has_params_section = "Parameters" in docstring or "Args" in docstring
            else:
                has_params_section = "Args:" in docstring

            if not has_params_section:
                issues.append("매개변수 설명 섹션이 없습니다.")

        # Returns 섹션 검사
        if func_info.return_type and func_info.name != "__init__":
            if self.style == "numpy":
                has_returns_section = "Returns" in docstring
            else:
                has_returns_section = "Returns:" in docstring

            if not has_returns_section:
                issues.append("반환값 설명 섹션이 없습니다.")

        # 매개변수 문서화 완성도 검사
        if has_args:
            documented_params = set()
            for line in lines:
                # 매개변수 문서화 패턴 찾기
                if ':' in line:
                    param_match = re.match(r'\s*(\w+)\s*[:(]', line)
                    if param_match:
                        documented_params.add(param_match.group(1))

            func_params = {arg[0] for arg in func_info.args if arg[0] != 'self'}
            undocumented = func_params - documented_params

            if undocumented:
                issues.append(f"문서화되지 않은 매개변수: {', '.join(undocumented)}")

        return issues

    def update_file_documentation(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        파일의 모든 문서화를 업데이트

        Args:
            file_path: 업데이트할 파일 경로
            output_path: 출력 파일 경로 (None이면 원본 파일 수정)

        Returns:
            업데이트된 파일 내용
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        functions, classes = self.analyze_python_file(file_path)

        # 문서화가 필요한 함수들 식별
        updates_needed = []

        for func in functions:
            if not func.is_private:  # 공개 함수만
                if not func.docstring:
                    new_docstring = self.generate_function_docstring(func)
                    updates_needed.append((func.name, "function", new_docstring))

        for cls in classes:
            if not cls.docstring:
                new_docstring = self.generate_class_docstring(cls)
                updates_needed.append((cls.name, "class", new_docstring))

        # 실제 파일 업데이트는 AST 기반으로 복잡하므로
        # 여기서는 권고사항만 반환
        report_lines = [f"📚 {file_path} 문서화 업데이트 권고사항:", ""]

        if updates_needed:
            for name, item_type, docstring in updates_needed:
                report_lines.append(f"## {item_type.title()}: {name}")
                report_lines.append("권장 docstring:")
                report_lines.append("```python")
                report_lines.append(docstring)
                report_lines.append("```")
                report_lines.append("")
        else:
            report_lines.append("모든 공개 함수/클래스에 docstring이 있습니다. ✅")

        return "\n".join(report_lines)

    def generate_api_documentation(self, src_dirs: List[str]) -> str:
        """
        API 문서 자동 생성

        Args:
            src_dirs: 소스 디렉토리 목록

        Returns:
            마크다운 형식 API 문서
        """
        doc_lines = ["# API 문서", "", "자동 생성된 API 문서입니다.", ""]

        for src_dir in src_dirs:
            src_path = Path(src_dir)
            if not src_path.exists():
                continue

            python_files = list(src_path.rglob("*.py"))

            for file_path in python_files:
                if file_path.name.startswith('__'):
                    continue

                functions, classes = self.analyze_python_file(str(file_path))

                if functions or classes:
                    module_name = str(file_path.relative_to(src_path)).replace('/', '.').replace('.py', '')
                    doc_lines.append(f"## 모듈: {module_name}")
                    doc_lines.append("")

                    # 클래스 문서화
                    for cls in classes:
                        doc_lines.append(f"### 클래스: {cls.name}")
                        if cls.docstring:
                            doc_lines.append(cls.docstring.split('\n')[0])  # 첫 줄만
                        else:
                            doc_lines.append("설명이 필요합니다.")

                        doc_lines.append("")

                        # 공개 메소드
                        public_methods = [m for m in cls.methods if not m.is_private]
                        if public_methods:
                            doc_lines.append("**주요 메소드:**")
                            for method in public_methods:
                                args_str = ", ".join([f"{arg[0]}: {arg[1] or 'Any'}" for arg in method.args if arg[0] != 'self'])
                                return_str = f" -> {method.return_type}" if method.return_type else ""
                                doc_lines.append(f"- `{method.name}({args_str}){return_str}`")

                            doc_lines.append("")

                    # 독립 함수
                    standalone_functions = [f for f in functions if not f.is_private]
                    if standalone_functions:
                        doc_lines.append("### 함수")
                        for func in standalone_functions:
                            args_str = ", ".join([f"{arg[0]}: {arg[1] or 'Any'}" for arg in func.args])
                            return_str = f" -> {func.return_type}" if func.return_type else ""
                            doc_lines.append(f"- `{func.name}({args_str}){return_str}`")

                            if func.docstring:
                                summary = func.docstring.split('\n')[0]
                                doc_lines.append(f"  {summary}")

                        doc_lines.append("")

        return "\n".join(doc_lines)

def main():
    """테스트 및 예제 실행"""
    print("📚 문서화 표준화 시스템 테스트")
    print("=" * 50)

    # 문서화 표준화기 초기화
    standardizer = DocumentationStandardizer(style="numpy")

    # 테스트 코드 생성
    test_code = '''#!/usr/bin/env python3
def calculate_mean(values, weights=None):
    if weights is None:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = []

    def process_data(self, input_data, normalize=True):
        processed = self._preprocess(input_data)
        if normalize:
            processed = self._normalize(processed)
        return processed

    def _preprocess(self, data):
        return data

    def _normalize(self, data):
        return data
'''

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_file = f.name

    try:
        # 파일 분석
        functions, classes = standardizer.analyze_python_file(temp_file)

        print("\n1. 분석된 함수들:")
        for func in functions:
            print(f"   {func.name}: {len(func.args)}개 인자, 복잡도 {func.complexity}")

        print("\n2. 분석된 클래스들:")
        for cls in classes:
            print(f"   {cls.name}: {len(cls.methods)}개 메소드")

        # Docstring 생성 예제
        if functions:
            print("\n3. 함수 docstring 생성 예제:")
            func = functions[0]
            docstring = standardizer.generate_function_docstring(func)
            print(f"함수: {func.name}")
            print(docstring)

        if classes:
            print("\n4. 클래스 docstring 생성 예제:")
            cls = classes[0]
            docstring = standardizer.generate_class_docstring(cls)
            print(f"클래스: {cls.name}")
            print(docstring)

        # 문서화 업데이트 권고
        print("\n5. 문서화 업데이트 권고:")
        print("-" * 40)
        recommendations = standardizer.update_file_documentation(temp_file)
        print(recommendations)

    finally:
        import os
        os.unlink(temp_file)

if __name__ == "__main__":
    main()