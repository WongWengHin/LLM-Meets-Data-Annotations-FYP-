"""*** 配置错误与运行时错误共用的异常类型定义。 ***"""

from __future__ import annotations


class ExternalAPIError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"External API returned status {status_code}")
        self.status_code = status_code
        self.body = body


class AnnotationValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors
