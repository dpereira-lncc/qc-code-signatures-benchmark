from __future__ import annotations


class SigningFailure(RuntimeError):
    """Bounded signing failure with the number of attempts made."""

    def __init__(self, message: str, attempts: int):
        super().__init__(message)
        self.attempts = attempts
