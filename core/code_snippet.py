# benchmark/core/code_snippet.py

from dataclasses import dataclass, field

@dataclass
class CodeSnippet:
    id: str
    code: str
    language: str