# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Leonardo Venuta
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause License found in the
# LICENSE.md file in the root directory of this source tree.

# benchmark/core/code_snippet.py

from dataclasses import dataclass, field

@dataclass
class CodeSnippet:
    id: str
    code: str
    language: str