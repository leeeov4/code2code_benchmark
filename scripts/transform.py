# benchmark/scripts/transform.py

import json
import re
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm

from vllm import LLM, vllm, SamplingParams
from transformers import AutoTokenizer

from ..datasets.codenet import CodeNet
from ..core.code_snippet import CodeSnippet



LLM_MODEL_PATH = "Qwen/Qwen2.5-Coder-32B"
MAX_MODEL_LEN = 8192

RETRY_CONFIGS = [
    {"max_tokens_multiplier": 1.5,  "accept_length": False},
    {"max_tokens_multiplier": 2.5,  "accept_length": True},
]


CHAIN = {
    "LLM": "original",
    "R1":  "LLM",
    "R2":  "LLM",
    "R3":  "R1",
}

PROMPTS = {
    "LLM": (
        "Rewrite the following {pl} code by adding only meaningful explanatory comments "
        "and using a cleaner programming style (indentation and whitespace).\n"
        "Do not modify any of the original code lines except for indentation and whitespace.\n"
        "Do not add comments to every single line — only to lines that represent important "
        "logic or steps that need explanation.\n"
        "Only output the modified code with comments, and nothing else.\n"
        "Comments must be in English.\n"
        "Here is the code:\n{source_code}\nHere is the rewritten code:"
    ),
    "R1": (
        "Rewrite the following {pl} code by renaming every function defined and every "
        "variable with dummy placeholders (e.g., func_0, var_0).\n"
        "Do not add or modify comments.\n"
        "Only output the modified code, and nothing else.\n"
        "Here is the code:\n{source_code}\nHere is the rewritten code:"
    ),
    "R2": (
        "Remove all comments (both single-line and multi-line) from the following {pl} code.\n"
        "Only output the modified code, and nothing else.\n"
        "Here is the code:\n{source_code}\nHere is the rewritten code:"
    ),
    "R3": (
        "Remove all comments (both single-line and multi-line) from the following {pl} code.\n"
        "Only output the modified code, and nothing else.\n"
        "Here is the code:\n{source_code}\nHere is the rewritten code:"
    ),
}

# ------------------------------------------------------------------ #
#  Regex source code                                             #
# ------------------------------------------------------------------ #

RE_CODEBLOCK  = re.compile(
    r"```[ \t]*([^\n`]*)\n([\s\S]*?)\n```", re.MULTILINE
)
RE_LANG_HEADER = re.compile(
    r"^([a-zA-Z0-9_+\-#]+)\n([\s\S]+)$", re.MULTILINE
)


def extract_source_code(text: str) -> str:
    text = text.strip()
    m = RE_CODEBLOCK.search(text)
    if m:
        return m.group(2).strip()
    m = RE_LANG_HEADER.match(text)
    if m:
        return m.group(2).strip()
    sys.stderr.write(f"WARNING - Code block not found. Returning raw output.\n")
    return text


# ------------------------------------------------------------------ #
#  Chunking                                                            #
# ------------------------------------------------------------------ #

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ------------------------------------------------------------------ #
#  Transformation                                                      #
# ------------------------------------------------------------------ #

class CodeNetTransformer:

    def __init__(self, language: str, batch_size: int, max_tokens: int):
        self.language   = language
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.dataset    = CodeNet()

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
        self.llm       = LLM(
            model=LLM_MODEL_PATH,
            runner="auto",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=0.8

        )
        self.params = SamplingParams(
            temperature=0,
            max_tokens=max_tokens
        )

    def run(self):
        start = time.time()

        for norm_type, source_version in CHAIN.items():
            print(f"\n[Transform] {norm_type} ← {source_version}")
            self._run_version(norm_type, source_version)

        print(f"\n[Transform] Completed in {time.time() - start:.2f}s")

    def _process(self, snippets: list[CodeSnippet], norm_type: str,
                    accept_length: bool = False, max_tokens_override: int = None) -> tuple[list[CodeSnippet], list[str]]:
    
        params = self.params
        if max_tokens_override is not None:
            params = SamplingParams(
                temperature=0,
                max_tokens=max_tokens_override
            )

        accepted_reasons = {"stop", "length"} if accept_length else {"stop"}

        prompts = [self._build_prompt(s.code, norm_type) for s in snippets]
        outputs = []
        for batch in tqdm(list(chunks(prompts, self.batch_size)), desc=f"{norm_type} inference"):
            outputs.extend(self.llm.generate(batch, params))

        transformed = []
        missing     = []

        for snippet, output in zip(snippets, outputs):
            reason = output.outputs[0].finish_reason
            if reason not in accepted_reasons:
                missing[snippet.id] = reason
                continue
            new_code = extract_source_code(output.outputs[0].text)
            transformed.append(CodeSnippet(
                id=snippet.id,
                code=new_code,
                language=snippet.language
            ))

        return transformed, missing

    def _run_version(self, norm_type: str, source_version: str):
        snippets = self.dataset.load_candidates(self.language, version=source_version)

        out_path     = self.dataset._version_path(self.language, norm_type) / "candidates.json"
        missing_path = self.dataset._version_path(self.language, norm_type) / "missing.json"
        queries_path = self.dataset._queries_path(self.language, norm_type)

        if out_path.exists():
            print(f"[Transform] {norm_type} already exists, skipping.")
            return

        # primo tentativo
        transformed, missing = self._process(snippets, norm_type, accept_length=False)

        # retry con parametri progressivi
        for config in RETRY_CONFIGS:
            if not missing:
                break
            multiplier     = config["max_tokens_multiplier"]
            accept_length  = config["accept_length"]
            print(f"[Transform] Retry with max_tokens×{multiplier}, accept_length={accept_length} "
                f"per {len(missing)} snippet...")
            #missing_snippets = [s for s in snippets if s.id in set(missing)]
            missing_snippets = [s for s in snippets if s.id in missing]  # missing è dict, in cerca le chiavi
            retried, missing = self._process(
                missing_snippets, norm_type,
                accept_length=accept_length,
                max_tokens_override=self.max_tokens * multiplier
            )
            transformed.extend(retried)

        self._save_candidates(transformed, norm_type)
        self._save_missing(missing, norm_type)
        self._save_queries(transformed, norm_type)
        print(f"[Transform] {norm_type}: {len(transformed)} ok, {len(missing)} failed.")


    def _build_prompt(self, code: str, norm_type: str) -> str:
        prompt   = PROMPTS[norm_type].format(pl=self.language, source_code=code)
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # ------------------------------------------------------------------ #
    #  I/O                                                                 #
    # ------------------------------------------------------------------ #

    def _save_candidates(self, transformed: list[CodeSnippet], norm_type: str):
        path = self.dataset._version_path(self.language, norm_type) / "candidates.json"
        self.dataset._save_to_file(transformed, path)

    def _save_missing(self, missing: dict[str, str], norm_type: str):
        path = self.dataset._version_path(self.language, norm_type) / "missing.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(missing, f, indent=2)
        
    def _save_queries(self, transformed: list[CodeSnippet], norm_type: str):
        original_queries   = self.dataset.load_queries(self.language, version="original")
        original_query_ids = {q.id for q in original_queries}
        transformed_map    = {s.id: s for s in transformed}
        queries = [
            transformed_map[qid]
            for qid in original_query_ids
            if qid in transformed_map
        ]
        path = self.dataset._queries_path(self.language, norm_type)
        self.dataset._save_to_file(queries, path)

# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="CodeNet transformation script.")
    parser.add_argument("--language",   required=True, choices=CodeNet.LANGUAGES)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=8192)
    args = parser.parse_args()

    transformer = CodeNetTransformer(
        language=args.language,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens
    )
    transformer.run()


if __name__ == "__main__":
    main()