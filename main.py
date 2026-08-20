# benchmark/main.py

import argparse
import itertools

from .config import BCB_DB_PATH, SEED, OUTPUT_PATH
from .core.base_dataset import BaseDataset
from .datasets.multiple import MultiPLE
from .datasets.codenet import CodeNet
from .datasets.xcodeeval import xCodeEval
from .datasets.bigclonebench import BigCloneBench
from .pipeline.pipeline import Pipeline
from .models.starencoder import StarEncoder
from .models.codebert import CodeBERT
from .models.codet5 import CodeT5
from .models.codellama import Codellama
from .models.coderank import CodeRank
from .models.codesage import CodeSage
from .models.codex import Codex
from .models.codex_2b import Codex2B
from .models.cotext import CoText
from .models.ibm_granite import IBMGranite
from .models.nomic_embed import NomicEmbed
from .models.graphcodebert import GraphCodeBERT
from .models.plbart import PlBART
from .models.qwen3_coder import Qwen3Coder
from .models.qwen3_emb import Qwen3Embedding
from .models.sptcode import SPTCode
from .models.unixcoder_wrapper import UniXcoderWrapper
from .models.codet5p import CodeT5P
from .models.codet5p_220m import CodeT5P220M

from .analysis.embedding_time import EmbeddingTimeAnalysis

# ------------------------------------------------------------------ #
#  Registry                                                            #
# ------------------------------------------------------------------ #

def build_dataset(args) -> BaseDataset:
    if args.dataset == "codenet":       return CodeNet()
    if args.dataset == "multiple":     return MultiPLE()
    if args.dataset == "xcodeeval":     return xCodeEval()
    if args.dataset == "bigclonebench": return BigCloneBench(clone_type=args.clone_type)
    raise ValueError(f"Dataset sconosciuto: {args.dataset}")


def build_model(args):
    if args.model == "starencoder":
        return StarEncoder()
    if args.model == "unixcoder":
        return UniXcoderWrapper()
    elif args.model == "codebert":
        return CodeBERT()
    elif args.model == "codet5":
        return CodeT5()
    elif args.model == "codet5_large":
        return CodeT5("large")
    elif args.model == "codellama":
        return Codellama()
    elif args.model == "coderank":
        return CodeRank()
    elif args.model == "codesage":
        return CodeSage()
    elif args.model == "codesage_large":
        return CodeSage("large")
    elif args.model == "codex":
        return Codex()
    elif args.model == "codex_2b":
        return Codex2B()
    elif args.model == "cotext":
        return CoText()
    elif args.model == "cotext":
        return CoText()
    elif args.model == "graphcodebert":
        return GraphCodeBERT()
    elif args.model == "ibm_granite_3b":
        return IBMGranite("3b")
    elif args.model == "ibm_granite_8b":
        return IBMGranite("8b")
    elif args.model == "nomic_embed":
        return NomicEmbed()
    elif args.model == "plbart":
        return PlBART()
    elif args.model == "sptcode":
        return SPTCode()
    elif args.model == "qwen3_coder":
        return Qwen3Coder()
    elif args.model == "qwen3_emb_600m":
        return Qwen3Embedding("600m")
    elif args.model == "qwen3_emb_8b":
        return Qwen3Embedding("8b")
    elif args.model == "codet5p":
        return CodeT5P()
    elif args.model == "codet5p_220m":
        return CodeT5P220M()

    raise NotImplementedError(f"Model not registered: {args.model}")


# ------------------------------------------------------------------ #
#  Stage handlers                                                      #
# ------------------------------------------------------------------ #

def run_setup(dataset: BaseDataset, args):
    if isinstance(dataset, BigCloneBench):
        dataset.extract_and_serialize()
        print("[Setup] BigCloneBench extraction completed.")
    else:
        for lang in _resolve_languages(dataset, args):
            dataset.select_queries(lang, seed=SEED)
            print(f"[Setup] Queries selected for {lang}.")
            
def run_transform(dataset: BaseDataset, args):
    from .transformers.transformer_factory import build_transformer
    if args.version is None:
        raise ValueError("--version is mandatory for code rewriting.")
    transformer = build_transformer(args.version)
    for lang in _resolve_languages(dataset, args):
        dataset.transform_and_serialize(lang, args.version, transformer)
        print(f"[Transform] Version {args.version} generated for {lang}.")


def run_embeddings(dataset: BaseDataset, pipeline: Pipeline, args):
    version = args.version or "original"
    for lang in _resolve_languages(dataset, args):
        pipeline.run_stage1_embeddings(lang, version)


def run_retrieval(dataset: BaseDataset, pipeline: Pipeline, args):
    q_versions = _resolve_versions(args.query_version)
    c_versions = _resolve_versions(args.candidate_version)
    for lang in _resolve_languages(dataset, args):
        for q_ver, c_ver in itertools.product(q_versions, c_versions):
            print(f"[Retrieval] {lang} | query={q_ver} candidates={c_ver}")
            pipeline.run_stage2_retrieval(lang, q_ver, c_ver)


def run_metrics(dataset: BaseDataset, pipeline: Pipeline, args):
    q_versions = _resolve_versions(args.query_version)
    c_versions = _resolve_versions(args.candidate_version)
    for lang in _resolve_languages(dataset, args):
        for q_ver, c_ver in itertools.product(q_versions, c_versions):
            print(f"[Metrics] {lang} | query={q_ver} candidates={c_ver}")
            summary = pipeline.run_stage3_metrics(lang, q_ver, c_ver)
            _print_metrics(summary)


# ------------------------------------------------------------------ #
#  Utility                                                             #
# ------------------------------------------------------------------ #

VERSIONS = ["original", "LLM", "R1", "R2", "R3"]


def _resolve_languages(dataset: BaseDataset, args) -> list[str]:
    """Returns the requested single language or all supported languages."""
    if args.language:
        if args.language not in dataset.supported_languages():
            raise ValueError(
                f"{args.language} is not supported by {dataset.__class__.__name__}. "
                f"Supported: {dataset.supported_languages()}"
            )
        return [args.language]
    return dataset.supported_languages()


def _resolve_versions(version_arg: str) -> list[str]:
    """'all' expands to all versions; otherwise, returns the requested version."""
    if version_arg == "all":
        return VERSIONS
    if version_arg not in VERSIONS:
        raise ValueError(f"Unknown version: {version_arg}. Available: {VERSIONS}")
    return [version_arg]


def _print_metrics(summary: dict):
    for metric, value in summary.items():
        print(f"  {metric}: {value:.4f}")


# ------------------------------------------------------------------ #
#  Argparse                                                            #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description="Code-to-code retrieval benchmark.")

    parser.add_argument(
        "--dataset",
        required=False,
        choices=["codenet", "multiple", "xcodeeval", "bigclonebench"]
    )
    parser.add_argument(
        "--stage",
        required=False,
        choices=["setup", "transform", "embeddings", "retrieval", "metrics", "all"]
    )
    parser.add_argument(
        "--model",
        required=False
    )
    parser.add_argument(
        "--language",
        required=False,
        help="If omitted, runs on all languages supported by the dataset."
    )
    parser.add_argument(
        "--version",
        required=False,
        help="Code version (original, LLM, R1, R2, R3). Used for embeddings and transformation."
    )
    parser.add_argument(
        "--query_version",
        default="original",
        choices=VERSIONS + ["all"],
        help="Query version for retrieval and metrics."
    )
    parser.add_argument(
        "--candidate_version",
        default="original",
        choices=VERSIONS + ["all"],
        help="Candidate version for retrieval and metrics."
    )
    parser.add_argument(
        "--clone_type",
        required=False,
        choices=["type1", "type2", "type3"],
        help="Required for BigCloneBench."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=False,
        default=None,
        help="If specified, saves only the top-k candidates for each query."
    )
    parser.add_argument(
        "--analysis",
        choices=["timing"],
        required=False
    )

    args = parser.parse_args()
    if args.analysis is None:
        if args.dataset is None:
            parser.error("--dataset is required unless --analysis is used.")
        if args.stage is None:
            parser.error("--stage is required unless --analysis is used.")
        if args.stage not in ("setup", "transform") and args.model is None:
            parser.error("--model is required for the embeddings, retrieval, and metrics stages.")
        if args.dataset == "bigclonebench" and args.clone_type is None:
            parser.error("--clone_type is required for BigCloneBench.")

    return args


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    if args.analysis == "timing":
        model    = build_model(args)
        analysis = EmbeddingTimeAnalysis(model)
        analysis.run(device=model.device)
        return
    
    if args.dataset is None:
        raise ValueError("--dataset is required.")
    if args.stage is None:
        raise ValueError("--stage is required.")

    dataset = build_dataset(args)

    # setup and transform do not require a model
    if args.stage == "setup":
        run_setup(dataset, args)
        return

    if args.stage == "transform":
        run_transform(dataset, args)
        return

    # All other stages require a model
    if args.model is None:
        raise ValueError("--model è obbligatorio per gli stage embeddings, retrieval, metrics.")

    model = build_model(args)
    pipeline = Pipeline(dataset, model, OUTPUT_PATH, top_k=args.top_k)

    if args.stage == "embeddings":
        run_embeddings(dataset, pipeline, args)

    elif args.stage == "retrieval":
        run_retrieval(dataset, pipeline, args)

    elif args.stage == "metrics":
        run_metrics(dataset, pipeline, args)

    elif args.stage == "all":
        version = args.version or "original"
        for lang in _resolve_languages(dataset, args):
            pipeline.run_stage1_embeddings(lang, version)
        run_retrieval(dataset, pipeline, args)
        run_metrics(dataset, pipeline, args)


if __name__ == "__main__":
    main()