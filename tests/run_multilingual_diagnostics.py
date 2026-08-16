"""
run_multilingual_diagnostics.py -- reproducible CLI runner for the
multilingual retrieval root-cause investigation.

Runs entirely against existing, unmodified production code
(src/retrieval/search.py, src/query/resolver.py via
tests/retrieval_evaluator.py) plus the diagnostic-only compositions in
tests/multilingual_diagnostics.py. Never modifies production files, never
overwrites output/chroma_db, never touches output/competitions/.

Usage:
    python tests/run_multilingual_diagnostics.py --phase baseline --out-dir <dir>
    python tests/run_multilingual_diagnostics.py --phase ablation --out-dir <dir>
    python tests/run_multilingual_diagnostics.py --phase entity --out-dir <dir>
    python tests/run_multilingual_diagnostics.py --phase language-entity --out-dir <dir>
    python tests/run_multilingual_diagnostics.py --phase models --out-dir <dir> --models BAAI/bge-m3
    python tests/run_multilingual_diagnostics.py --phase all --out-dir <dir>

Each phase writes one JSON file to --out-dir and is independently
rerunnable (e.g. to add another candidate embedding model later, use
--phase models --models <new-model-name>).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.retrieval_evaluator import (
    DEFAULT_CANDIDATE_CHUNK_DEPTH,
    DEFAULT_K_VALUES,
    aggregate_by_group,
    aggregate_case_results,
    evaluate_retrieval_method,
    load_retrieval_module,
    reset_retrieval_caches,
    run_retrieval_baseline,
    temporary_chroma_copy,
    validate_ground_truth_and_chunks,
)
from tests.multilingual_retrieval_cases import (
    LANGUAGES,
    ENTITY_SCRIPT_DIAGNOSTIC_VARIANTS,
    build_ground_truth_bundle,
    build_translated_cases,
)
from tests.multilingual_diagnostics import (
    ENTITY_TRANSLITERATIONS,
    TempModelIndex,
    build_case_lookup,
    evaluate_model_dense_method,
    evaluate_raw_rrf_method,
    make_translated_case,
)

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _write(out_dir: Path, name: str, data) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Phase 1/2: baseline reproduction (EN/MSA/EGY x BM25/Dense/Hybrid).
# Dense-only results are directly extractable from this same output --
# Phase 2 requires no separate run, since dense_search() is already called
# directly, with no safeguards, by the existing evaluator's "dense" method.
# ---------------------------------------------------------------------------


def run_baseline_phase(out_dir: Path, methods=("bm25", "dense", "hybrid"),
                        k_values=DEFAULT_K_VALUES, candidate_chunk_depth=DEFAULT_CANDIDATE_CHUNK_DEPTH):
    results = {}
    for language in LANGUAGES:
        print(f"=== baseline: language={language} ===", flush=True)
        bundle = build_ground_truth_bundle(language)
        result = run_retrieval_baseline(
            methods=methods, k_values=k_values, candidate_chunk_depth=candidate_chunk_depth,
            project_root=PROJECT_ROOT, artifact_paths=None, ground_truth=bundle,
        )
        assert result["integrity"]["original_chroma_unchanged"], (
            f"Chroma was modified during {language} baseline run -- aborting."
        )
        results[language] = result

    # Reproduction check: default (unmodified) English ground truth must
    # match the "en" multilingual variant bit-for-bit.
    default_result = run_retrieval_baseline(
        methods=methods, k_values=k_values, candidate_chunk_depth=candidate_chunk_depth,
        project_root=PROJECT_ROOT, artifact_paths=None, ground_truth=None,
    )
    reproduction_ok = {}
    for method in methods:
        default_m = default_result["methods"][method]["overall"]["metrics_by_k"]
        en_m = results["en"]["methods"][method]["overall"]["metrics_by_k"]
        reproduction_ok[method] = (default_m == en_m)

    output = {"results_by_language": results, "reproduction_check": reproduction_ok}
    _write(out_dir, "phase1_baseline.json", output)
    return output


# ---------------------------------------------------------------------------
# Phase 3: Hybrid component ablation -- BM25-only / Dense-only / raw RRF /
# full Hybrid, focused on team/l4/multi groups but run over all 24 cases
# per language for completeness.
# ---------------------------------------------------------------------------


def run_ablation_phase(out_dir: Path, k_values=DEFAULT_K_VALUES,
                        candidate_chunk_depth=DEFAULT_CANDIDATE_CHUNK_DEPTH):
    retrieval_module = load_retrieval_module(PROJECT_ROOT)
    chunks_path = str(Path(PROJECT_ROOT) / "output" / "chunks.json")
    chroma_dir = str(Path(PROJECT_ROOT) / "output" / "chroma_db")

    import hashlib
    chroma_sqlite = Path(chroma_dir) / "chroma.sqlite3"
    before_hash = hashlib.sha256(chroma_sqlite.read_bytes()).hexdigest() if chroma_sqlite.exists() else None

    results = {}
    try:
        with temporary_chroma_copy(chroma_dir, retrieval_module) as tmp_chroma:
            from tests.retrieval_evaluator import LegacyTemporaryChromaArtifactPaths
            tmp_artifact_paths = LegacyTemporaryChromaArtifactPaths(
                project_root=Path(PROJECT_ROOT), chroma_dir_override=Path(tmp_chroma),
            )

            for language in LANGUAGES:
                print(f"=== ablation: language={language} ===", flush=True)
                cases = build_translated_cases(language)
                _, _, document_levels = validate_ground_truth_and_chunks(chunks_path=chunks_path, ground_truth=None)

                reset_retrieval_caches(retrieval_module)
                bm25_only = evaluate_retrieval_method(
                    "bm25", cases, retrieval_module, document_levels, k_values, candidate_chunk_depth,
                    artifact_paths=None,
                )
                reset_retrieval_caches(retrieval_module)
                dense_only = evaluate_retrieval_method(
                    "dense", cases, retrieval_module, document_levels, k_values, candidate_chunk_depth,
                    artifact_paths=tmp_artifact_paths,
                )
                reset_retrieval_caches(retrieval_module)
                raw_rrf = evaluate_raw_rrf_method(
                    cases, document_levels, k_values, candidate_chunk_depth,
                    artifact_paths=tmp_artifact_paths,
                )
                reset_retrieval_caches(retrieval_module)
                full_hybrid = evaluate_retrieval_method(
                    "hybrid", cases, retrieval_module, document_levels, k_values, candidate_chunk_depth,
                    artifact_paths=tmp_artifact_paths,
                )
                reset_retrieval_caches(retrieval_module)

                results[language] = {
                    "bm25_only": {"overall": aggregate_case_results(bm25_only, k_values),
                                  "by_group": aggregate_by_group(bm25_only, k_values), "cases": bm25_only},
                    "dense_only": {"overall": aggregate_case_results(dense_only, k_values),
                                   "by_group": aggregate_by_group(dense_only, k_values), "cases": dense_only},
                    "raw_rrf": {"overall": aggregate_case_results(raw_rrf, k_values),
                                "by_group": aggregate_by_group(raw_rrf, k_values), "cases": raw_rrf},
                    "full_hybrid": {"overall": aggregate_case_results(full_hybrid, k_values),
                                     "by_group": aggregate_by_group(full_hybrid, k_values), "cases": full_hybrid},
                }
    finally:
        reset_retrieval_caches(retrieval_module)

    after_hash = hashlib.sha256(chroma_sqlite.read_bytes()).hexdigest() if chroma_sqlite.exists() else None
    assert before_hash == after_hash, "Chroma was modified during the ablation phase -- aborting."

    _write(out_dir, "phase3_ablation.json", results)
    return results


# ---------------------------------------------------------------------------
# Phase 4: entity-script isolation (expanded minimal pairs).
# ---------------------------------------------------------------------------


def run_entity_phase(out_dir: Path, k_values=DEFAULT_K_VALUES,
                      candidate_chunk_depth=DEFAULT_CANDIDATE_CHUNK_DEPTH, minimal_pairs=None):
    """
    `minimal_pairs`: list of (case_id, latin_query, arabic_query, entity_kind)
    tuples. Each pair MUST share identical relevance truth (same case_id)
    and differ ONLY in entity script.
    """
    retrieval_module = load_retrieval_module(PROJECT_ROOT)
    chunks_path = str(Path(PROJECT_ROOT) / "output" / "chunks.json")
    chroma_dir = str(Path(PROJECT_ROOT) / "output" / "chroma_db")
    _, _, document_levels = validate_ground_truth_and_chunks(chunks_path=chunks_path, ground_truth=None)

    import hashlib
    chroma_sqlite = Path(chroma_dir) / "chroma.sqlite3"
    before_hash = hashlib.sha256(chroma_sqlite.read_bytes()).hexdigest() if chroma_sqlite.exists() else None

    results = []
    try:
        with temporary_chroma_copy(chroma_dir, retrieval_module) as tmp_chroma:
            from tests.retrieval_evaluator import LegacyTemporaryChromaArtifactPaths
            tmp_artifact_paths = LegacyTemporaryChromaArtifactPaths(
                project_root=Path(PROJECT_ROOT), chroma_dir_override=Path(tmp_chroma),
            )
            for case_id, latin_query, arabic_query, entity_kind in minimal_pairs:
                print(f"=== entity pair: {case_id} ({entity_kind}) ===", flush=True)
                latin_case = make_translated_case(case_id, latin_query)
                arabic_case = make_translated_case(case_id, arabic_query)

                reset_retrieval_caches(retrieval_module)
                latin_result = evaluate_retrieval_method(
                    "dense", [latin_case], retrieval_module, document_levels, k_values,
                    candidate_chunk_depth, artifact_paths=tmp_artifact_paths,
                )[0]
                reset_retrieval_caches(retrieval_module)
                arabic_result = evaluate_retrieval_method(
                    "dense", [arabic_case], retrieval_module, document_levels, k_values,
                    candidate_chunk_depth, artifact_paths=tmp_artifact_paths,
                )[0]
                reset_retrieval_caches(retrieval_module)

                results.append({
                    "case_id": case_id, "entity_kind": entity_kind,
                    "latin_query": latin_query, "arabic_query": arabic_query,
                    "latin_dense_result": latin_result, "arabic_dense_result": arabic_result,
                })
    finally:
        reset_retrieval_caches(retrieval_module)

    after_hash = hashlib.sha256(chroma_sqlite.read_bytes()).hexdigest() if chroma_sqlite.exists() else None
    assert before_hash == after_hash, "Chroma was modified during the entity phase -- aborting."

    _write(out_dir, "phase4_entity_script.json", results)
    return results


# ---------------------------------------------------------------------------
# Phase 5: language-vs-entity 5-variant (A-E) diagnostic.
# ---------------------------------------------------------------------------


def run_language_entity_phase(out_dir: Path, variant_cases: dict, k_values=DEFAULT_K_VALUES,
                               candidate_chunk_depth=DEFAULT_CANDIDATE_CHUNK_DEPTH):
    """
    `variant_cases`: {case_id: {"A": query, "B": query, "C": query, "D": query, "E": query}}
    A=EN+Latin, B=MSA+Latin, C=EGY+Latin, D=MSA+Arabic-entity, E=EGY+Arabic-entity.
    """
    retrieval_module = load_retrieval_module(PROJECT_ROOT)
    chunks_path = str(Path(PROJECT_ROOT) / "output" / "chunks.json")
    chroma_dir = str(Path(PROJECT_ROOT) / "output" / "chroma_db")
    _, _, document_levels = validate_ground_truth_and_chunks(chunks_path=chunks_path, ground_truth=None)

    import hashlib
    chroma_sqlite = Path(chroma_dir) / "chroma.sqlite3"
    before_hash = hashlib.sha256(chroma_sqlite.read_bytes()).hexdigest() if chroma_sqlite.exists() else None

    results = {}
    try:
        with temporary_chroma_copy(chroma_dir, retrieval_module) as tmp_chroma:
            from tests.retrieval_evaluator import LegacyTemporaryChromaArtifactPaths
            tmp_artifact_paths = LegacyTemporaryChromaArtifactPaths(
                project_root=Path(PROJECT_ROOT), chroma_dir_override=Path(tmp_chroma),
            )
            for case_id, variants in variant_cases.items():
                print(f"=== language-entity variants: {case_id} ===", flush=True)
                case_results = {}
                for label, query in variants.items():
                    case = make_translated_case(case_id, query)
                    reset_retrieval_caches(retrieval_module)
                    r = evaluate_retrieval_method(
                        "dense", [case], retrieval_module, document_levels, k_values,
                        candidate_chunk_depth, artifact_paths=tmp_artifact_paths,
                    )[0]
                    case_results[label] = {"query": query, "result": r}
                results[case_id] = case_results
    finally:
        reset_retrieval_caches(retrieval_module)

    after_hash = hashlib.sha256(chroma_sqlite.read_bytes()).hexdigest() if chroma_sqlite.exists() else None
    assert before_hash == after_hash, "Chroma was modified during the language-entity phase -- aborting."

    _write(out_dir, "phase5_language_entity.json", results)
    return results


# ---------------------------------------------------------------------------
# Phase 6: embedding model comparison (isolated temp Chroma per model).
# ---------------------------------------------------------------------------


def run_model_phase(out_dir: Path, model_names, k_values=DEFAULT_K_VALUES,
                     candidate_chunk_depth=DEFAULT_CANDIDATE_CHUNK_DEPTH):
    import json as _json
    chunks_path = Path(PROJECT_ROOT) / "output" / "chunks.json"
    with open(chunks_path, encoding="utf-8") as f:
        chunks = _json.load(f)

    _, _, document_levels = validate_ground_truth_and_chunks(
        chunks_path=str(chunks_path), ground_truth=None,
    )

    results = {}
    for model_name in model_names:
        print(f"\n=== embedding model: {model_name} ===", flush=True)
        model_results = {}
        with TempModelIndex(chunks, model_name) as idx:
            for language in LANGUAGES:
                print(f"  language={language}", flush=True)
                cases = build_translated_cases(language)
                case_results = evaluate_model_dense_method(
                    idx, cases, document_levels, k_values, candidate_chunk_depth,
                )
                model_results[language] = {
                    "overall": aggregate_case_results(case_results, k_values),
                    "by_group": aggregate_by_group(case_results, k_values),
                    "cases": case_results,
                }

            # Entity-script diagnostic for this model too.
            entity_results = []
            for v in ENTITY_SCRIPT_DIAGNOSTIC_VARIANTS:
                case = make_translated_case(v.source_case_id, v.query)
                r = evaluate_model_dense_method(idx, [case], document_levels, k_values, candidate_chunk_depth)[0]
                entity_results.append({"case_id": v.source_case_id, "query": v.query, "result": r})
            model_results["entity_script_diagnostic"] = entity_results

        results[model_name] = model_results
        _write(out_dir, f"phase6_model_{model_name.replace('/', '_')}.json", model_results)

    _write(out_dir, "phase6_models_combined.json", results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Multilingual retrieval root-cause diagnostics")
    parser.add_argument("--phase", required=True,
                         choices=["baseline", "ablation", "entity", "language-entity", "models", "all"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", nargs="*", default=["BAAI/bge-m3"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.phase in ("baseline", "all"):
        run_baseline_phase(out_dir)
    if args.phase in ("ablation", "all"):
        run_ablation_phase(out_dir)
    if args.phase in ("entity", "all"):
        from tests.run_phase4_phase5 import PHASE4_PAIRS
        run_entity_phase(out_dir, minimal_pairs=PHASE4_PAIRS)
    if args.phase in ("language-entity", "all"):
        from tests.run_phase4_phase5 import PHASE5_VARIANTS
        run_language_entity_phase(out_dir, variant_cases=PHASE5_VARIANTS)
    if args.phase in ("models", "all"):
        run_model_phase(out_dir, args.models)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
