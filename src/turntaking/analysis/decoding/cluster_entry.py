
"""Service entrypoint for decoding cluster statistics."""

import argparse
from typing import Any

from turntaking.analysis.decoding.io import (
    DecodingScorePaths,
    get_decoding_out_dir,
    load_decoding_scores,
    write_decoding_cluster_outputs,
)
from turntaking.stats.decoding_cluster_test import (
    DecodingClusterTestParams,
    make_decoding_cluster_summary,
    run_decoding_cluster_test,
)


def run_decoding_cluster(args: argparse.Namespace, cfg: Any) -> None:
    if args.feature != "erp":
        raise ValueError(f"Only feature='erp' is supported right now; got feature={args.feature!r}.")
    out_dir = cfg.io.out_dir
    decoding_dir = get_decoding_out_dir(out_dir, args.contrast)
    scores, times_s = load_decoding_scores(DecodingScorePaths.from_dir(decoding_dir))
    params = _load_params(cfg)
    t_values, clusters, p_values, h0 = run_decoding_cluster_test(scores=scores, params=params)
    summary_df = make_decoding_cluster_summary(clusters=clusters, p_values=p_values, times_s=times_s)
    write_decoding_cluster_outputs(
        out_dir=out_dir,
        contrast=args.contrast,
        t_values=t_values,
        clusters=clusters,
        p_values=p_values,
        h0=h0,
        summary=summary_df,
    )


def _load_params(cfg: Any) -> DecodingClusterTestParams:
    decoding_cfg = cfg.analysis.decoding
    return DecodingClusterTestParams(
        threshold=decoding_cfg.threshold,
        n_permutations=int(decoding_cfg.n_permutations),
        tail=int(getattr(decoding_cfg, "tail", 1)),
        n_jobs=int(getattr(decoding_cfg, "n_jobs", 1)),
        chance_level=float(getattr(decoding_cfg, "chance_level", 0.5)),
    )
