
"""
ERP I/O orchestration layer.

This module is the domain-level boundary for ERP data flow:
- load epochs from disk into in-memory containers,
- run ERP-oriented orchestration around pure core functions,
- persist ERP artifacts through low-level sink functions.

Pure numerical routines stay in ``turntaking.analysis.erp.core``.
File sink details (FIF/NPY/CSV/HDF5 contract) stay in
``turntaking.analysis.erp.outputs``.
"""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import mne
import numpy as np
import pandas as pd

from turntaking.analysis.erp.core import (
    apply_baseline,
    compute_contrast,
    compute_erp_average,
    summarize_erp,
)
from turntaking.analysis.erp.outputs import (
    ErpConditionNames,
    get_erp_condition_names,
    write_erp_outputs,
)
from turntaking.analysis.selection import SelectionParams, select_epochs, split_epochs_median
from turntaking.analysis.utils.epochs import load_epochs as _load_epochs_from_disk
from turntaking.analysis.types import EpochBundle
from turntaking.analysis.utils.io import save_dataframe_csv, save_hdf5_dataset
from turntaking.stats.cluster_test import ClusterTestResult

__all__ = [
    "ErpConditionNames",
    "get_erp_condition_names",
    "write_erp_outputs",
    "load_epochs",
    "run_erp_analysis",
    "save_erp_results",
    "write_cluster_outputs",
    "read_cluster_outputs",
]


def _resolve_baseline(config: Mapping[str, Any]) -> tuple[float, float] | None:
    return _as_window(config.get("baseline"))


def _resolve_summary_window(config: Mapping[str, Any], times: np.ndarray) -> tuple[float, float]:
    summary_window = _as_window(config.get("summary_window"))
    if summary_window is not None:
        return summary_window
    return float(times[0]), float(times[-1])


def _as_window(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"Expected a 2-item window (tmin, tmax), got: {value!r}")
    return float(value[0]), float(value[1])


def _extract_selection_params(config: Mapping[str, Any]) -> SelectionParams | None:
    candidates: list[Mapping[str, Any]] = []
    if isinstance(config.get("selection"), Mapping):
        candidates.append(config["selection"])
    if isinstance(config.get("constraints"), Mapping):
        candidates.append(config["constraints"])
    candidates.append(config)

    for candidate in candidates:
        if "min_latency" not in candidate or "max_latency" not in candidate:
            continue
        min_self_duration = candidate.get("min_self_duration", candidate.get("min_response_duration"))
        if min_self_duration is None:
            continue
        return SelectionParams(
            min_latency=float(candidate["min_latency"]),
            max_latency=float(candidate["max_latency"]),
            min_self_duration=float(min_self_duration),
        )
    return None


def _resolve_contrast(config: Mapping[str, Any]) -> str:
    raw = config.get("contrast")
    if raw is None and isinstance(config.get("analysis"), Mapping):
        raw = config["analysis"].get("contrast")
    if raw is None:
        raise ValueError("`config` must include `contrast` ('duration' or 'latency').")
    contrast = str(raw)
    if contrast not in {"duration", "latency"}:
        raise ValueError(f"Unknown contrast: {contrast!r}. Expected 'duration' or 'latency'.")
    return contrast


def _subject_from_metadata(metadata: pd.DataFrame, fallback: str) -> str:
    if "subject" in metadata.columns and len(metadata) > 0:
        subject = str(metadata["subject"].iloc[0]).strip()
        if subject:
            return subject
    return fallback


def _select_and_split_epochs(
    bundle: EpochBundle,
    *,
    contrast: str,
    selection_params: SelectionParams | None,
) -> tuple[mne.BaseEpochs, mne.BaseEpochs, dict[str, str]]:
    epochs = bundle.epochs if selection_params is None else select_epochs(bundle.epochs, selection_params)
    cond1, cond2, labels = split_epochs_median(epochs, contrast=contrast)
    mne.epochs.equalize_epoch_counts([cond1, cond2])
    return cond1, cond2, labels


def _compute_erps(
    cond1: mne.BaseEpochs,
    cond2: mne.BaseEpochs,
    *,
    baseline: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cond1_data = cond1.get_data(copy=True)
    cond2_data = cond2.get_data(copy=True)
    cond1_mask = np.ones(cond1_data.shape[0], dtype=bool)
    cond2_mask = np.ones(cond2_data.shape[0], dtype=bool)
    times = cond1.times.copy()

    erp_condition1 = apply_baseline(compute_erp_average(cond1_data, cond1_mask), times, baseline)
    erp_condition2 = apply_baseline(compute_erp_average(cond2_data, cond2_mask), times, baseline)
    contrast_erp = compute_contrast(erp_condition1, erp_condition2)
    return erp_condition1, erp_condition2, contrast_erp, times


def _build_summary_table(
    *,
    erp_condition1: np.ndarray,
    erp_condition2: np.ndarray,
    contrast_erp: np.ndarray,
    times: np.ndarray,
    labels: Mapping[str, str],
    summary_window: tuple[float, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": labels["cond_1"], **summarize_erp(erp_condition1, times, summary_window)},
            {"name": labels["cond_2"], **summarize_erp(erp_condition2, times, summary_window)},
            {"name": "difference", **summarize_erp(contrast_erp, times, summary_window)},
        ]
    )


def _compute_subject_summary(
    erp_condition1: np.ndarray,
    erp_condition2: np.ndarray,
    contrast_erp: np.ndarray,
    times: np.ndarray,
    labels: Mapping[str, str],
    summary_window: tuple[float, float],
) -> pd.DataFrame:
    return _build_summary_table(
        erp_condition1=erp_condition1,
        erp_condition2=erp_condition2,
        contrast_erp=contrast_erp,
        times=times,
        labels=labels,
        summary_window=summary_window,
    )


def _build_trial_tables(
    *,
    cond1: mne.BaseEpochs,
    cond2: mne.BaseEpochs,
    labels: Mapping[str, str],
    subject: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_trials = pd.DataFrame(
        [
            {
                "subject": subject,
                labels["cond_1"]: int(len(cond1)),
                labels["cond_2"]: int(len(cond2)),
            }
        ]
    )

    cond1_offsets = cond1.metadata.copy()
    cond2_offsets = cond2.metadata.copy()
    cond1_offsets["condition"] = labels["cond_1"]
    cond2_offsets["condition"] = labels["cond_2"]
    offsets = pd.concat([cond1_offsets, cond2_offsets], ignore_index=True)
    offsets["subject"] = subject
    return n_trials, offsets


def _build_evokeds(
    *,
    info: mne.Info,
    times: np.ndarray,
    erp_condition1: np.ndarray,
    erp_condition2: np.ndarray,
    contrast_erp: np.ndarray,
    labels: Mapping[str, str],
) -> tuple[mne.Evoked, mne.Evoked, mne.Evoked]:
    ev1 = mne.EvokedArray(
        erp_condition1,
        info=info.copy(),
        tmin=float(times[0]),
        comment=labels["cond_1"],
    )
    ev2 = mne.EvokedArray(
        erp_condition2,
        info=info.copy(),
        tmin=float(times[0]),
        comment=labels["cond_2"],
    )
    evd = mne.EvokedArray(
        contrast_erp,
        info=info.copy(),
        tmin=float(times[0]),
        comment=f"{labels['cond_1']}-{labels['cond_2']}",
    )
    return ev1, ev2, evd


def _build_results_payload(
    *,
    contrast: str,
    labels: Mapping[str, str],
    subject: str,
    times: np.ndarray,
    ch_names: Sequence[str],
    evoked_data_shape: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "kind": "erp",
        "contrast": contrast,
        "cond_1": labels["cond_1"],
        "cond_2": labels["cond_2"],
        "subjects": np.array([subject], dtype=object),
        "n_subjects": 1,
        "times": times,
        "ch_names": np.array(ch_names, dtype=object),
        "data_shape": np.array(evoked_data_shape, dtype=int),
        "difference_definition": f"{labels['cond_1']}-{labels['cond_2']}",
    }


def _persist_single_subject_outputs(
    *,
    save_path: str,
    contrast: str,
    bundle: EpochBundle,
    epoch_path: str,
    cond1: mne.BaseEpochs,
    cond2: mne.BaseEpochs,
    labels: Mapping[str, str],
    erp_condition1: np.ndarray,
    erp_condition2: np.ndarray,
    contrast_erp: np.ndarray,
    times: np.ndarray,
    summary_window: tuple[float, float],
) -> None:
    artifacts = _prepare_single_subject_artifacts(
        contrast=contrast,
        bundle=bundle,
        epoch_path=epoch_path,
        cond1=cond1,
        cond2=cond2,
        labels=labels,
        erp_condition1=erp_condition1,
        erp_condition2=erp_condition2,
        contrast_erp=contrast_erp,
        times=times,
        summary_window=summary_window,
    )
    save_erp_results(save_path, contrast=contrast, overwrite=True, **artifacts)


def _prepare_single_subject_artifacts(
    *,
    contrast: str,
    bundle: EpochBundle,
    epoch_path: str,
    cond1: mne.BaseEpochs,
    cond2: mne.BaseEpochs,
    labels: Mapping[str, str],
    erp_condition1: np.ndarray,
    erp_condition2: np.ndarray,
    contrast_erp: np.ndarray,
    times: np.ndarray,
    summary_window: tuple[float, float],
) -> dict[str, Any]:
    summary = _compute_subject_summary(
        erp_condition1, erp_condition2, contrast_erp, times, labels, summary_window
    )
    subject = _subject_from_metadata(bundle.metadata, fallback=Path(epoch_path).stem)
    n_trials, offsets = _build_trial_tables(cond1=cond1, cond2=cond2, labels=labels, subject=subject)
    evoked_data = np.stack([erp_condition1, erp_condition2, contrast_erp], axis=0)[np.newaxis, :, :, :]
    ev1, ev2, evd = _build_evokeds(
        info=bundle.info,
        times=times,
        erp_condition1=erp_condition1,
        erp_condition2=erp_condition2,
        contrast_erp=contrast_erp,
        labels=labels,
    )
    results = _build_results_payload(
        contrast=contrast,
        labels=labels,
        subject=subject,
        times=times,
        ch_names=ev1.ch_names,
        evoked_data_shape=evoked_data.shape,
    )
    return {
        "evokeds_cond_1": [ev1],
        "evokeds_cond_2": [ev2],
        "evokeds_difference": [evd],
        "evoked_data": evoked_data,
        "n_trials": n_trials,
        "results": results,
        "offsets": offsets,
        "summary": summary,
    }


def load_epochs(path: str) -> EpochBundle:
    """
    Load one epochs file and return an in-memory ``EpochBundle``.

    The returned bundle includes the original MNE object and copied metadata/info
    so downstream ERP orchestration can run without additional disk I/O.
    """
    epochs = _load_epochs_from_disk(Path(path), preload=False)
    if epochs.metadata is None:
        raise ValueError(f"epochs.metadata is required for ERP analysis: {path}")
    metadata = epochs.metadata.copy()
    return EpochBundle(epochs=epochs, metadata=metadata, info=epochs.info.copy())


def run_erp_analysis(
    epoch_path: str,
    config: dict,
    *,
    save_path: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Run single-subject ERP analysis for one epochs file.

    Parameters
    ----------
    epoch_path
        Path to an MNE epochs FIF file.
    config
        ERP analysis configuration mapping. At minimum it must contain
        ``contrast`` (``"duration"`` or ``"latency"``). Selection thresholds may
        be provided under ``selection`` or ``constraints``.
    save_path
        If provided, writes the per-subject ERP outputs into this directory.

    Returns
    -------
    outputs
        Dictionary containing ``erp_condition1``, ``erp_condition2``, ``contrast``,
        and ``times`` arrays.
    """
    bundle = load_epochs(epoch_path)
    contrast = _resolve_contrast(config)
    selection_params = _extract_selection_params(config)
    cond1, cond2, labels = _select_and_split_epochs(
        bundle,
        contrast=contrast,
        selection_params=selection_params,
    )
    erp_condition1, erp_condition2, contrast_erp, times = _compute_erps(
        cond1,
        cond2,
        baseline=_resolve_baseline(config),
    )
    if save_path is not None:
        _persist_single_subject_outputs(
            save_path=save_path,
            contrast=contrast,
            bundle=bundle,
            epoch_path=epoch_path,
            cond1=cond1,
            cond2=cond2,
            labels=labels,
            erp_condition1=erp_condition1,
            erp_condition2=erp_condition2,
            contrast_erp=contrast_erp,
            times=times,
            summary_window=_resolve_summary_window(config, times),
        )
    return {
        "erp_condition1": erp_condition1,
        "erp_condition2": erp_condition2,
        "contrast": contrast_erp,
        "times": times,
    }


def save_erp_results(
    save_path: str | Path,
    *,
    contrast: str,
    evokeds_cond_1: Sequence[mne.Evoked],
    evokeds_cond_2: Sequence[mne.Evoked],
    evokeds_difference: Sequence[mne.Evoked],
    evoked_data: np.ndarray,
    n_trials: pd.DataFrame,
    results: Mapping[str, Any],
    offsets: pd.DataFrame,
    summary: pd.DataFrame | None = None,
    overwrite: bool = True,
) -> None:
    """
    Persist ERP results using the ERP output contract.

    This function delegates fixed artifact writing to
    ``turntaking.analysis.erp.outputs.write_erp_outputs`` and writes the
    optional ERP summary table.
    """
    out_dir = Path(save_path)
    write_erp_outputs(
        out_dir,
        contrast=contrast,
        evokeds_cond_1=evokeds_cond_1,
        evokeds_cond_2=evokeds_cond_2,
        evokeds_difference=evokeds_difference,
        evoked_data=evoked_data,
        n_trials=n_trials,
        results=results,
        offsets=offsets,
        overwrite=overwrite,
    )
    if summary is not None:
        save_dataframe_csv(summary, out_dir / "summary.csv")


def write_cluster_outputs(out_dir: Path, result: ClusterTestResult) -> None:
    """Write ERP cluster permutation test outputs (HDF5 + CSV summary)."""
    out_dir = Path(out_dir)
    payload = _cluster_payload(result)
    save_hdf5_dataset(out_dir / "cluster_results.hdf5", payload)
    save_dataframe_csv(_cluster_summary(result), out_dir / "cluster_summary.csv")


def _cluster_payload(result: ClusterTestResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "t-values": np.asarray(result.t_values, dtype=float),
        "p-values": np.asarray(result.p_values, dtype=float),
        "h0": np.asarray(result.h0, dtype=float),
        "meta/json": np.bytes_(json.dumps(result.metadata, sort_keys=True).encode("utf-8")),
    }
    for idx, cluster in enumerate(result.clusters):
        for dim_i, inds in enumerate(cluster):
            payload[f"clusters/{dim_i}-{idx}"] = np.asarray(inds, dtype=int)
    return payload


def _cluster_summary(result: ClusterTestResult) -> pd.DataFrame:
    p_values = np.asarray(result.p_values, dtype=float)
    return pd.DataFrame(
        [
            {
                **result.metadata,
                "n_clusters": int(p_values.size),
                "min_p": float(np.min(p_values)) if p_values.size else float("nan"),
                "n_p_lt_0_05": int(np.sum(p_values < 0.05)) if p_values.size else 0,
            }
        ]
    )


def read_cluster_outputs(path: Path) -> ClusterTestResult:
    """Read ERP cluster permutation outputs from ``cluster_results.hdf5``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cluster results not found: {path}")
    try:
        import h5py
    except Exception as exc:  # noqa: BLE001
        raise ImportError("h5py is required to read cluster_results.hdf5.") from exc
    with h5py.File(path, "r") as handle:
        t_values = np.asarray(handle["t-values"], dtype=float)
        p_values = np.asarray(handle["p-values"], dtype=float)
        h0 = np.asarray(handle["h0"], dtype=float)
        metadata = _read_cluster_metadata(handle)
        clusters = _read_cluster_index_groups(handle)
    return ClusterTestResult(t_values=t_values, p_values=p_values, h0=h0, clusters=clusters, metadata=metadata)


def _read_cluster_metadata(handle: Any) -> dict[str, Any]:
    if "meta/json" not in handle:
        return {}
    raw = handle["meta/json"][()]
    text = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else bytes(raw).decode("utf-8")
    return json.loads(text)


def _read_cluster_index_groups(handle: Any) -> list[tuple[np.ndarray, ...]]:
    group = handle.get("clusters", None)
    if group is None:
        return []
    by_cluster: dict[int, dict[int, np.ndarray]] = {}
    for name in group.keys():
        dim_str, idx_str = name.split("-", 1)
        cluster_idx = int(idx_str)
        dim_idx = int(dim_str)
        by_cluster.setdefault(cluster_idx, {})[dim_idx] = np.asarray(group[name], dtype=int)
    clusters: list[tuple[np.ndarray, ...]] = []
    for cluster_idx in sorted(by_cluster.keys()):
        dims = by_cluster[cluster_idx]
        max_dim = max(dims.keys()) if dims else -1
        clusters.append(tuple(dims[dim_idx] for dim_idx in range(max_dim + 1)))
    return clusters
