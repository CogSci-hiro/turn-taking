from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import mne
import numpy as np
import pandas as pd

from turntaking.analysis.utils.epochs import load_epochs, parse_epochs_filepath
from turntaking.analysis.selection import Contrast, SelectionParams, select_epochs, split_epochs_median

Kind = Literal["erp", "tfr"]


@dataclass(frozen=True)
class EvokedDatasetRaw:
    """Raw per-subject epoch groups after selection/splitting/equalization."""

    subject_ids: list[str]
    cond1_epochs: list[np.ndarray]
    cond2_epochs: list[np.ndarray]
    cond1_metadata: list[pd.DataFrame]
    cond2_metadata: list[pd.DataFrame]
    times: np.ndarray
    ch_names: list[str]
    labels: dict[str, str]
    infos: list[mne.Info]


@dataclass(frozen=True)
class EvokedDatasetResult:
    """Container for subject-level ERP/TFR evoked outputs."""

    evokeds_cond_1: list[mne.Evoked]
    evokeds_cond_2: list[mne.Evoked]
    evokeds_difference: list[mne.Evoked]
    evoked_data: np.ndarray
    n_trials: pd.DataFrame
    offsets: pd.DataFrame
    results: Mapping[str, Any]


def _stable_sort_key(path: Path) -> tuple[str, str, str]:
    info = parse_epochs_filepath(path)
    run_key = "" if getattr(info, "run", None) is None else str(info.run)
    task_key = "" if getattr(info, "task", None) is None else str(info.task)
    return task_key, run_key, path.name


def _group_paths_by_subject(epoch_paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in epoch_paths:
        info = parse_epochs_filepath(path)
        grouped[info.subject].append(path)
    for subject, paths in grouped.items():
        grouped[subject] = sorted(paths, key=_stable_sort_key)
    return dict(grouped)


def _load_subject_epochs(paths: list[Path], *, sfreq: float | None) -> mne.BaseEpochs:
    epochs_list = [load_epochs(path) for path in paths]
    epochs = epochs_list[0] if len(epochs_list) == 1 else mne.concatenate_epochs(epochs_list)
    if sfreq is not None:
        epochs = epochs.copy().resample(float(sfreq))
    return epochs


def _split_subject_epochs(
    epochs: mne.BaseEpochs,
    *,
    contrast: Contrast,
    selection_params: SelectionParams,
) -> tuple[mne.BaseEpochs, mne.BaseEpochs, dict[str, str]]:
    selected = select_epochs(epochs, selection_params)
    cond1, cond2, labels = split_epochs_median(selected, contrast=contrast)
    mne.epochs.equalize_epoch_counts([cond1, cond2])
    return cond1, cond2, labels


def _require_metadata(epochs: mne.BaseEpochs) -> pd.DataFrame:
    if epochs.metadata is None:
        raise ValueError("epochs.metadata is required for evoked dataset building.")
    return epochs.metadata.copy()


def _assert_times_match(reference: np.ndarray, current: np.ndarray, *, subject: str) -> None:
    if current.shape != reference.shape or not np.allclose(current, reference, atol=0.0, rtol=0.0):
        raise ValueError(f"Time axis mismatch for subject={subject}.")


def _subject_split_data(
    grouped_paths: dict[str, list[Path]],
    *,
    subject: str,
    contrast: Contrast,
    selection_params: SelectionParams,
    sfreq: float | None,
) -> tuple[mne.BaseEpochs, mne.BaseEpochs, dict[str, str]]:
    epochs = _load_subject_epochs(grouped_paths[subject], sfreq=sfreq)
    return _split_subject_epochs(epochs, contrast=contrast, selection_params=selection_params)


def _resolve_labels(
    labels: dict[str, str] | None,
    split_labels: dict[str, str],
    *,
    subject: str,
) -> dict[str, str]:
    if labels is None:
        return dict(split_labels)
    if labels != split_labels:
        raise ValueError(f"Inconsistent split labels for subject={subject}: {split_labels} vs {labels}")
    return labels


def _update_reference_axes(
    *,
    reference_ch_names: list[str] | None,
    reference_times: np.ndarray | None,
    cond1: mne.BaseEpochs,
    subject: str,
) -> tuple[list[str], np.ndarray]:
    if reference_ch_names is None:
        next_ch_names = list(cond1.ch_names)
    elif list(cond1.ch_names) != reference_ch_names:
        raise ValueError(f"Channel order mismatch for subject={subject}.")
    else:
        next_ch_names = reference_ch_names
    if reference_times is None:
        next_times = cond1.times.copy()
    else:
        _assert_times_match(reference_times, cond1.times, subject=subject)
        next_times = reference_times
    return next_ch_names, next_times


def _append_subject_arrays(
    *,
    cond1: mne.BaseEpochs,
    cond2: mne.BaseEpochs,
    cond1_epochs: list[np.ndarray],
    cond2_epochs: list[np.ndarray],
    cond1_metadata: list[pd.DataFrame],
    cond2_metadata: list[pd.DataFrame],
    infos: list[mne.Info],
) -> None:
    cond1_epochs.append(cond1.get_data(copy=True))
    cond2_epochs.append(cond2.get_data(copy=True))
    cond1_metadata.append(_require_metadata(cond1))
    cond2_metadata.append(_require_metadata(cond2))
    infos.append(cond1.info.copy())


def _collect_subject_payloads(
    *,
    grouped: dict[str, list[Path]],
    subject_ids: list[str],
    contrast: Contrast,
    selection_params: SelectionParams,
    sfreq: float | None,
):
    cond1_epochs: list[np.ndarray] = []
    cond2_epochs: list[np.ndarray] = []
    cond1_metadata: list[pd.DataFrame] = []
    cond2_metadata: list[pd.DataFrame] = []
    infos: list[mne.Info] = []
    labels: dict[str, str] | None = None
    reference_ch_names: list[str] | None = None
    reference_times: np.ndarray | None = None

    for subject in subject_ids:
        cond1, cond2, split_labels = _subject_split_data(
            grouped,
            subject=subject,
            contrast=contrast,
            selection_params=selection_params,
            sfreq=sfreq,
        )
        labels = _resolve_labels(labels, split_labels, subject=subject)
        reference_ch_names, reference_times = _update_reference_axes(
            reference_ch_names=reference_ch_names,
            reference_times=reference_times,
            cond1=cond1,
            subject=subject,
        )
        _append_subject_arrays(
            cond1=cond1,
            cond2=cond2,
            cond1_epochs=cond1_epochs,
            cond2_epochs=cond2_epochs,
            cond1_metadata=cond1_metadata,
            cond2_metadata=cond2_metadata,
            infos=infos,
        )
    return cond1_epochs, cond2_epochs, cond1_metadata, cond2_metadata, infos, labels, reference_ch_names, reference_times


def _result_computer(kind: Kind, contrast: Contrast, band: str | None):
    if kind == "erp":
        from turntaking.analysis.erp.core import compute_evoked_dataset_result

        return lambda raw: compute_evoked_dataset_result(raw, contrast=contrast)
    if kind == "tfr":
        from turntaking.analysis.tfr.core import compute_induced_dataset_result

        return lambda raw: compute_induced_dataset_result(raw, band=str(band), contrast=contrast)
    raise ValueError(f"Unknown kind={kind!r}")


def _assert_partial_axes(
    *,
    partial: EvokedDatasetResult,
    subject: str,
    ch_names_ref: list[str] | None,
    times_ref: np.ndarray | None,
) -> tuple[list[str], np.ndarray]:
    cur_ch_names = list(partial.evokeds_cond_1[0].ch_names)
    cur_times = partial.evokeds_cond_1[0].times.copy()
    if ch_names_ref is not None and cur_ch_names != ch_names_ref:
        raise ValueError(f"Channel order mismatch for subject={subject}.")
    if times_ref is not None and not np.allclose(cur_times, times_ref, atol=0.0, rtol=0.0):
        raise ValueError(f"Time axis mismatch for subject={subject}.")
    return cur_ch_names, cur_times


def _stack_evoked_data(
    evokeds_cond_1: list[mne.Evoked],
    evokeds_cond_2: list[mne.Evoked],
    evokeds_difference: list[mne.Evoked],
) -> np.ndarray:
    return np.stack(
        [
            np.stack([ev.data for ev in evokeds_cond_1], axis=0),
            np.stack([ev.data for ev in evokeds_cond_2], axis=0),
            np.stack([ev.data for ev in evokeds_difference], axis=0),
        ],
        axis=1,
    )


def _build_results_metadata(
    *,
    kind: Kind,
    contrast: Contrast,
    band: str | None,
    cond_1: str,
    cond_2: str,
    subjects: list[str],
    times_ref: np.ndarray,
    ch_names_ref: list[str],
    evoked_data: np.ndarray,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "kind": str(kind),
        "contrast": str(contrast),
        "cond_1": cond_1,
        "cond_2": cond_2,
        "subjects": np.array(subjects, dtype=object),
        "n_subjects": int(len(subjects)),
        "times": times_ref,
        "ch_names": np.array(ch_names_ref, dtype=object),
        "data_shape": np.array(evoked_data.shape, dtype=int),
        "difference_definition": f"{cond_1}-{cond_2}",
    }
    if kind == "tfr":
        metadata["band"] = str(band)
    return metadata


def _compute_partials(
    *,
    grouped: dict[str, list[Path]],
    subjects: list[str],
    contrast: Contrast,
    selection_params: SelectionParams,
    sfreq: float | None,
    compute_result,
) -> tuple[
    list[mne.Evoked],
    list[mne.Evoked],
    list[mne.Evoked],
    list[pd.DataFrame],
    list[pd.DataFrame],
    str | None,
    str | None,
    np.ndarray | None,
    list[str] | None,
]:
    evokeds_cond_1: list[mne.Evoked] = []
    evokeds_cond_2: list[mne.Evoked] = []
    evokeds_difference: list[mne.Evoked] = []
    n_trials_frames: list[pd.DataFrame] = []
    offsets_frames: list[pd.DataFrame] = []
    cond_1: str | None = None
    cond_2: str | None = None
    times_ref: np.ndarray | None = None
    ch_names_ref: list[str] | None = None

    for subject in subjects:
        raw = build_raw_evoked_dataset(
            grouped[subject],
            contrast=contrast,
            selection_params=selection_params,
            sfreq=sfreq,
        )
        partial = compute_result(raw)
        cur_ch_names, cur_times = _assert_partial_axes(
            partial=partial,
            subject=subject,
            ch_names_ref=ch_names_ref,
            times_ref=times_ref,
        )
        if cond_1 is None:
            cond_1 = str(partial.results["cond_1"])
            cond_2 = str(partial.results["cond_2"])
            ch_names_ref = cur_ch_names
            times_ref = cur_times
        evokeds_cond_1.extend(partial.evokeds_cond_1)
        evokeds_cond_2.extend(partial.evokeds_cond_2)
        evokeds_difference.extend(partial.evokeds_difference)
        n_trials_frames.append(partial.n_trials)
        offsets_frames.append(partial.offsets)

    return (
        evokeds_cond_1,
        evokeds_cond_2,
        evokeds_difference,
        n_trials_frames,
        offsets_frames,
        cond_1,
        cond_2,
        times_ref,
        ch_names_ref,
    )


def build_raw_evoked_dataset(
    epoch_paths: list[Path],
    *,
    contrast: Contrast,
    selection_params: SelectionParams,
    sfreq: float | None = None,
) -> EvokedDatasetRaw:
    """Build raw split/equalized per-subject epoch arrays for ERP/TFR cores."""
    if len(epoch_paths) == 0:
        raise ValueError("No epoch files provided.")
    grouped = _group_paths_by_subject(epoch_paths)
    subject_ids = sorted(grouped.keys())
    (
        cond1_epochs,
        cond2_epochs,
        cond1_metadata,
        cond2_metadata,
        infos,
        labels,
        reference_ch_names,
        reference_times,
    ) = _collect_subject_payloads(
        grouped=grouped,
        subject_ids=subject_ids,
        contrast=contrast,
        selection_params=selection_params,
        sfreq=sfreq,
    )
    if labels is None or reference_ch_names is None or reference_times is None:
        raise ValueError("No valid subject data found for evoked dataset building.")
    return EvokedDatasetRaw(
        subject_ids=subject_ids,
        cond1_epochs=cond1_epochs,
        cond2_epochs=cond2_epochs,
        cond1_metadata=cond1_metadata,
        cond2_metadata=cond2_metadata,
        times=reference_times,
        ch_names=reference_ch_names,
        labels=labels,
        infos=infos,
    )


def build_evoked_dataset(
    epoch_paths: list[Path],
    *,
    kind: Kind,
    contrast: Contrast,
    selection_params: SelectionParams,
    band: str | None = None,
    sfreq: float | None = None,
) -> EvokedDatasetResult:
    """
    Backward-compatible API shim:
    dataset selection/grouping here, computation delegated to domain cores.
    """
    if kind == "tfr" and band is None:
        raise ValueError("kind='tfr' requires band=...")
    grouped = _group_paths_by_subject(epoch_paths)
    subjects = sorted(grouped.keys())
    if len(subjects) == 0:
        raise ValueError("No epoch files provided.")
    compute_result = _result_computer(kind, contrast, band)
    (
        evokeds_cond_1,
        evokeds_cond_2,
        evokeds_difference,
        n_trials_frames,
        offsets_frames,
        cond_1,
        cond_2,
        times_ref,
        ch_names_ref,
    ) = _compute_partials(
        grouped=grouped,
        subjects=subjects,
        contrast=contrast,
        selection_params=selection_params,
        sfreq=sfreq,
        compute_result=compute_result,
    )
    if cond_1 is None or cond_2 is None or times_ref is None or ch_names_ref is None:
        raise ValueError("No evokeds computed (maybe selection removed all epochs).")
    evoked_data = _stack_evoked_data(evokeds_cond_1, evokeds_cond_2, evokeds_difference)
    results = _build_results_metadata(
        kind=kind,
        contrast=contrast,
        band=band,
        cond_1=cond_1,
        cond_2=cond_2,
        subjects=subjects,
        times_ref=times_ref,
        ch_names_ref=ch_names_ref,
        evoked_data=evoked_data,
    )
    return EvokedDatasetResult(
        evokeds_cond_1=evokeds_cond_1,
        evokeds_cond_2=evokeds_cond_2,
        evokeds_difference=evokeds_difference,
        evoked_data=evoked_data,
        n_trials=pd.concat(n_trials_frames, ignore_index=True),
        offsets=pd.concat(offsets_frames, ignore_index=True) if offsets_frames else pd.DataFrame(),
        results=results,
    )
