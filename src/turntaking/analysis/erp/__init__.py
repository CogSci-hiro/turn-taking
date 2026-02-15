"""ERP analysis helpers split into pure computation and I/O orchestration."""

from turntaking.analysis.erp.core import (
    apply_baseline,
    compute_contrast,
    compute_evoked_dataset_result,
    compute_erp_average,
    summarize_erp,
)
from turntaking.analysis.erp.io import (
    load_epochs,
    read_cluster_outputs,
    run_erp_analysis,
    save_erp_results,
    write_cluster_outputs,
)
from turntaking.analysis.erp.outputs import (
    ErpConditionNames,
    get_erp_condition_names,
    write_erp_outputs,
)

__all__ = [
    "compute_erp_average",
    "compute_contrast",
    "compute_evoked_dataset_result",
    "apply_baseline",
    "summarize_erp",
    "ErpConditionNames",
    "get_erp_condition_names",
    "write_erp_outputs",
    "load_epochs",
    "run_erp_analysis",
    "save_erp_results",
    "write_cluster_outputs",
    "read_cluster_outputs",
]
