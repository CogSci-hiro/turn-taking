# Project Snapshot

- Generated: `2026-02-13T23:05:55+01:00`

- Root: `/Users/hiro/PycharmProjects/turn-taking-working`

- Config: `/Users/hiro/PycharmProjects/turn-taking-working/workflow/config.yaml`

## Environment

```json
{
  "python": "3.11.8 (v3.11.8:db85d51d3e, Feb  6 2024, 18:02:37) [Clang 13.0.0 (clang-1300.0.29.30)]",
  "executable": "/Users/hiro/PycharmProjects/dcap/.venv/bin/python",
  "platform": "macOS-14.4.1-arm64-arm-64bit",
  "cwd": "/Users/hiro/PycharmProjects/turn-taking-working"
}
```

## Git

- HEAD: `67eefbeceee3fd9115abcf8767f9eebcdceed654`

- Dirty files: `20`

```text
?? .venv/
?? figures/
?? project_snapshot.md
?? src/turntaking/__pycache__/
?? src/turntaking/analysis/__pycache__/
?? src/turntaking/analysis/datasets/__pycache__/
?? src/turntaking/analysis/decoding/__pycache__/
?? src/turntaking/analysis/io/__pycache__/
?? src/turntaking/analysis/mixed_effect/__pycache__/
?? src/turntaking/beh/__pycache__/
?? src/turntaking/cli/__pycache__/
?? src/turntaking/cli/commands/__pycache__/
?? src/turntaking/cli/commands/viz/__pycache__/
?? src/turntaking/config/__pycache__/
?? src/turntaking/stats/__pycache__/
?? src/turntaking/viz/__pycache__/
?? src/turntaking/viz/components/__pycache__/
?? src/turntaking/viz/figures/__pycache__/
?? workflow/results/
?? workflow/templates/
```

## Snakemake Rules

- `all`
- `analyze_erp`
- `beh_turn_table`
- `decoding`
- `decoding_all`
- `erp`
- `fig_behavior`
- `fig_decoding`
- `fig_erp_timecourse`
- `fig_erp_topomap_svg`
- `fig_erp_topos`
- `fig_tfr_topomaps`
- `figures_main`
- `figures_supp`
- `install_turntaking`
- `lmm_fit`
- `mixed_effect`
- `mixed_effect_all`
- `test_decoding`
- `test_decoding_all`
- `test_erp`
- `test_erp_all`
- `test_tfr`
- `test_tfr_all`
- `tfr`

## Key Config Excerpts

- Top-level keys: `['analysis', 'constraints', 'dataset', 'execution', 'io', 'paths', 'viz']`

- Resolved out_dir: `/Volumes/work-4T/turn-taking-new`

### viz

```yaml
viz:
  erp_timecourse:
    duration_long_fif: /Volumes/work-4T/turn-taking-new/erp/duration/long_ave.fif
    duration_short_fif: /Volumes/work-4T/turn-taking-new/erp/duration/short_ave.fif
    latency_fast_fif: /Volumes/work-4T/turn-taking-new/erp/latency/fast_ave.fif
    latency_slow_fif: /Volumes/work-4T/turn-taking-new/erp/latency/slow_ave.fif
    out_base: /Volumes/work-4T/turn-taking-new/figures/main/F_erp_timecourse.tif
    xlim_ms:
    - -1500
    - 500
    ylim_uv:
    - -2.8
    - 1.9
  behavior:
    duration_offsets_csv: /Volumes/work-4T/turn-taking-new/erp/duration/offsets.csv
    latency_offsets_csv: /Volumes/work-4T/turn-taking-new/erp/latency/offsets.csv
    turn_table_csv: /Volumes/work-4T/turn-taking-new/beh/turn_table.csv
    out_base: /Volumes/work-4T/turn-taking-new/figures/F_behavior
    n_bins: 100
  erp_topo:
    duration_cluster_hdf5: /Volumes/work-4T/turn-taking-new/stats/erp/duration/cluster_results.hdf5
    latency_cluster_hdf5: /Volumes/work-4T/turn-taking-new/stats/erp/latency/cluster_results.hdf5
    info_source_fif: /Volumes/work-4T/turn-taking-new/erp/duration/difference_ave.fif
    out_duration: /Volumes/work-4T/turn-taking-new/figures/supp/F_erp_topo_duration.tif
    out_latency: /Volumes/work-4T/turn-taking-new/figures/supp/F_erp_topo_latency.tif
    tmin_s: -2
    tmax_s: 0
    step_ms: 100
    max_cols: 10
    p_threshold: 0.01
  decoding:
    duration_scores_npy: /Volumes/work-4T/turn-taking-new/decoding/erp/duration/scores.npy
    duration_times_npy: /Volumes/work-4T/turn-taking-new/decoding/erp/duration/times.npy
    latency_scores_npy: /Volumes/work-4T/turn-taking-new/decoding/erp/latency/scores.npy
    latency_times_npy: /Volumes/work-4T/turn-taking-new/decoding/erp/latency/times.npy
    duration_cluster_hdf5: /Volumes/work-4T/turn-taking-new/stats/decoding/erp/duration/cluster_results.hdf5
    latency_cluster_hdf5: /Volumes/work-4T/turn-taking-new/stats/decoding/erp/latency/cluster_results.hdf5
    out_base: /Volumes/work-4T/turn-taking-new/figures/main/F_decoding
    p_threshold: 0.05
    figure_profile: jneuro_2col
    ymax: 0.65
    lim: 0.04
  erp_topomaps:
    template_svg: workflow/templates/ERP-timeline.svg
    parts_dir: workflow/results/parts_erp_topomaps
    out_svg: workflow/results/F_erp_topomaps.svg
    info_source_fif: /Volumes/work-4T/turn-taking-new/erp/duration/difference_ave.fif
    duration_cluster_hdf5: /Volumes/work-4T/turn-taking-new/stats/erp/duration/cluster_results.hdf5
    latency_cluster_hdf5: /Volumes/work-4T/turn-taking-new/stats/erp/latency/cluster_results.hdf5
    p_threshold: 0.05
    n_duration_maps: 2
    n_latency_maps: 3

```

## Repository File Summary

Top 117 files by size (excluding .git/.venv/__pycache__/.snakemake):

| file | size |
|---|---:|
| `figures/supp/F_erp_topo_duration.tif` | 5.3 MB |
| `figures/supp/F_erp_topo_latency.tif` | 5.3 MB |
| `figures/supp/F_erp_topo_duration.eps` | 1.8 MB |
| `figures/supp/F_erp_topo_latency.eps` | 1.8 MB |
| `workflow/results/F_erp_topomaps.svg` | 414.1 KB |
| `workflow/results/parts_erp_topomaps/slot_lat_tw3.svg` | 86.6 KB |
| `workflow/results/parts_erp_topomaps/slot_dur_tw1.svg` | 85.0 KB |
| `workflow/results/parts_erp_topomaps/slot_dur_tw2.svg` | 83.6 KB |
| `workflow/results/parts_erp_topomaps/slot_lat_tw1.svg` | 73.4 KB |
| `workflow/results/parts_erp_topomaps/slot_lat_tw2.svg` | 72.1 KB |
| `dev/ERP_Alpha_Behaviour_Analytical_Summary.docx` | 36.8 KB |
| `.idea/workspace.xml` | 25.9 KB |
| `dev/temp/make_figures_viz.py` | 21.6 KB |
| `project_snapshot.md` | 21.5 KB |
| `src/turntaking/viz/figures/erp.py` | 17.3 KB |
| `src/turntaking/config/analysis_schema.py` | 15.8 KB |
| `src/turntaking/cli/commands/viz/topomaps.py` | 15.0 KB |
| `src/turntaking/viz/figures/behavior.py` | 14.9 KB |
| `src/turntaking/viz/svg_pipeline.py` | 13.5 KB |
| `workflow/templates/ERP-timeline.svg` | 13.4 KB |
| `workflow/scripts/fit_lmm.R` | 12.0 KB |
| `project_info.py` | 11.7 KB |
| `src/turntaking/analysis/datasets/evoked_dataset.py` | 11.4 KB |
| `src/turntaking/cli/commands/decoding.py` | 10.5 KB |
| `src/turntaking/viz/figures/tfr.py` | 10.1 KB |
| `src/turntaking/cli/commands/cluster.py` | 9.7 KB |
| `workflow/results/parts_erp_topomaps/colorbar.svg` | 9.0 KB |
| `src/turntaking/cli/commands/erp.py` | 7.1 KB |
| `src/turntaking/analysis/io/tfr.py` | 6.9 KB |
| `src/turntaking/viz/_style.py` | 6.8 KB |
| `dev/projectkit_specs.md` | 6.6 KB |
| `workflow/rules/stats.smk` | 6.5 KB |
| `dev/project_summary.md` | 6.4 KB |
| `src/turntaking/analysis/io/decoding.py` | 6.4 KB |
| `src/turntaking/analysis/io/erp.py` | 6.2 KB |
| `workflow/rules/common.smk` | 6.2 KB |
| `src/turntaking/cli/commands/tfr.py` | 5.8 KB |
| `workflow/rules/viz.smk` | 5.8 KB |
| `src/turntaking/stats/decoding_cluster_test.py` | 5.6 KB |
| `src/turntaking/cli/commands/viz/erp_topo.py` | 5.5 KB |
| `src/turntaking/analysis/mixed_effect/eeg_features.py` | 5.0 KB |
| `src/turntaking/cli/commands/viz/decoding.py` | 4.9 KB |
| `src/turntaking/viz/components/decoding.py` | 4.8 KB |
| `src/turntaking/viz/components/electrodes.py` | 4.8 KB |
| `src/turntaking/analysis/mixed_effect/make_table.py` | 4.7 KB |
| `workflow/config.yaml` | 4.7 KB |
| `src/turntaking/cli/commands/decoding_cluster.py` | 4.1 KB |
| `src/turntaking/analysis/io/cluster.py` | 4.0 KB |
| `workflow/rules/analysis.smk` | 4.0 KB |
| `src/turntaking/analysis/datasets/decoding_dataset.py` | 3.8 KB |
| `src/turntaking/analysis/decoding/run_decoding.py` | 3.7 KB |
| `src/turntaking/viz/figures/decoding.py` | 3.4 KB |
| `.ruff_cache/0.14.14/14069236942626210827` | 3.4 KB |
| `src/turntaking/cli/commands/viz/erp_timecourse.py` | 3.3 KB |
| `src/turntaking/stats/cluster_test.py` | 3.3 KB |
| `.idea/inspectionProfiles/Project_Default.xml` | 3.3 KB |
| `src/turntaking/cli/commands/viz/behavior.py` | 3.2 KB |
| `src/turntaking/cli/main.py` | 3.1 KB |
| `src/turntaking/beh/turn_table.py` | 3.0 KB |
| `src/turntaking/analysis/io/core.py` | 3.0 KB |
| `src/turntaking/viz/_utils.py` | 3.0 KB |
| `src/turntaking/analysis/decoding/dataset.py` | 2.6 KB |
| `src/turntaking/cli/commands/mixed_effect.py` | 2.3 KB |
| `src/turntaking/analysis/io/epochs.py` | 2.3 KB |
| `src/turntaking/analysis/decoding/cache_hdf5.py` | 2.2 KB |
| `src/turntaking/analysis/selection.py` | 2.1 KB |
| `src/turntaking/cli/commands/turn_table.py` | 2.0 KB |
| `src/turntaking/analysis/io/decoding_cluster.py` | 1.7 KB |
| `src/turntaking/stats/cropping.py` | 1.6 KB |
| `workflow/Snakefile` | 1.6 KB |
| `src/turntaking/analysis/mixed_effect/schema.py` | 1.4 KB |
| `dev/remove_future.py` | 956.0 B |
| `src/turntaking/viz/__init__.py` | 938.0 B |
| `workflow/rules/bootstrap.smk` | 764.0 B |
| `src/turntaking/analysis/mixed_effect/constants.py` | 716.0 B |
| `src/turntaking/analysis/features/erp.py` | 685.0 B |
| `.idea/turn-taking-working.iml` | 643.0 B |
| `src/turntaking/config.py` | 602.0 B |
| `workflow/rules/behavior.smk` | 544.0 B |
| `src/turntaking/viz/_io.py` | 497.0 B |
| `src/turntaking/io/paths.py` | 451.0 B |
| `src/turntaking/cli/types.py` | 435.0 B |
| `src/turntaking/analysis/features/roi.py` | 379.0 B |
| `src/turntaking/config/loader.py` | 347.0 B |
| `dev/TODO.md` | 319.0 B |
| `pyproject.toml` | 312.0 B |
| `.idea/misc.xml` | 307.0 B |
| `src/turntaking/cli/commands/decode.py` | 215.0 B |
| `README.md` | 197.0 B |
| `.idea/copilot.data.migration.ask2agent.xml` | 194.0 B |
| `.idea/copilot.data.migration.agent.xml` | 190.0 B |
| `.idea/copilot.data.migration.edit.xml` | 189.0 B |
| `.idea/vcs.xml` | 180.0 B |
| `src/turntaking/config/defaults.yaml` | 179.0 B |
| `.idea/inspectionProfiles/profiles_settings.xml` | 174.0 B |
| `.idea/snakemake-settings.xml` | 128.0 B |
| `src/turntaking/analysis/mixed_effect/__init__.py` | 117.0 B |
| `src/turntaking/viz/figures/__init__.py` | 93.0 B |
| `src/turntaking/analysis/constants.py` | 91.0 B |
| `src/turntaking/viz/components/__init__.py` | 76.0 B |
| `src/turntaking/analysis/__init__.py` | 71.0 B |
| `src/turntaking/analysis/features/__init__.py` | 55.0 B |
| `src/turntaking/cli/__init__.py` | 49.0 B |
| `.idea/.gitignore` | 47.0 B |
| `src/turntaking/analysis/datasets/__init__.py` | 45.0 B |
| `.ruff_cache/CACHEDIR.TAG` | 43.0 B |
| `src/turntaking/analysis/decoding/__init__.py` | 43.0 B |
| `src/turntaking/cli/commands/__init__.py` | 36.0 B |
| `.ruff_cache/.gitignore` | 35.0 B |
| `src/turntaking/__init__.py` | 26.0 B |
| `.gitignore` | 25.0 B |
| `src/turntaking/analysis/io/__init__.py` | 0.0 B |
| `src/turntaking/config/__init__.py` | 0.0 B |
| `src/turntaking/io/__init__.py` | 0.0 B |
| `src/turntaking/cli/commands/viz/__init__.py` | 0.0 B |
| `src/turntaking/beh/__init__.py` | 0.0 B |
| `src/turntaking/stats/__init__.py` | 0.0 B |

## Output Artifact Inventory

out_dir: `/Volumes/work-4T/turn-taking-new`

Listing first 184 files:

| file | size |
|---|---:|
| `beh/._turn_table.csv` | 4.0 KB |
| `beh/turn_table.csv` | 2.1 MB |
| `decoding/erp/duration/scores.npy` | 73.1 MB |
| `decoding/erp/duration/times.npy` | 1.5 KB |
| `decoding/erp/latency/._scores.npy` | 4.0 KB |
| `decoding/erp/latency/._times.npy` | 4.0 KB |
| `decoding/erp/latency/scores.npy` | 73.1 MB |
| `decoding/erp/latency/times.npy` | 1.5 KB |
| `erp/duration/difference_ave.fif` | 10.8 MB |
| `erp/duration/evoked-data.npy` | 64.8 MB |
| `erp/duration/long_ave.fif` | 10.8 MB |
| `erp/duration/metadata.hdf5` | 20.8 KB |
| `erp/duration/n_trials.csv` | 527.0 B |
| `erp/duration/offsets.csv` | 1.2 MB |
| `erp/duration/short_ave.fif` | 10.8 MB |
| `erp/latency/difference_ave.fif` | 10.8 MB |
| `erp/latency/evoked-data.npy` | 64.8 MB |
| `erp/latency/fast_ave.fif` | 10.8 MB |
| `erp/latency/metadata.hdf5` | 20.8 KB |
| `erp/latency/n_trials.csv` | 526.0 B |
| `erp/latency/offsets.csv` | 1.1 MB |
| `erp/latency/slow_ave.fif` | 10.8 MB |
| `figures/main/._F_behavior.tif` | 4.0 KB |
| `figures/main/._F_erp_timecourse.tif` | 4.0 KB |
| `figures/main/._F_erp_topo.tif` | 4.0 KB |
| `figures/main/F_behavior.eps` | 62.3 KB |
| `figures/main/F_behavior.tif` | 7.2 MB |
| `figures/main/F_erp_timecourse.eps` | 626.5 KB |
| `figures/main/F_erp_timecourse.tif` | 11.4 MB |
| `figures/main/F_erp_topo.eps` | 2.3 MB |
| `figures/main/F_erp_topo.tif` | 5.9 MB |
| `figures/supp/F_erp_topo_duration.eps` | 2.4 MB |
| `figures/supp/F_erp_topo_duration.tif` | 4.4 MB |
| `figures/supp/F_erp_topo_latency.eps` | 2.4 MB |
| `figures/supp/F_erp_topo_latency.tif` | 4.4 MB |
| `figures/supp/S1_response_duration_hist.eps` | 116.5 KB |
| `figures/supp/S1_response_duration_hist.tif` | 7.2 MB |
| `figures/supp/S2_previous_speech_duration_hist.eps` | 216.4 KB |
| `figures/supp/S2_previous_speech_duration_hist.tif` | 7.7 MB |
| `figures/supp/S3_fast_joint.eps` | 181.1 KB |
| `figures/supp/S3_fast_joint.tif` | 14.9 MB |
| `figures/supp/S3_long_joint.eps` | 182.7 KB |
| `figures/supp/S3_long_joint.tif` | 14.9 MB |
| `figures/supp/S3_short_joint.eps` | 181.9 KB |
| `figures/supp/S3_short_joint.tif` | 14.9 MB |
| `figures/supp/S3_slow_joint.eps` | 184.1 KB |
| `figures/supp/S3_slow_joint.tif` | 14.9 MB |
| `mixed_effect/lmm/models/.done` | 0.0 B |
| `mixed_effect/lmm/models/latency__tw1_alpha_anterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/latency__tw1_alpha_anterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/latency__tw1_alpha_posterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/latency__tw1_alpha_posterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/latency__tw1_beta_anterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/latency__tw1_beta_anterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/latency__tw1_beta_posterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/latency__tw1_beta_posterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/latency__tw1_mean_anterior__base.rds` | 1.3 MB |
| `mixed_effect/lmm/models/latency__tw1_mean_anterior__full.rds` | 1.5 MB |
| `mixed_effect/lmm/models/latency__tw1_mean_posterior__base.rds` | 1.3 MB |
| `mixed_effect/lmm/models/latency__tw1_mean_posterior__full.rds` | 1.5 MB |
| `mixed_effect/lmm/models/latency__tw2_alpha_anterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/latency__tw2_alpha_anterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/latency__tw2_alpha_posterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/latency__tw2_alpha_posterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/latency__tw2_beta_anterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/latency__tw2_beta_anterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/latency__tw2_beta_posterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/latency__tw2_beta_posterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/latency__tw2_mean_anterior__base.rds` | 1.3 MB |
| `mixed_effect/lmm/models/latency__tw2_mean_anterior__full.rds` | 1.5 MB |
| `mixed_effect/lmm/models/latency__tw2_mean_posterior__base.rds` | 1.3 MB |
| `mixed_effect/lmm/models/latency__tw2_mean_posterior__full.rds` | 1.5 MB |
| `mixed_effect/lmm/models/self_duration__tw1_alpha_anterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/self_duration__tw1_alpha_anterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/self_duration__tw1_alpha_posterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/self_duration__tw1_alpha_posterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/self_duration__tw1_beta_anterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/self_duration__tw1_beta_anterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/self_duration__tw1_beta_posterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/self_duration__tw1_beta_posterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/self_duration__tw1_mean_anterior__base.rds` | 1.3 MB |
| `mixed_effect/lmm/models/self_duration__tw1_mean_anterior__full.rds` | 1.5 MB |
| `mixed_effect/lmm/models/self_duration__tw1_mean_posterior__base.rds` | 1.3 MB |
| `mixed_effect/lmm/models/self_duration__tw1_mean_posterior__full.rds` | 1.5 MB |
| `mixed_effect/lmm/models/self_duration__tw2_alpha_anterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/self_duration__tw2_alpha_anterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/self_duration__tw2_alpha_posterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/self_duration__tw2_alpha_posterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/self_duration__tw2_beta_anterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/self_duration__tw2_beta_anterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/self_duration__tw2_beta_posterior__base.rds` | 1.1 MB |
| `mixed_effect/lmm/models/self_duration__tw2_beta_posterior__full.rds` | 1.3 MB |
| `mixed_effect/lmm/models/self_duration__tw2_mean_anterior__base.rds` | 1.3 MB |
| `mixed_effect/lmm/models/self_duration__tw2_mean_anterior__full.rds` | 1.5 MB |
| `mixed_effect/lmm/models/self_duration__tw2_mean_posterior__base.rds` | 1.3 MB |
| `mixed_effect/lmm/models/self_duration__tw2_mean_posterior__full.rds` | 1.5 MB |
| `mixed_effect/lmm/sessionInfo.txt` | 1.3 KB |
| `mixed_effect/lmm/summaries_full/latency__tw1_alpha_anterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/latency__tw1_alpha_anterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/latency__tw1_alpha_posterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/latency__tw1_alpha_posterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/latency__tw1_beta_anterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/latency__tw1_beta_anterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/latency__tw1_beta_posterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/latency__tw1_beta_posterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/latency__tw1_mean_anterior__base.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/latency__tw1_mean_anterior__full.txt` | 1.6 KB |
| `mixed_effect/lmm/summaries_full/latency__tw1_mean_posterior__base.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/latency__tw1_mean_posterior__full.txt` | 1.6 KB |
| `mixed_effect/lmm/summaries_full/latency__tw2_alpha_anterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/latency__tw2_alpha_anterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/latency__tw2_alpha_posterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/latency__tw2_alpha_posterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/latency__tw2_beta_anterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/latency__tw2_beta_anterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/latency__tw2_beta_posterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/latency__tw2_beta_posterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/latency__tw2_mean_anterior__base.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/latency__tw2_mean_anterior__full.txt` | 1.6 KB |
| `mixed_effect/lmm/summaries_full/latency__tw2_mean_posterior__base.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/latency__tw2_mean_posterior__full.txt` | 1.6 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw1_alpha_anterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw1_alpha_anterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw1_alpha_posterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw1_alpha_posterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw1_beta_anterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw1_beta_anterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw1_beta_posterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw1_beta_posterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw1_mean_anterior__base.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw1_mean_anterior__full.txt` | 1.6 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw1_mean_posterior__base.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw1_mean_posterior__full.txt` | 1.6 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw2_alpha_anterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw2_alpha_anterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw2_alpha_posterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw2_alpha_posterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw2_beta_anterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw2_beta_anterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw2_beta_posterior__base.txt` | 1.2 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw2_beta_posterior__full.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw2_mean_anterior__base.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw2_mean_anterior__full.txt` | 1.6 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw2_mean_posterior__base.txt` | 1.4 KB |
| `mixed_effect/lmm/summaries_full/self_duration__tw2_mean_posterior__full.txt` | 1.6 KB |
| `mixed_effect/lmm/tables/._fixed_effects.csv` | 4.0 KB |
| `mixed_effect/lmm/tables/._lrt_comparisons.csv` | 4.0 KB |
| `mixed_effect/lmm/tables/fixed_effects.csv` | 29.5 KB |
| `mixed_effect/lmm/tables/lrt_comparisons.csv` | 101.0 B |
| `mixed_effect/lmm/tables/models.csv` | 11.8 KB |
| `stats/decoding/erp/duration/cluster_results.hdf5` | 520.8 KB |
| `stats/decoding/erp/duration/cluster_summary.csv` | 9.7 KB |
| `stats/decoding/erp/latency/._cluster_results.hdf5` | 4.0 KB |
| `stats/decoding/erp/latency/._cluster_summary.csv` | 4.0 KB |
| `stats/decoding/erp/latency/cluster_results.hdf5` | 518.2 KB |
| `stats/decoding/erp/latency/cluster_summary.csv` | 10.3 KB |
| `stats/erp/duration/cluster_results.hdf5` | 904.9 KB |
| `stats/erp/duration/cluster_summary.csv` | 299.0 B |
| `stats/erp/latency/cluster_results.hdf5` | 953.3 KB |
| `stats/erp/latency/cluster_summary.csv` | 299.0 B |
| `tfr/duration/alpha/difference_ave.fif` | 10.8 MB |
| `tfr/duration/alpha/induced-data.npy` | 64.8 MB |
| `tfr/duration/alpha/long_ave.fif` | 10.8 MB |
| `tfr/duration/alpha/metadata.hdf5` | 20.8 KB |
| `tfr/duration/alpha/n_trials.csv` | 527.0 B |
| `tfr/duration/alpha/short_ave.fif` | 10.8 MB |
| `tfr/duration/beta/difference_ave.fif` | 10.8 MB |
| `tfr/duration/beta/induced-data.npy` | 64.8 MB |
| `tfr/duration/beta/long_ave.fif` | 10.8 MB |
| `tfr/duration/beta/metadata.hdf5` | 20.8 KB |
| `tfr/duration/beta/n_trials.csv` | 527.0 B |
| `tfr/duration/beta/short_ave.fif` | 10.8 MB |
| `tfr/latency/alpha/difference_ave.fif` | 10.8 MB |
| `tfr/latency/alpha/fast_ave.fif` | 10.8 MB |
| `tfr/latency/alpha/induced-data.npy` | 64.8 MB |
| `tfr/latency/alpha/metadata.hdf5` | 20.8 KB |
| `tfr/latency/alpha/n_trials.csv` | 526.0 B |
| `tfr/latency/alpha/slow_ave.fif` | 10.8 MB |
| `tfr/latency/beta/difference_ave.fif` | 10.8 MB |
| `tfr/latency/beta/fast_ave.fif` | 10.8 MB |
| `tfr/latency/beta/induced-data.npy` | 64.8 MB |
| `tfr/latency/beta/metadata.hdf5` | 20.8 KB |
| `tfr/latency/beta/n_trials.csv` | 526.0 B |
| `tfr/latency/beta/slow_ave.fif` | 10.8 MB |


## pip freeze

```text
asttokens==3.0.1
audioread==3.1.0
certifi==2026.1.4
cffi==2.0.0
charset-normalizer==3.4.4
click==8.3.1
comm==0.2.3
contourpy==1.3.3
cycler==0.12.1
darkdetect==0.8.0
-e git+https://github.com/CogSci-hiro/dcap.git@8e688bd897f52c99eae9af69acbc4b698acdef50#egg=dcap
decorator==5.2.1
executing==2.2.1
fonttools==4.61.1
h5py==3.15.1
idna==3.11
importlib_resources==6.5.2
iniconfig==2.3.0
ipyevents==2.0.4
ipython==9.10.0
ipython_pygments_lexers==1.1.1
ipywidgets==8.1.8
jedi==0.19.2
Jinja2==3.1.6
joblib==1.5.3
jupyterlab_widgets==3.0.16
kiwisolver==1.4.9
lazy_loader==0.4
librosa==0.11.0
llvmlite==0.46.0
lxml==6.0.2
markdown-it-py==4.0.0
MarkupSafe==3.0.3
mat73==0.65
matplotlib==3.10.8
matplotlib-inline==0.2.1
mdurl==0.1.2
meegkit==0.1.9
mne==1.11.0
mne-bids==0.18.0
msgpack==1.1.2
nibabel==5.3.3
numba==0.63.1
numpy==2.3.5
packaging==26.0
pandas==3.0.0
parso==0.8.5
patsy==1.0.2
pexpect==4.9.0
pillow==12.1.0
platformdirs==4.5.1
pluggy==1.6.0
pooch==1.8.2
praat-parselmouth==0.4.7
praatio==6.2.2
prompt_toolkit==3.0.52
ptyprocess==0.7.0
pure_eval==0.2.3
pyarrow==23.0.0
pycparser==3.0
pydab==0.0.0.dev0
pydub==0.25.1
Pygments==2.19.2
pyparsing==3.3.2
PyQt5==5.15.11
PyQt5-Qt5==5.15.18
PyQt5_sip==12.18.0
PyQt6-Qt6==6.10.2
PyQt6_sip==13.11.0
pyriemann==0.10
pytest==9.0.2
python-dateutil==2.9.0.post0
pyvista==0.46.5
pyvistaqt==0.11.3
PyYAML==6.0.3
QtPy==2.4.3
requests==2.32.5
rich==14.3.2
scikit-learn==1.8.0
scipy==1.17.0
scooby==0.11.0
seaborn==0.13.2
shellingham==1.5.4
six==1.17.0
soundfile==0.13.1
soxr==1.0.0
stack-data==0.6.3
statsmodels==0.14.6
tabulate==0.9.0
threadpoolctl==3.6.0
tqdm==4.67.1
traitlets==5.14.3
-e git+https://github.com/CogSci-hiro/turn-taking-working.git@67eefbeceee3fd9115abcf8767f9eebcdceed654#egg=turntaking
typer==0.21.1
typing_extensions==4.15.0
urllib3==2.6.3
vtk==9.5.2
wcwidth==0.5.3
widgetsnbextension==4.0.15
```