# Fixed-Basis SVD-DeepONet and STS star_M DeepONet Benchmarks

This repository provides self-contained MATLAB benchmarks for the fixed-basis SVD-DeepONet and the STS star_M DeepONet on four nonlinear PDEs. Each benchmark generates a shared reference dataset, trains both methods under identical data splits, exports their terminal models, and evaluates them on the same test set.

## Requirements

- MATLAB R2024a or newer.
- Parallel Computing Toolbox.
- Deep Learning Toolbox.
- An NVIDIA GPU compatible with MATLAB when `common.use_gpu=true` in `config.m`. Set `common.use_gpu=false` for CPU-only training.

## Quick Start

Open one benchmark folder as the MATLAB current folder, edit `config.m` if needed, and execute:

```matlab
run
```

For example, from the repository root:

```matlab
cd('1d_kdv')
run
```

Use one benchmark folder at a time: each benchmark folder intentionally contain files with the same names. Each folder can also be copied and used on its own.

Configuration can be supplied programmatically without modifying the configuration file:

```matlab
cfg = config();
cfg.common.use_gpu = false;
experiment = run(cfg);
disp(experiment.run_dir)
```

For each individual run, all result files are written to a run-specific directory, `RUNs/yyyyMMdd_HHmm/`, located under the corresponding benchmark folder. The timestamp in the directory name is generated from the computer's local system time.

## Files and Configuration

Each benchmark contains exactly six MATLAB files:

| File | Purpose |
| --- | --- |
| `run.m` | Entry point for each benchmark. Loads the configuration, performs preflight resource checks, orchestrates reference-data generation, network training, and test evaluation, manages the run lifecycle and output directory, and verifies the committed artifacts. |
| `config.m` | Supplies the default settings for each benchmark, matching the configurations used in the paper, and allows any of these settings to be adjusted manually. The configuration is organized into four groups: `common`, `reference_data`, `svd`, and `sts`. The data seed defaults to `reference_data.data_seed=42`, while `training_seed=1000` sets the network-training and optimizer seed for both methods. |
| `reference_data_generator.m` | Generates the shared training, validation, and test reference solutions for the benchmark PDE using the prescribed numerical solver, and optionally saves them as `gt_*.mat` files. |
| `svd_benchmark.m` | Runs the fixed-basis SVD-DeepONet pipeline: builds the global SVD basis from the training snapshots, fits normalization and projection, executes the staged optimization (Adam, output-head refitting, joint L-BFGS, and VP), exports the terminal model, and performs numeric inference. |
| `sts_benchmark.m` | Runs the STS star_M DeepONet pipeline: performs slice-wise spatial truncation, extracts a compact temporal basis, trains the branch network using staged optimization, exports the terminal model, and performs numeric inference. |
| `log_writer.m` | Writes reference-data and method statuses, available evaluation metrics, storage and timing statistics, hardware information, and failure details to `log.txt` in a colon-separated, sectioned format. |

The table below summarizes the training, validation, and test sets, the basis and normalization fit cases, the spatio-temporal output grid, the sensor locations, and the branch architecture for each benchmark. Training, validation, and test share a common output grid, whereas the input sensors are defined on a separate grid.

| Benchmark | Train / validation / test | Basis and normalization fit | Output grid | Sensor locations | Branch / Adam epochs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| KdV | 2000 / 400 / 400 | First 2000 training cases | 256 x 201 | 128 | Three 128-wide tanh layers / 20000 |
| Burgers | 2000 / 400 / 400 | First 2000 training cases | 256 x 100 | 128 | Three 256-wide tanh layers / 20000 |
| Fisher-KPP | 2000 / 400 / 400 | First 2000 training cases | 256 x 100 | 128 | Three 128-wide tanh layers / 20000 |
| Allen-Cahn | 2000 / 200 / 200 | First 1000 training cases | 64 x 64 x 101 | Fixed 16 x 16 | Two convolution/pooling blocks and dense 64 / 10000 |

Every benchmark uses a branch network with a linear skip connection. For Allen-Cahn, training uses all 2000 training cases, while the basis and normalization are fitted on the first 1000 cases only.

All remaining parameters, such as learning-rate schedules, optimizer budgets, and reference-grid settings, are defined in `config.m` and follow the settings reported in the paper. 


## Outputs

| File | Contents | Configuration key (default) |
| --- | --- | --- |
| `gt_train.mat` | Training ground truth `u` and metadata. | `save_gt_train` (false) |
| `gt_validation.mat` | Validation ground truth `u` and metadata. | `save_gt_validation` (false) |
| `gt_test.mat` | Test ground truth `u` and metadata. | `save_gt_test` (true) |
| `svd_model.mat` | Final trained fixed-basis SVD DeepONet model. | `save_svd_model` (true) |
| `sts_model.mat` | Final trained STS star_M DeepONet model. | `save_sts_model` (true) |
| `svd_prediction.mat` | Test-set predictions from the final trained fixed-basis SVD DeepONet model. | `save_svd_prediction` (true) |
| `sts_prediction.mat` | Test-set predictions from the final trained STS star_M DeepONet model. | `save_sts_prediction` (true) |
| `svd_relative_l2_vs_time.mat` | Mean test relative L2 error at each time point. | `save_svd_relative_l2_vs_time` (true) |
| `sts_relative_l2_vs_time.mat` | Mean test relative L2 error at each time point. | `save_sts_relative_l2_vs_time` (true) |
| `log.txt` | Run log. | `save_log` (true) |

All keys above could be manually adjusted in `config.m`. Method-specific outputs additionally require `run_svd_deeponet` or `run_sts_deeponet` to be `true`.

## Reload a Trained Model

From the relevant benchmark folder:

```matlab
result_dir = fullfile('RUNs', 'your-run-folder');
S = load(fullfile(result_dir, 'svd_model.mat'), 'model');
D = load(fullfile(result_dir, 'gt_test.mat'), 'rawsensors');
prediction = svd_benchmark('predict', S.model, D.rawsensors);

T = load(fullfile(result_dir, 'sts_model.mat'), 'model');
sts_prediction = sts_benchmark('predict', T.model, D.rawsensors);
```

## Tested Environments

All scripts were tested locally on Windows with MATLAB R2025b Update 2, an AMD Ryzen AI 7 H 350 CPU with 8 physical cores, and an NVIDIA GeForce RTX 5060 Laptop GPU with CUDA forward compatibility. macOS and Linux execution was not tested.

## License

Copyright (c) 2026 Haojun Yan. This project is licensed under the [MIT License](LICENSE).

MATLAB and the required toolboxes remain subject to their own licenses.
