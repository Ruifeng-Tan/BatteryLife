## Note

All results were reproduced on BatteryLife v4 with two GPUs. The `batch_size` values below are the per-process values passed to `accelerate`; the effective global batch size is therefore `batch_size * 2`.

The hyperparameter search used the following ranges for CPMLP and CPTransformer: actual batch size based on a single GPU [8, 16, 32, 64, 128, 256], learning rate [5e-05, 5e-04, 5e-03, 1e-03], dropout [0, 0.05, 0.1], d_model [32, 64, 128, 256], d_ff [32, 64, 128, 256], e_layers [0-12], and d_layers [0-12].

For CPGRU and CPLSTM, the hyperparameter search used the following ranges: actual batch size based on a single GPU [8, 16, 32, 64, 128, 256], learning rate [5e-05, 5e-04, 5e-03, 1e-03], dropout [0, 0.05, 0.1], d_model [32, 64, 128, 256], d_ff [32, 64, 128, 256], e_layers [0-12], and d_layers [0-12]. 



## Selected Hyperparameters

| model | dataset | seed | batch_size | d_model | d_ff | e_layers | d_layers | dropout | learning_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPMLP | Li-ion | 42 | 16 | 32 | 256 | 12 | 7 | 0 | 5e-05 |
| CPMLP | Li-ion | 2021 | 16 | 32 | 256 | 12 | 7 | 0 | 5e-05 |
| CPMLP | Li-ion | 2024 | 16 | 32 | 256 | 12 | 7 | 0 | 5e-05 |
| CPMLP | Zn-ion | 42 | 64 | 64 | 64 | 5 | 9 | 0.1 | 0.0005 |
| CPMLP | Zn-ion | 2021 | 64 | 64 | 64 | 5 | 9 | 0.1 | 0.0005 |
| CPMLP | Zn-ion | 2024 | 64 | 64 | 64 | 5 | 9 | 0.1 | 0.0005 |
| CPMLP | Na-ion | 42 | 64 | 64 | 32 | 1 | 9 | 0 | 5e-05 |
| CPMLP | Na-ion | 2021 | 64 | 64 | 32 | 1 | 9 | 0 | 5e-05 |
| CPMLP | Na-ion | 2024 | 64 | 64 | 32 | 1 | 9 | 0 | 5e-05 |
| CPMLP | CALB | 42 | 4 | 128 | 128 | 7 | 9 | 0.05 | 5e-05 |
| CPMLP | CALB | 2021 | 8 | 32 | 32 | 12 | 6 | 0.1 | 5e-05 |
| CPMLP | CALB | 2024 | 8 | 256 | 128 | 12 | 6 | 0 | 5e-05 |
| CPTransformer | Li-ion | 42 | 128 | 256 | 64 | 1 | 12 | 0 | 5e-05 |
| CPTransformer | Li-ion | 2021 | 128 | 256 | 64 | 1 | 12 | 0 | 5e-05 |
| CPTransformer | Li-ion | 2024 | 128 | 256 | 64 | 1 | 12 | 0 | 5e-05 |
| CPTransformer | Zn-ion | 42 | 32 | 64 | 128 | 1 | 11 | 0 | 0.001 |
| CPTransformer | Zn-ion | 2021 | 32 | 64 | 128 | 1 | 11 | 0 | 0.001 |
| CPTransformer | Zn-ion | 2024 | 32 | 64 | 128 | 1 | 11 | 0 | 0.001 |
| CPTransformer | Na-ion | 42 | 16 | 128 | 128 | 4 | 9 | 0.1 | 5e-05 |
| CPTransformer | Na-ion | 2021 | 16 | 128 | 128 | 4 | 9 | 0.1 | 5e-05 |
| CPTransformer | Na-ion | 2024 | 16 | 128 | 128 | 4 | 9 | 0.1 | 5e-05 |
| CPTransformer | CALB | 42 | 64 | 256 | 256 | 6 | 7 | 0.05 | 5e-05 |
| CPTransformer | CALB | 2021 | 8 | 64 | 256 | 9 | 9 | 0.1 | 5e-05 |
| CPTransformer | CALB | 2024 | 4 | 128 | 256 | 7 | 6 | 0 | 5e-05 |
| CPGRU | Li-ion | 42 | 8 | 32 | 32 | 9 | 2 | 0.05 | 0.0005 |
| CPGRU | Li-ion | 2021 | 8 | 32 | 32 | 9 | 2 | 0.05 | 0.0005 |
| CPGRU | Li-ion | 2024 | 8 | 32 | 32 | 9 | 2 | 0.05 | 0.0005 |
| CPGRU | Zn-ion | 42 | 32 | 256 | 32 | 12 | 2 | 0.05 | 0.001 |
| CPGRU | Zn-ion | 2021 | 32 | 256 | 32 | 12 | 2 | 0.05 | 0.001 |
| CPGRU | Zn-ion | 2024 | 32 | 256 | 32 | 12 | 2 | 0.05 | 0.001 |
| CPGRU | Na-ion | 42 | 64 | 128 | 32 | 10 | 2 | 0.05 | 0.001 |
| CPGRU | Na-ion | 2021 | 64 | 128 | 32 | 10 | 2 | 0.05 | 0.001 |
| CPGRU | Na-ion | 2024 | 64 | 128 | 32 | 10 | 2 | 0.05 | 0.001 |
| CPGRU | CALB | 42 | 4 | 32 | 256 | 2 | 2 | 0 | 5e-05 |
| CPGRU | CALB | 2021 | 4 | 32 | 256 | 2 | 2 | 0 | 5e-05 |
| CPGRU | CALB | 2024 | 4 | 32 | 256 | 2 | 2 | 0 | 5e-05 |
| CPLSTM | Li-ion | 42 | 8 | 128 | 256 | 10 | 2 | 0.05 | 5e-05 |
| CPLSTM | Li-ion | 2021 | 8 | 128 | 256 | 10 | 2 | 0.05 | 5e-05 |
| CPLSTM | Li-ion | 2024 | 8 | 128 | 256 | 10 | 2 | 0.05 | 5e-05 |
| CPLSTM | Zn-ion | 42 | 64 | 32 | 256 | 2 | 2 | 0 | 0.005 |
| CPLSTM | Zn-ion | 2021 | 64 | 32 | 256 | 2 | 2 | 0 | 0.005 |
| CPLSTM | Zn-ion | 2024 | 64 | 32 | 256 | 2 | 2 | 0 | 0.005 |
| CPLSTM | Na-ion | 42 | 128 | 128 | 256 | 2 | 2 | 0.05 | 5e-05 |
| CPLSTM | Na-ion | 2021 | 128 | 128 | 256 | 2 | 2 | 0.05 | 5e-05 |
| CPLSTM | Na-ion | 2024 | 128 | 128 | 256 | 2 | 2 | 0.05 | 5e-05 |
| CPLSTM | CALB | 42 | 4 | 32 | 32 | 8 | 2 | 0.05 | 5e-05 |
| CPLSTM | CALB | 2021 | 4 | 32 | 32 | 8 | 2 | 0.05 | 5e-05 |
| CPLSTM | CALB | 2024 | 4 | 32 | 32 | 8 | 2 | 0.05 | 5e-05 |
