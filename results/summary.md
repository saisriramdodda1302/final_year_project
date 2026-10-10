# Results

All numbers are means over seeds (± standard deviation across seeds).
Final numbers come from 5 evaluation episodes per seed on a task stream never used in training.

## Main comparison (6 UAVs, 40 devices)

| Method | Time (s) | Energy per UAV (kJ) | Weighted cost | Deadline met | Coverage | Tasks dropped |
| --- | --- | --- | --- | --- | --- | --- |
| AirFed | 0.765 ± 0.071 | 24.21 ± 0.04 | 0.157 ± 0.041 | 100.0% | 96.1% ± 1.6 | 4.5% |
| w/o GAT | 0.814 ± 0.076 | 24.21 ± 0.03 | 0.184 ± 0.044 | 100.0% | 90.9% ± 4.8 | 8.6% |
| w/o reputation | 0.756 ± 0.120 | 24.20 ± 0.05 | 0.151 ± 0.069 | 100.0% | 95.9% ± 1.2 | 3.7% |
| w/o federation | 0.816 ± 0.117 | 24.41 ± 0.15 | 0.189 ± 0.069 | 99.9% | 82.3% ± 8.6 | 17.4% |
| w/o quantization | 0.775 ± 0.093 | 24.20 ± 0.04 | 0.162 ± 0.054 | 100.0% | 90.5% ± 2.7 | 8.8% |
| Independent learner | 0.924 ± 0.033 | 26.00 ± 2.36 | 0.285 ± 0.068 | 100.0% | 81.2% ± 5.2 | 18.8% |
| Static + balanced | 0.670 ± 0.053 | 24.18 ± 0.03 | 0.102 ± 0.031 | 100.0% | 96.7% ± 1.2 | 3.5% |
| Static + local | 0.660 ± 0.049 | 24.17 ± 0.03 | 0.096 ± 0.029 | 100.0% | 96.7% ± 1.2 | 3.5% |
| Random | 0.879 ± 0.012 | 41.85 ± 0.07 | 0.592 ± 0.007 | 100.0% | 75.0% ± 1.8 | 24.9% |

## Federated communication (per UAV, whole training run)

| Method | Data sent (MB) |
| --- | --- |
| AirFed | 245.7 ± 24.7 |
| w/o reputation | 205.1 ± 37.6 |
| w/o quantization | 527.5 ± 111.0 |

Quantization cuts the data AirFed sends by **53.4%**.

## Scalability (weighted cost, evaluation)

| Setup (UAVs-devices) | AirFed | Static + balanced | Random |
| --- | --- | --- | --- |
| 4-40 | 0.193 ± 0.057 | 0.166 ± 0.082 | 0.623 ± 0.060 |
| 6-40 | 0.157 ± 0.041 | 0.102 ± 0.031 | 0.592 ± 0.007 |
| 8-40 | 0.186 ± 0.052 | 0.068 ± 0.061 | 0.729 ± 0.005 |
| 6-20 | 0.081 ± 0.032 | 0.049 ± 0.034 | 0.772 ± 0.013 |
| 6-60 | 0.114 ± 0.032 | 0.083 ± 0.010 | 0.620 ± 0.037 |
