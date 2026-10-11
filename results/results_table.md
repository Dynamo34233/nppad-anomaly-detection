# Benchmark Results: NPPAD Anomaly Detection

### Overall Model Comparison
| Model | Params | ROC-AUC | Detection Rate (%) | False Alarm Rate (/1k) | Mean Delay (steps) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **B1: PCA** | 13,200 | 0.9998 | 100.00% | 9.90 | 1.00 |
| **B2: FC-AE** | 215,042 | 1.0000 | 100.00% | 9.90 | 1.00 |
| **B3: Conv1D-AE** | 29,623 | 1.0000 | 100.00% | 9.90 | 0.99 |
| **B4: LSTM-AE** | 21,655 | 0.9998 | 100.00% | 9.90 | 0.00 |
| **P0: TCN-AE (All)** | 18,399 | 1.0000 | 100.00% | 9.90 | 0.99 |
| **P1: TCN-AE (Selected)** | 14,754 | 0.9992 | 100.00% | 9.90 | 3.83 |
| **P2: Proposed (TCN+Adaptive)** | 14,754 | 0.9992 | 100.00% | 21.90 | 3.00 |

### Performance Breakdown by Severity Bin
| Model | Low Det Rate (%) | Med Det Rate (%) | High Det Rate (%) | Low Delay | Med Delay | High Delay |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| B1: PCA | 100.0% | 100.0% | 100.0% | 1.0 | 1.0 | 1.0 |
| B2: FC-AE | 100.0% | 100.0% | 100.0% | 1.0 | 1.0 | 1.0 |
| B3: Conv1D-AE | 100.0% | 100.0% | 100.0% | 1.0 | 1.0 | 1.0 |
| B4: LSTM-AE | 100.0% | 100.0% | 100.0% | 0.0 | 0.0 | 0.0 |
| P0: TCN-AE (All) | 100.0% | 100.0% | 100.0% | 1.0 | 1.0 | 1.0 |
| P1: TCN-AE (Selected) | 100.0% | 100.0% | 100.0% | 3.9 | 3.8 | 3.8 |
| P2: Proposed (TCN+Adaptive) | 100.0% | 100.0% | 100.0% | 3.0 | 3.0 | 3.0 |