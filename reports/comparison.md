| Eksperyment | Parametry [mln] | Rozmiar [MB] | Val acc | Test top-1 | Test top-5 | F1 macro | Czas CPU [ms] | Czas GPU [ms] |
|---|---|---|---|---|---|---|---|---|
| resnet50 | 23.53 | 90.0 | 0.984 | 0.992 | 1.000 | 0.992 | 62.97 | 9.62 |
| efficientnet_b0 | 4.02 | 15.6 | 0.996 | 0.980 | 1.000 | 0.980 | 29.51 | 11.64 |
| mobilenet_v2 | 2.24 | 8.7 | 0.972 | 0.968 | 0.996 | 0.968 | 24.43 | 7.22 |
| simple_cnn | 4.72 | 18.0 | 0.688 | 0.716 | 0.964 | 0.723 | 32.91 | 1.47 |

Wybrany model: **resnet50** (najwyższa dokładność top-1 na zbiorze testowym).
