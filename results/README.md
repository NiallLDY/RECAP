# Per-seed results

One row per method (or variant, or hyperparameter value), dataset and seed. All metrics are on the test set, using the checkpoint with the highest validation HR@1. Seeds are 3407, 12050 and 2026; AgentMove needs no training and has one run per dataset (seed 2024). The paper reports the mean over seeds.

| File | Paper |
|---|---|
| `main_results.csv` | Table: performance comparison in HR and NDCG |
| `ablation.csv` | Table: ablation results |
| `tail_unseen.csv` | Table: results on tail and unseen transitions |
| `graph_hops_unvisited.csv` | Appendix table: number of graph hops $N$, column H@20 |
| `graph_hops_candidates.csv` | Appendix table: number of graph hops $N$, columns Avg. cand., Coverage and $10^3\times$SNR |
| `input_window.csv` | Appendix figure: number of trajectory tokens $k$ |
| `lambda.csv` | Figure on revisit and transitive evidence, panel (c): overall HR@1 and HR@20 on unvisited targets under different $\lambda$ |
| `history_use.csv` | Appendix table: how the observed history is used |
| `extra_baselines.csv` | Not in the paper tables: CaST-POI and RepeatNet under the same protocol |
| `graph_hops.csv` | Not in the paper tables: overall HR@1 and HR@20 of RECAP (with the revisit scorer) for each number of graph hops $N$, with $\lambda$ fixed to the value in `main_results.csv` |

The mean and standard deviation over seeds for each file are in `summary/`, except `graph_hops_candidates.csv`.

`lambda` is the weight of the revisit scorer used for that row. In `history_use.csv`, it is empty for the rows that have no revisit scorer.

## Graph hops

`graph_hops_unvisited.csv` is for the backbone with the graph token and without the revisit scorer, as in the appendix table. `unvisited_HR@20` is HR@20 on the test samples whose target the user has not visited before, the same group of samples as in `lambda.csv`, and `unvisited_count` is the number of these samples. A POI counts as visited if it is among the POIs that the revisit scorer reads for the sample, which are the user's 512 most recently visited POIs (`history_cap` in the configs). In CA, one user has visited more than 512 POIs, so one test sample counts as unvisited here although its target is among the user's earlier check-ins (3,969 samples instead of 3,968).

`graph_hops_candidates.csv` has one row per dataset and $N$. It is computed from the transition graph of the training data and the unseen transitions of the validation data, so it does not depend on the seed and has no table in `summary/`. `avg_candidates` is the average number of graph candidates $|C_N(s)|$ over the sources of the unseen validation transitions, `coverage` is the share of unseen validation transitions whose target is among the graph candidates, in percent, and `snr` is the signal-to-noise ratio multiplied by $10^3$. These are the units of the appendix table, which rounds them to 1, 2 and 3 decimal places.

## Summary tables

`summary/` has one table for each file above except `graph_hops_candidates.csv`, with the same file name. Each row is one group of runs that have the same values in every column except `seed` and the metrics, for example `method` and `city`. `n_seeds` is the number of runs in the group. For each metric, `<metric>_mean` is the mean over seeds and `<metric>_std` is the sample standard deviation over seeds, with n-1 in the denominator. Both are rounded to 6 decimal places. When a group has only one run (AgentMove), `<metric>_std` is empty.
