# Training Data Influence Experiments

With the following experiments, we conduct training data influence studies that investigate the impact of different parameters on the agent's performance. The parameters include the number of unknown query templates and, if the number of unknown templates is fixed, how the selection of unknown templates can affect the performance.

## Experiment 1 - Impact of the number of unknown query templates

The following experiment examines how well the agent is able to generalize to unseen workloads depending on the number of unknown query templates during training. The experiment's corresponding JSON files can be found in the [experiment_1](./experiment_1/) folder for reproduction or further investigation.

### Experiment settings

The experiment is conducted with workloads generated from the TPC-H benchmark. We chose this benchmark because its relatively small number of query templates (22 or 19 in accordance with the 3 templates excluded as described in Section 6.1 of the paper) enables us to remove query templates that are essential for generalization easily. The RL algorithm settings are the same as described in the paper, see Section 5 and particularly Table 2.


### Training

For this experiment, we have trained 5 models. For the five models, the number of unknown queries is successively increased. The unknown queries are chosen randomly. All query templates that were unknown to a model are also unknown to the subsequent model.

- The first model is trained on queries based on all 19 query templates, which means that, for this model, there are no unknown query templates during the final evaluation. (At the same time, it is improbable that the model has seen the exact combination of query template and frequency before.)
- The second model is trained on queries that are generated from 16 of the 19 query templates. 3 from 19 or 16% of query templates used for the evaluation are unknown to the model. Unknown query template IDs: [1, 7, 13]
- The third model is trained on queries that are generated from 13 of the 19 query templates. 6 from 19 or 32% of query templates used for the evaluation are unknown to the model. Unknown query template IDs: [1, 7, 13, 21, 4, 16]
- The fourth model is trained on queries that are generated from 10 of the 19 query templates. 9 from 19 or 47% of query templates used for the evaluation are unknown to the model. Unknown query template IDs: [1, 7, 13, 21, 4, 16, 22, 3, 14]
- The fifth model is trained on queries that are generated from 7 of the 19 query templates. 12 from 19 or 63% of query templates used for the evaluation are unknown to the model. Unknown query template IDs: [1, 7, 13, 21, 4, 16, 22, 3, 14, 19, 6, 10]


### Evaluation & Results

For the experiment's evaluation, the performance of all trained agents is evaluated with the same 50 workloads that each consist of queries generated from 5 different query templates, which are drawn at random from the set of all of TPC-H's query templates. The frequencies are also assigned randomly per template and per workload. Budgets are chosen randomly per workload, too.

The following table shows the results of the evaluation of 50 randomly generated workloads as described above. The third and fourth column show the difference of the mean performance over all 50 workloads to the Extend and DB2Advis approaches (for more details on these approaches, consult Section 3.1 of the paper), `+1.2pp` means that our models mean performance is `1.2` percentage points worse than the competitor's performance. The fifth column displays the performance of the model relative to the performance of Model No. 1, i.e., a model that has seen queries generated from all templates during training. The last column shows the cost for executing the workload relative to executing it without any indexes (*RC* in the paper).

| Model No. | # Unknown Queries | Δ to Extend | Δ to DB2Advis | Performance relative to Model No. 1 | Relative cost |
|----------:|------------------:|------------:|--------------:|------------------------------------:|--------------:|
| 1         | 0/19              | +1.2pp      | +0.2pp        | 100%                                | 77.4%         |
| 2         | 3/19              | +3.1pp      | +2.1pp        | 91.6%                               | 79.3%         |
| 3         | 6/19              | +4.2pp      | +3.2pp        | 86.7%                               | 80.4%         |
| 4         | 9/19              | +4.8pp      | +5.8pp        | 84.1%                               | 81.0%         |
| 5         | 12/19             | +9.3pp      | +10.3pp       | 64.2%                               | 85.5%         |

These results demonstrate that with an increasing number of unknown queries, the agent's performance degrades continuously. Since not all queries contribute equally to the agent's learning efforts, the performance differences between the different models are varying. In general, we can observe that if more query templates are unknown, the agent is less capable of generalizing, resulting in degraded performance.

____________________


## Experiment 2 - Impact of the particular selection of unknown queries

With the following experiment, we want to investigate whether - for a sufficiently large workload/set of query templates - the agent's performance strongly depends on a well-selected set of unknown queries during training. The corresponding JSON files can be found in the [experiment_2](./experiment_2/) folder for reproduction or further investigation.


### Experiment settings

The experiment is conducted with workloads generated based on the TPC-DS benchmark. We chose this benchmark because, due to its size (in terms of different query templates), it allows many different exclusions sets and large workloads simultaneously. In contrast to the Join Order Benchmark, the training and evaluation times are significantly shorter. The RL algorithm settings are the same as described in the paper, see Section 5 and particularly Table 2.

For this experiment, we trained 10 agents. The random seeds for both, the selection of the query templates that are unknown during training, and the initialization of the weights of the ANN are different for each of the 10 agents.

### Evaluation & Results 

For the experiment's evaluation, the performance of every trained agent is evaluated with 20 different workloads (which are also different for every agent) that each consist of queries generated from 50 different query templates, which are drawn at random from the set of all of TPC-DS's query templates. Also, it is ensured that 20% of these query templates are drawn from the set of templates that were unknown during the training of the particular agent. The frequencies are also assigned randomly per template and per workload. Budgets are chosen randomly per workload, too.

The following table shows the results for the evaluation of the 10 agents as described above. The third and fourth column show the difference of the mean performance over all 20 workloads to the Extend and DB2Advis approaches (for more details on these approaches, consult Section 3.1 of the paper), `-0.9pp` means that our models mean performance is `0.9` percentage points better than the competitor's performance. The fifth column shows the cost for executing the workload relative to executing it without any indexes (*RC* in the paper). The last column shows the IDs of the query templates that were unknown during training of the respective model.

| Model No. | Random Seed | Δ to Extend | Δ to DB2Advis | Relative cost | IDs of unknown templates |
|----------:|------------:|------------:|--------------:|--------------:|--------------------------|
| 1         | 61          | -0.9pp      | -0.9pp        | 84.3%         |[26, 28, 36, 37, 39, 40, 42, 48, 52, 53, 57, 60, 82, 87]|
| 2         | 62          | +1.8pp      | -0.7pp        | 85.7%         |[12, 20, 21, 29, 52, 56, 58, 62, 63, 67, 68, 82, 87, 92]|
| 3         | 63          | -1.2pp      | -1.1pp        | 84.7%         |[5, 7, 25, 28, 36, 37, 48, 57, 63, 64, 68, 70, 81, 92]|
| 4         | 64          | +1.0pp      | -1.5pp        | 86.0%         |[5, 22, 29, 34, 36, 43, 52, 60, 65, 68, 82, 86, 87, 97]|
| 5         | 65          | -1.0pp      | -2.0pp        | 82.0%         |[7, 12, 21, 25, 36, 37, 38, 43, 48, 50, 53, 55, 57, 69]|
| 6         | 66          | +0.2pp      | -0.1pp        | 83.5%         |[12, 20, 21, 26, 27, 28, 37, 42, 52, 60, 62, 63, 84, 86]|
| 7         | 67          | +2.0pp      | -0.4pp        | 85.5%         |[7, 21, 36, 48, 51, 62, 63, 70, 73, 82, 84, 87, 96, 97]|
| 8         | 68          | +1.3pp      | -0.8pp        | 84.6%         |[3, 12, 20, 25, 29, 33, 36, 40, 43, 50, 69, 70, 82, 87]|
| 9         | 69          | +1.9pp      | -0.3pp        | 82.5%         |[12, 22, 26, 28, 29, 34, 39, 40, 48, 53, 55, 73, 96, 97]|
| 10        | 70          | +2.1pp      | -0.3pp        | 84.0%         |[1, 5, 12, 21, 25, 28, 30, 37, 52, 53, 57, 67, 96, 99]|

The table demonstrates that the performance of the trained agents is consistently good compared to the two competitive approaches Extend and DB2Advis. It appears that the final performance of the agents does not depend significantly on the particular set of query templates that are chosen to be unknown during training.

