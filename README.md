# SWIRL: Selection of Workload-aware Indexes using Reinforcement Learning

This repository provides some additional experimental data for the [EDBT 2022 paper](https://openproceedings.org/2022/conf/edbt/paper-37.pdf) _SWIRL: Selection of Workload-aware Indexes using Reinforcement Learning_ and the source code for SWIRL. The repository is [licensed](LICENSE) under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0) [license](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

If you have any questions, feel free to contact the authors, e.g., Jan Kossmann via jan.kossmann@hpi.de


## Setup

The provided setup was tested with Python 3.7.9 (due to Tensorflow version dependencies) and PostgreSQL (12.5) only. The presented implementation requires several python libraries that are listed in the requirements.txt. Furthermore, there are two submodules:

1. [StableBaselines v2](https://github.com/Bensk1/stable-baselines/tree/action_mask_453) for the RL algorithms. The submodule includes a [modified version](https://github.com/hill-a/stable-baselines/pull/453) to enable [invalid action masking](https://arxiv.org/abs/2006.14171).
2. The [index selection evaluation platform](https://github.com/hyrise/index_selection_evaluation/tree/rl_index_selection) in a slightly modified version to simplify RL experiments. The platform handles hypothetical indexes and data generation and loading (adding `-O2` to the Makefiles of the tpch-kit and tpcds-kit might speedup this process, see Miscellaneous below).

Please refer to the install script and the [README](https://github.com/hyrise/index_selection_evaluation/blob/rl_index_selection/README.md) of the index selection evaluation platform before proceeding.


### Example workflow and model training

```
git submodule update --init --recursive # Fetch submodules
python3.7 -m venv venv                  # Create virtualenv
source venv/bin/activate                # Activate virtualenv
pip install -r requirements.txt         # Install requirements with pip
python -m swirl experiments/tpch.json   # Run TPC-H example experiment
```

Experiments can be controlled with the (mostly self-explanatory) json-file. There is another example file in the _experiments_ folder. Results will be written into a configurable folder, for the test experiments it is set to _experiment\_results_. If you want to use tensoboard for logging, create the necessary folder: `mkdir tensor_log`.

The index selection evaluation platform allows generating and loading TPC-H and TPC-DS benchmark data. It is recommended to populate a PostgreSQL instance via the platform with the benchmark data before executing the experiments. However, if the requested data (benchmark data and scale factor) is not already present, the experiment should generate and load the data but this functionality is not tested well.

For descriptions of the components and functioning, consult our [EDBT paper](https://openproceedings.org/2022/conf/edbt/paper-37.pdf). Query files were reduced to 10 queries per template for efficiency reasons.


## DRLinda as an RL-based Competitor

This repository will also contain a reimplementation of the reinforcement learning index selection approach DRLinda based on Sadri et al.'s publications shown below. The reimplementation consists of the following classes: `DRLindaActionManager` in *action_manager.py*, `DRLindaObservationManager` in *observation_manager.py*, `DRLindaReward` in *reward_calculator.py*, and a specialized environment in */gym_db/envs/db_env_v3.py*. Results of comparisons with DRLinda are presented in the paper.

We describe our attempt to DRLinda with Lan et al.'s solution to achieve multi-attribute index support in [experiments/drlinda_multi_attribute/](experiments/drlinda_multi_attribute/).


## Referenced publications

- Zahra Sadri, Le Gruenwald, and Eleazar Leal. 2020. DRLindex: deep reinforcement learning index advisor for a cluster database. In Proceedings of the International Database Engineering and Applications Symposium (IDEAS). Pp. 11:1–11:8.
- Zahra Sadri, Le Gruenwald, and Eleazar Leal. 2020. Online Index Selection Using Deep Reinforcement Learning for a Cluster Database. In Proceedings of the International Conference on Data Engineering (ICDE) Workshops. Pp. 158–161.
- Hai Lan, Zhifeng Bao, and Yuwei Peng. 2020. An Index Advisor Using Deep Reinforcement Learning. In Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM). Pp. 2105-2108


## JSON Configuration files
The experiments and models are configured via JSON files. For examples, check the `.json` files in the _experiments_. In the following, we explain the different configuration options:

- `id` (`str`): The name or identifier for the experiment. Output files are named accordingly.
- `description` (`str`): A textual description of the experiment. Just for documentation purposes.
- `result_path` (`str`): The path to store results, i.e., the final training report including performance numbers and the trained model files.
- `gym_version` (`int`): Version of the Index Selection Environment Gym. Typically set to `1`. Change only if you provide an alternative Gym implementation.
- `timesteps` (`int`): The number of time steps until the training finishes. Should be chosen based on the complexity of the task. Large workloads need larger values.
- `random_seed` (`int`): The random seed for the experiment is used for everything that is based on random generators. E.g., it is passed to the StableBaselines model, TensorFlow/Pytorch, and used for workload generation.
- `parallel_environments` (`int`): The number of parallel environments used for training. Greatly impacts training durations __and__ memory usage.
- `action_manager` (`str`): The name of the action manager class to use. For more information consult the paper and `action_manager.py`, which contains all available managers. The paper's experiments use the `MultiColumnIndexActionManager`.
- `observation_manager` (`str`): The name of the action manager class to use. For more information consult the paper and `observation_manager.py`, which contains all available managers. The paper's experiments use the `SingleColumnIndexPlanEmbeddingObservationManagerWithCost` (it supports multi-column indexes, this is only a naming issue).
- `reward_calculator` (`str`): The name of the reward calculation method to use. For more information consult the paper and `reward_calculator.py`, which contains all available reward calculation methods. The paper's experiments use the `RelativeDifferenceRelativeToStorageReward`.
- `max_steps_per_episode` (`int`): The number of maximum admitted index selection steps per episode. This influences the time spent per training episode. The paper's experiments use a value of `200`.
- `validation_frequency` (`int`): The performance of the model under training is evaluated every `n` steps. This setting defines `n`. Only relevant for debugging/monitoring purposes.
- `filter_utilized_columns` (`bool`): __Deprecated, going to be removed soon.__ Set to `false`!
- `max_index_width` (`int`): The maximum number of columns per index. This can largely impact training times as the number of index candidates increases with higher widths.
- `reenable_indexes` (`bool`): Experimental, probably going to be removed. Set to `false`!

To be continued...


## Miscellaneous

For the index_selection_evaluation platform's submodules: Before loading TPC-DS data, build the TPC-DS kit with optimization to speed up table generation:

```
@@ -56,7 +56,7 @@ CC            = $($(OS)_CC)
 # CFLAGS
 AIX_CFLAGS             = -q64 -O3 -D_LARGE_FILES
 HPUX_CFLAGS            = -O3 -Wall
-LINUX_CFLAGS   = -g -Wall
+LINUX_CFLAGS   = -O2 -g -Wall
```

This is similar for the TPC-H kit.
