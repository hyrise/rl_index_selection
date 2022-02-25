# SWIRL: Selection of Workload-aware Indexes using Reinforcement Learning

Currently, this repository provides some additional experimental data for the EDBT 2022 paper _SWIRL: Selection of Workload-aware Indexes using Reinforcement Learning_. Soon, this repository will contain the source code for SWIRL, too. Due to open licensing questions, the code cannot be released at this point of time. Please check back soon.

If you have any questions regarding the paper, please contact Jan Kossmann via jan.kossmann@hpi.de


## DRLinda as an RL-based Competitor

This repository will also contain a reimplementation of the reinforcement learning index selection approach DRLinda based on Sadri et al.'s publications shown below. The reimplementation consists of the following classes: `DRLindaActionManager` in *action_manager.py*, `DRLindaObservationManager` in *observation_manager.py*, `DRLindaReward` in *reward_calculator.py*, and a specialized environment in */gym_db/envs/db_env_v3.py*. Results of comparisons with DRLinda are presented in the paper.

We describe our attempt to DRLinda with Lan et al.'s solution to achieve multi-attribute index support in [experiments/drlinda_multi_attribute/](experiments/drlinda_multi_attribute/).


## Referenced publications

- Zahra Sadri, Le Gruenwald, and Eleazar Leal. 2020. DRLindex: deep reinforcement learning index advisor for a cluster database. In Proceedings of the International Database Engineering and Applications Symposium (IDEAS). Pp. 11:1–11:8.
- Zahra Sadri, Le Gruenwald, and Eleazar Leal. 2020. Online Index Selection Using Deep Reinforcement Learning for a Cluster Database. In Proceedings of the International Conference on Data Engineering (ICDE) Workshops. Pp. 158–161.
- Hai Lan, Zhifeng Bao, and Yuwei Peng. 2020. An Index Advisor Using Deep Reinforcement Learning. In Proceedings of the ACM International Conference on Information and Knowledge Management (CIKM). Pp. 2105-2108
