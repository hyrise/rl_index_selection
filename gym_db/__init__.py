from gym.envs.registration import register

register(id="DB-v1", entry_point="gym_db.envs:DBEnvV1")
# This environment does not make use of invalid actiong masking.
register(id="DB-v2", entry_point="gym_db.envs:DBEnvV2")
# This environment is for reproducing the implementation of the paper:
# DRLindex: Deep Reinforcement Learning Index Advisor for a Cluster Databse by Sadri et al.
# Action masking is only used to prevent repeated indexing actions.
register(id="DB-v3", entry_point="gym_db.envs:DBEnvV3")
