import json
import logging


class ConfigurationParser(object):
    def __init__(self, configuration_file):
        self.REQUIRED_CONFIGURATION_OPTIONS_FIRST_LEVEL = [
            "id",
            "description",
            "rl_algorithm",
            "workload",
            "gym_version",
            "random_seed",
            "budgets",
            "max_steps_per_episode",
            "action_manager",
            "observation_manager",
            "reward_calculator",
            "timesteps",
            "parallel_environments",
            "validation_frequency",
            "comparison_algorithms",
            "filter_utilized_columns",
            "max_index_width",
            "result_path",
            "reenable_indexes",
            "pickle_cost_estimation_caches",
        ]

        self.REQUIRED_CONFIGURATION_OPTIONS_FURTHER = {
            "workload": [
                "benchmark",
                "training_instances",
                "validation_testing",
                "size",
                "excluded_query_classes",
                "scale_factor",
                "similar_workloads",
                "unknown_queries",
            ],
            "rl_algorithm": ["stable_baselines_version", "algorithm", "gamma", "policy", "args"],
            "budgets": ["training", "validation_and_testing"],
        }

        with open(configuration_file) as f:
            self.config = json.load(f)

        # Check if configuration options are missing in json file
        self._determine_missing_configuration_options(
            self.REQUIRED_CONFIGURATION_OPTIONS_FIRST_LEVEL, self.config.keys()
        )
        for key, required_options in self.REQUIRED_CONFIGURATION_OPTIONS_FURTHER.items():
            self._determine_missing_configuration_options(required_options, self.config[key].keys())

        # Check if the json file has unknown configuration options
        self._determine_missing_configuration_options(
            self.config.keys(), self.REQUIRED_CONFIGURATION_OPTIONS_FIRST_LEVEL, crash_on_fail=False
        )

        self._translate_budgets()
        self._translate_column_filters()
        self._translate_model_architecture()

        self._check_dependencies()

    def _determine_missing_configuration_options(
        self, expected_configuration_options, actual_configuration_options, crash_on_fail=True
    ):
        missing_configuration_options = set(expected_configuration_options) - set(actual_configuration_options)

        if crash_on_fail:
            assert (
                missing_configuration_options == frozenset()
            ), f"Configuration misses required configuration option: {missing_configuration_options}"
        else:
            if len(missing_configuration_options) > 0:
                logging.warning(
                    f"The following configuration options are missing or optional: {missing_configuration_options}"
                )

    def _translate_model_architecture(self):
        if "model_architecture" in self.config["rl_algorithm"]:
            return
        self.config["rl_algorithm"]["model_architecture"] = None

    def _translate_budgets(self):
        for key in self.config["budgets"].keys():
            if self.config["budgets"][key] is False:
                self.config["budgets"][key] = None

    def _translate_column_filters(self):
        if "column_filters" in self.config:
            return

        self.config["column_filters"] = {}

    def _check_dependencies(self):
        if self.config["rl_algorithm"]["algorithm"] == "DQN":
            if self.config["parallel_environments"] > 1:
                raise ValueError("For DQN parallel parallel_environments must be 1.")

        if "Embedding" in self.config["observation_manager"]:
            assert (
                "workload_embedder" in self.config
            ), "A WorkloadEmbedder must be specified for embedding ObservationManagers."

        if self.config["workload"]["unknown_queries"] > 0:
            assert (
                sorted(self.config["workload"]["validation_testing"]["unknown_query_probabilities"])
                == self.config["workload"]["validation_testing"]["unknown_query_probabilities"]
            ), "unknown_query_probabilities should be sorted."
            assert (
                self.config["workload"]["validation_testing"]["unknown_query_probabilities"][-1] > 0
            ), "There is no point in specyfing unknown queries if unknown_query_probability is 0."

        if self.config["workload"]["benchmark"] != "JOB":
            assert (
                len(self.config["workload"]["excluded_query_classes"]) > 0
            ), "Are you sure that these workloads should not exclude certain query classes?"

        if (
            "query_class_change_frequency" in self.config["workload"]
            and self.config["workload"]["query_class_change_frequency"] is not None
        ):
            assert (
                self.config["workload"]["similar_workloads"] is True
            ), "Workloads must be similar if query_class_change_frequency is specified"

        if self.config["max_index_width"] > 1:
            assert (
                "Multi" in self.config["action_manager"]
            ), "MultiIndexActionManager must be used for max_index_width > 1"

        if self.config["reenable_indexes"] is True:
            assert (
                self.config["max_index_width"] > 1
            ), "If indexes should be reenabled, the max_index_width must be > 1 to have effect"
