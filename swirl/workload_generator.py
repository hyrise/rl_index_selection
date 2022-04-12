import copy
import logging
import random

import numpy as np

import swirl.embedding_utils as embedding_utils
from index_selection_evaluation.selection.candidate_generation import (
    candidates_per_query,
    syntactically_relevant_indexes,
)
from index_selection_evaluation.selection.cost_evaluation import CostEvaluation
from index_selection_evaluation.selection.dbms.postgres_dbms import PostgresDatabaseConnector
from index_selection_evaluation.selection.utils import get_utilized_indexes
from index_selection_evaluation.selection.workload import Query, Workload

from .workload_embedder import WorkloadEmbedder

QUERY_PATH = "query_files"


class WorkloadGenerator(object):
    def __init__(
        self, config, workload_columns, random_seed, database_name, experiment_id=None, filter_utilized_columns=None
    ):
        assert config["benchmark"] in [
            "TPCH",
            "TPCDS",
            "JOB",
        ], f"Benchmark '{config['benchmark']}' is currently not supported."

        # For create view statement differentiation
        self.experiment_id = experiment_id
        self.filter_utilized_columns = filter_utilized_columns

        self.rnd = random.Random()
        self.rnd.seed(random_seed)
        self.np_rnd = np.random.default_rng(seed=random_seed)

        self.workload_columns = workload_columns
        self.database_name = database_name

        self.benchmark = config["benchmark"]
        self.number_of_query_classes = self._set_number_of_query_classes()
        self.excluded_query_classes = set(config["excluded_query_classes"])
        self.varying_frequencies = config["varying_frequencies"]

        # self.query_texts is list of lists. Outer list for query classes, inner list for instances of this class.
        self.query_texts = self._retrieve_query_texts()
        self.query_classes = set(range(1, self.number_of_query_classes + 1))
        self.available_query_classes = self.query_classes - self.excluded_query_classes

        self.globally_indexable_columns = self._select_indexable_columns(self.filter_utilized_columns)

        validation_instances = config["validation_testing"]["number_of_workloads"]
        test_instances = config["validation_testing"]["number_of_workloads"]
        self.wl_validation = []
        self.wl_testing = []

        if config["similar_workloads"] and config["unknown_queries"] == 0:
            # Todo: this branch can probably be removed
            assert self.varying_frequencies, "Similar workloads can only be created with varying frequencies."
            self.wl_validation = [None]
            self.wl_testing = [None]
            _, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
                0, validation_instances, test_instances, config["size"]
            )
            if config["query_class_change_frequency"] is None:
                self.wl_training = self._generate_similar_workloads(config["training_instances"], config["size"])
            else:
                self.wl_training = self._generate_similar_workloads_qccf(
                    config["training_instances"], config["size"], config["query_class_change_frequency"]
                )
        elif config["unknown_queries"] > 0:
            assert (
                config["validation_testing"]["unknown_query_probabilities"][-1] > 0
            ), "Query unknown_query_probabilities should be larger 0."

            embedder_connector = PostgresDatabaseConnector(self.database_name, autocommit=True)
            embedder = WorkloadEmbedder(
                # Transform globally_indexable_columns to list of lists.
                self.query_texts,
                0,
                embedder_connector,
                [list(map(lambda x: [x], self.globally_indexable_columns))],
                retrieve_plans=True,
            )
            self.unknown_query_classes = embedding_utils.which_queries_to_remove(
                embedder.plans, config["unknown_queries"], random_seed
            )

            self.unknown_query_classes = frozenset(self.unknown_query_classes) - self.excluded_query_classes
            missing_classes = config["unknown_queries"] - len(self.unknown_query_classes)
            self.unknown_query_classes = self.unknown_query_classes | frozenset(
                self.rnd.sample(self.available_query_classes - frozenset(self.unknown_query_classes), missing_classes)
            )
            assert len(self.unknown_query_classes) == config["unknown_queries"]

            self.known_query_classes = self.available_query_classes - frozenset(self.unknown_query_classes)
            embedder = None

            for query_class in self.excluded_query_classes:
                assert query_class not in self.unknown_query_classes

            logging.critical(f"Global unknown query classes: {sorted(self.unknown_query_classes)}")
            logging.critical(f"Global known query classes: {sorted(self.known_query_classes)}")

            for unknown_query_probability in config["validation_testing"]["unknown_query_probabilities"]:
                _, wl_validation, wl_testing = self._generate_workloads(
                    0,
                    validation_instances,
                    test_instances,
                    config["size"],
                    unknown_query_probability=unknown_query_probability,
                )
                self.wl_validation.append(wl_validation)
                self.wl_testing.append(wl_testing)

            assert (
                len(self.wl_validation)
                == len(config["validation_testing"]["unknown_query_probabilities"])
                == len(self.wl_testing)
            ), "Validation/Testing workloads length fail"

            # We are temporarily restricting the available query classes now to exclude certain classes for training
            original_available_query_classes = self.available_query_classes
            self.available_query_classes = self.known_query_classes

            if config["similar_workloads"]:
                if config["query_class_change_frequency"] is not None:
                    logging.critical(
                        f"Similar workloads with query_class_change_frequency: {config['query_class_change_frequency']}"
                    )
                    self.wl_training = self._generate_similar_workloads_qccf(
                        config["training_instances"], config["size"], config["query_class_change_frequency"]
                    )
                else:
                    self.wl_training = self._generate_similar_workloads(config["training_instances"], config["size"])
            else:
                self.wl_training, _, _ = self._generate_workloads(config["training_instances"], 0, 0, config["size"])
            # We are removing the restriction now.
            self.available_query_classes = original_available_query_classes
        else:
            self.wl_validation = [None]
            self.wl_testing = [None]
            self.wl_training, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
                config["training_instances"], validation_instances, test_instances, config["size"]
            )

        logging.critical(f"Sample training workloads: {self.rnd.sample(self.wl_training, 10)}")
        logging.info("Finished generating workloads.")

    def _set_number_of_query_classes(self):
        if self.benchmark == "TPCH":
            return 22
        elif self.benchmark == "TPCDS":
            return 99
        elif self.benchmark == "JOB":
            return 113
        else:
            raise ValueError("Unsupported Benchmark type provided, only TPCH, TPCDS, and JOB supported.")

    def _retrieve_query_texts(self):
        query_files = [
            open(f"{QUERY_PATH}/{self.benchmark}/{self.benchmark}_{file_number}.txt", "r")
            for file_number in range(1, self.number_of_query_classes + 1)
        ]

        finished_queries = []
        for query_file in query_files:
            queries = query_file.readlines()[:1]
            queries = self._preprocess_queries(queries)

            finished_queries.append(queries)

            query_file.close()

        assert len(finished_queries) == self.number_of_query_classes

        return finished_queries

    def _preprocess_queries(self, queries):
        processed_queries = []
        for query in queries:
            query = query.replace("limit 100", "")
            query = query.replace("limit 20", "")
            query = query.replace("limit 10", "")
            query = query.strip()

            if "create view revenue0" in query:
                query = query.replace("revenue0", f"revenue0_{self.experiment_id}")

            processed_queries.append(query)

        return processed_queries

    def _store_indexable_columns(self, query):
        if self.benchmark != "JOB":
            for column in self.workload_columns:
                if column.name in query.text:
                    query.columns.append(column)
        else:
            query_text = query.text
            assert "WHERE" in query_text, f"Query without WHERE clause encountered: {query_text} in {query.nr}"

            split = query_text.split("WHERE")
            assert len(split) == 2, "Query split for JOB query contains subquery"
            query_text_before_where = split[0]
            query_text_after_where = split[1]

            for column in self.workload_columns:
                if column.name in query_text_after_where and f"{column.table.name} " in query_text_before_where:
                    query.columns.append(column)

    def _workloads_from_tuples(self, tuples, unknown_query_probability=None):
        workloads = []
        unknown_query_probability = "" if unknown_query_probability is None else unknown_query_probability

        for tupl in tuples:
            query_classes, query_class_frequencies = tupl
            queries = []

            for query_class, frequency in zip(query_classes, query_class_frequencies):
                query_text = self.rnd.choice(self.query_texts[query_class - 1])

                query = Query(query_class, query_text, frequency=frequency)

                self._store_indexable_columns(query)
                assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

                queries.append(query)

            assert isinstance(queries, list), f"Queries is not of type list but of {type(queries)}"
            previously_unseen_queries = (
                round(unknown_query_probability * len(queries)) if unknown_query_probability != "" else 0
            )
            workloads.append(
                Workload(queries, description=f"Contains {previously_unseen_queries} previously unseen queries.")
            )

        return workloads

    def _generate_workloads(
        self, train_instances, validation_instances, test_instances, size, unknown_query_probability=None
    ):
        required_unique_workloads = train_instances + validation_instances + test_instances

        unique_workload_tuples = set()

        while required_unique_workloads > len(unique_workload_tuples):
            workload_tuple = self._generate_random_workload(size, unknown_query_probability)
            unique_workload_tuples.add(workload_tuple)

        validation_tuples = self.rnd.sample(unique_workload_tuples, validation_instances)
        unique_workload_tuples = unique_workload_tuples - set(validation_tuples)

        test_workload_tuples = self.rnd.sample(unique_workload_tuples, test_instances)
        unique_workload_tuples = unique_workload_tuples - set(test_workload_tuples)

        assert len(unique_workload_tuples) == train_instances
        train_workload_tuples = unique_workload_tuples

        assert (
            len(train_workload_tuples) + len(test_workload_tuples) + len(validation_tuples) == required_unique_workloads
        )

        validation_workloads = self._workloads_from_tuples(validation_tuples, unknown_query_probability)
        test_workloads = self._workloads_from_tuples(test_workload_tuples, unknown_query_probability)
        train_workloads = self._workloads_from_tuples(train_workload_tuples, unknown_query_probability)

        return train_workloads, validation_workloads, test_workloads

    # The core idea is to create workloads that are similar and only change slightly from one to another.
    # For the following workload, we remove one random element, add another random one with frequency, and
    # randomly change the frequency of one element (including the new one).
    def _generate_similar_workloads(self, instances, size):
        assert size <= len(
            self.available_query_classes
        ), "Cannot generate workload with more queries than query classes"

        workload_tuples = []

        query_classes = self.rnd.sample(self.available_query_classes, size)
        available_query_classes = self.available_query_classes - frozenset(query_classes)
        frequencies = list(self.np_rnd.zipf(1.5, size))

        workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        for workload_idx in range(instances - 1):
            # Remove a random element
            idx_to_remove = self.rnd.randrange(len(query_classes))
            query_classes.pop(idx_to_remove)
            frequencies.pop(idx_to_remove)

            # Draw a new random element, the removed one is excluded
            query_classes.append(self.rnd.sample(available_query_classes, 1)[0])
            frequencies.append(self.np_rnd.zipf(1.5, 1)[0])

            frequencies[self.rnd.randrange(len(query_classes))] = self.np_rnd.zipf(1.5, 1)[0]

            available_query_classes = self.available_query_classes - frozenset(query_classes)
            workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        workloads = self._workloads_from_tuples(workload_tuples)

        return workloads

    # This version uses the same query id selction for query_class_change_frequency workloads
    def _generate_similar_workloads_qccf(self, instances, size, query_class_change_frequency):
        assert size <= len(
            self.available_query_classes
        ), "Cannot generate workload with more queries than query classes"

        workload_tuples = []

        while len(workload_tuples) < instances:
            if len(workload_tuples) % query_class_change_frequency == 0:
                query_classes = self.rnd.sample(self.available_query_classes, size)

            frequencies = list(self.np_rnd.integers(1, 10000, size))
            workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        workloads = self._workloads_from_tuples(workload_tuples)

        return workloads

    def _generate_random_workload(self, size, unknown_query_probability=None):
        assert size <= self.number_of_query_classes, "Cannot generate workload with more queries than query classes"

        workload_query_classes = None
        if unknown_query_probability is not None:
            number_of_unknown_queries = round(size * unknown_query_probability)
            number_of_known_queries = size - number_of_unknown_queries
            assert number_of_known_queries + number_of_unknown_queries == size

            known_query_classes = self.rnd.sample(self.known_query_classes, number_of_known_queries)
            unknown_query_classes = self.rnd.sample(self.unknown_query_classes, number_of_unknown_queries)
            query_classes = known_query_classes
            query_classes.extend(unknown_query_classes)
            workload_query_classes = tuple(query_classes)
            assert len(workload_query_classes) == size
        else:
            workload_query_classes = tuple(self.rnd.sample(self.available_query_classes, size))

        # Create frequencies
        if self.varying_frequencies:
            query_class_frequencies = tuple(list(self.np_rnd.integers(1, 10000, size)))
        else:
            query_class_frequencies = tuple([1 for frequency in range(size)])

        workload_tuple = (workload_query_classes, query_class_frequencies)

        return workload_tuple

    def _only_utilized_indexes(self, indexable_columns):
        frequencies = [1 for frequency in range(len(self.available_query_classes))]
        workload_tuple = (self.available_query_classes, frequencies)
        workload = self._workloads_from_tuples([workload_tuple])[0]

        candidates = candidates_per_query(
            workload,
            max_index_width=1,
            candidate_generator=syntactically_relevant_indexes,
        )

        connector = PostgresDatabaseConnector(self.database_name, autocommit=True)
        connector.drop_indexes()
        cost_evaluation = CostEvaluation(connector)

        utilized_indexes, query_details = get_utilized_indexes(workload, candidates, cost_evaluation, True)

        columns_of_utilized_indexes = set()
        for utilized_index in utilized_indexes:
            column = utilized_index.columns[0]
            columns_of_utilized_indexes.add(column)

        output_columns = columns_of_utilized_indexes & set(indexable_columns)
        excluded_columns = set(indexable_columns) - output_columns
        logging.critical(f"Excluding columns based on utilization:\n   {excluded_columns}")

        return output_columns

    def _select_indexable_columns(self, only_utilized_indexes=False):
        available_query_classes = tuple(self.available_query_classes)
        query_class_frequencies = tuple([1 for frequency in range(len(available_query_classes))])

        logging.info(f"Selecting indexable columns on {len(available_query_classes)} query classes.")

        workload = self._workloads_from_tuples([(available_query_classes, query_class_frequencies)])[0]

        indexable_columns = workload.indexable_columns()
        if only_utilized_indexes:
            indexable_columns = self._only_utilized_indexes(indexable_columns)
        selected_columns = []

        global_column_id = 0
        for column in self.workload_columns:
            if column in indexable_columns:
                column.global_column_id = global_column_id
                global_column_id += 1

                selected_columns.append(column)

        return selected_columns
