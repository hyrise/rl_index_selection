from index_selection_evaluation.selection.utils import b_to_mb


class RewardCalculator(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.accumulated_reward = 0

    def calculate_reward(self, environment_state):
        current_cost = environment_state["current_cost"]
        previous_cost = environment_state["previous_cost"]
        initial_cost = environment_state["initial_cost"]
        new_index_size = environment_state["new_index_size"]

        assert new_index_size is not None

        reward = self._calculate_reward(current_cost, previous_cost, initial_cost, new_index_size)

        self.accumulated_reward += reward

        return reward

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        raise NotImplementedError


class AbsoluteDifferenceRelativeToStorageReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = (previous_cost - current_cost) / new_index_size

        return reward


class AbsoluteDifferenceToPreviousReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = previous_cost - current_cost

        return reward


class RelativeDifferenceToPreviousReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = (previous_cost - current_cost) / initial_cost

        return reward


class RelativeDifferenceRelativeToStorageReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

        self.SCALER = 1

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        assert new_index_size > 0

        reward = ((previous_cost - current_cost) / initial_cost) / b_to_mb(new_index_size) * self.SCALER

        return reward


class DRLindaReward(RewardCalculator):
    def __init__(self):
        RewardCalculator.__init__(self)

    def _calculate_reward(self, current_cost, previous_cost, initial_cost, new_index_size):
        reward = ((initial_cost - current_cost) / initial_cost) * 100

        return reward
