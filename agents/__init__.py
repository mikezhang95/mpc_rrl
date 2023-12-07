import abc


class Agent(object):
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        return None

    @abc.abstractmethod
    def train(self, training=True):
        """Sets the agent in either training or evaluation mode."""
        pass

    @abc.abstractmethod
    def update(self, replay_buffer, logger, step):
        """Main function of the agent that performs learning."""
        pass

    @abc.abstractmethod
    def act(self, obs, sample=False):
        """Issues an action given an observation."""
        pass

    @abc.abstractmethod
    def load(self, agent_dir):
        return

    @abc.abstractmethod
    def save(self, agent_dir):
        pass
