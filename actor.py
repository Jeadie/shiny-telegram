

class BaseActor(object):

    def __init__(self):
        self.name = "Base Actor"

    def choose_action(self, env, observation):
        """ Selects an action to apply to the environment given the
            observation of the previous state.

        :param env: The environment the actor is in. 
        :param observation: The previous observations from the environment.

        :return: An action to choose from the given environment.
        
        """
        action = env.action_space_sample()
        print(f"{self.name} performed {action}")
        return action

    def get_reward(self, observation, reward, done, info):
        """ Returns the reward received from the previous action.

        :param observation: The observed state of the environment.
        :param reward: The reward from the previous action.
        :param done: True, if the episode has finished.
        :param info: Additional information.

        """
        print(f"New state, {observation}")

    def mutate(self, other):
        """ Returns an offspring that is a mutation between self and an other actor.

        :param other: Another actor of the same class type
        :return: A separate, new actor. 

        """
        return self.copy()
