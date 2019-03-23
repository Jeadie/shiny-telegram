

class BaseActor(object):

    def __init__(self):
        self.name = "Base Actor"

    def choose_action(self, env, observation):
        action = env.action_space_sample()
        print(f"{self.name} performed {action}")
        return action

    def add_observation(self, observation, reward, done, info):
        print(f"New state, {observation}")


