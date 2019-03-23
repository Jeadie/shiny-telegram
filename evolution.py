import gym
env = gym.make('Pendulum-v0')


for i_episode in range(2):
    observation = env.reset()
    for t in range(10):
        env.render()
        print("Observation", observation)
        action = env.action_space.sample()
        print("Action", action)
        observation, reward, done, info = env.step(action)
        print("Reward", reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

def run_episode(env, actor, episode_steps: int, show=True):
    """ Runs a single episode of the environment for a set number of steps,
        unless the episode has finished prior.

        :param env: The environment to act in 
        :param actor: The actor to work in the environment. It is assumed the
                      actor can operate in the environment. 
        :param episode_steps: The maximum number of steps to run the episode for.

    """
    observation = env.reset()
    for t in range(episode_steps):
        if show:
            env.render()
    
        # Get action from actor, given observation
        action = actor.choose_action(observation) #)env.action_space.sample()

        # get reward and new environment state from env
        observation, reward, done, info = env.step(action)

        # Feed reward and new state to actor.
        actor.add_observation(observation, reward, done, info)
        
        # If episode is done, finish
        if done:
            return



