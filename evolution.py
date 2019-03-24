import gym

from actor import PendulumDNNActor
import utils


def run_episode(env, actor, episode_steps: int, show=True):
    """ Runs a single episode of the environment for a set number of steps,
        unless the episode has finished prior.

        :param env: The environment to act in 
        :param actor: The actor to work in the environment. It is assumed the
                      actor can operate in the environment. 
        :param episode_steps: The maximum number of steps to run the episode for.

        :return: a list of the rewards the actor has recieved. 

    """
    rewards = []
    observation = env.reset()
    for t in range(episode_steps):
        if show:
            env.render()
    
        # Get action from actor, given observation
        action = actor.choose_action(env, observation) #)env.action_space.sample()

        # get reward and new environment state from env
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        
        # Feed reward and new state to actor.
        actor.get_reward(observation, reward, done, info)
        
        # If episode is done, finish
        if done:
            break

    return rewards

def get_best_actors(actors, rewards):
    """ Returns the two best actors based on the rewards they have received.

        :param actors:
        :param rewards: 

        :return: Two actors, who scored the best rewards (may not be the
                maximum, some sort of cumulative maximum function).

    """
    # Sum the rewards 
    sum_rewards = [sum(actor_rewards) for actor_rewards in rewards]
    print(f"{max(sum_rewards)}, {sum(sum_rewards)/len(sum_rewards)}")
    # Pair rewards with actor
    actor_reward = list(zip(sum_rewards, actors))

    # Sort by score
    actor_reward = sorted(actor_reward, key=lambda x: x[0], reverse=True)

    # Take 2 best actors
    return [actor[-1] for actor in actor_reward[0:2]]


def mutate_actors(actors, no_offspring):
    """ Creates new offspring based on the two actors given.

    :param actors: 
    :param no_offspring:

    :return: A list of no_offspring actors mutated from actors. 
    """
    actor_1, actor_2 = actors
    return [actor_1.mutate(actor_2) for i in range(no_offspring)]


def run_experiment(actor_fn, env_fn, num_actors, no_generation, no_episode, ep_duration, writer=None): 
    """ Runs a neuroevolution experiment. 

        :param actor_fn: 
        :param env_fn: 
        :param num_actors: 
        :param no_generation: 
        :param no_episode: 
        :param ep_duration:

        :return: 
    """
    
    actors = [actor_fn() for i in range(num_actors)]

    env = env_fn()
    env.reset()

    for g in range(no_generation):
        actor_rewards = []
        for actor in actors:
            generation_reward = []
            for ep in range(no_episode):
                episode_reward = run_episode(env, actor, ep_duration, show=False)                
                generation_reward.extend(episode_reward)

            actor_rewards.append(generation_reward)

        # Find best actors, 
        winner_actors = get_best_actors(actors, actor_rewards)
        # mutate generations
        offspring = mutate_actors(winner_actors, num_actors-2)
        actors = offspring
        actors.extend(winner_actors)
        
    env.reset()
    return get_best_actors(actors, actor_rewards)

def main(args):
    writer = tf.summary.FileWriter(args.logdir)

    actor_fn = lambda: PendulumDNNActor()
    env_fn = lambda: gym.make("Pendulum-v0")
    actors = 5
    generations = 100
    episodes = 5
    ep_duration = 10

    winner = run_experiment(actor_fn, env_fn, actors, generations, episodes, ep_duration, writer=writer)
    

