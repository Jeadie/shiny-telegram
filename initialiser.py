import tensorflow as tf
import gym

import utils
from evolution import run_experiment

if __name__ == '__main__':
    args = utils.setup_args()

    writer = tf.summary.FileWriter(args["logdir"])
    actor_fn = lambda env: args["actor_type"](env)
    env_fn = lambda: gym.make(args["env"])

    winner = run_experiment(actor_fn, env_fn, args["actors"], args["generations"], args["episodes"], args["duration"], writer=writer)

