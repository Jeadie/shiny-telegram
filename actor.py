import random

import numpy as np
import tensorflow as tf


class BaseActor(object):

    def __init__(self, env):
        self.name = "Base Actor"
        self.env = env

    def choose_action(self, observation, session=None):
        """ Selects an action to apply to the environment given the
            observation of the previous state.

        :param env: The environment the actor is in. 
        :param observation: The previous observations from the environment.

        :return: An action to choose from the given environment.
        
        """
        action = self.env.action_space.sample()
        return action

    def get_reward(self, observation, reward, done, info, session=None):
        """Updates the actor based on the reward received, given the previous
           action/state space.

        :param observation: The observed state of the environment.
        :param reward: The reward from the previous action.
        :param done: True, if the episode has finished.
        :param info: Additional information.

        """

    def mutate(self, other):
        """ Returns an offspring that is a mutation between self and an other actor.

        :param other: Another actor of the same class type
        :return: A separate, new actor. 

        """
        return BaseActor()


class MutationContinousActor(BaseActor):

    OUTPUT_LAYER_NAME = "cts_action_value"
    CHILD_e = 0.005

    def __init__(self, env, graph=None, model=None):
        self.env = env
        self.name = "Pendulum DNN Actor"
        self.graph=graph 
        if model:
            self.model = model
        else:
            self.model = self.create_model()

    def create_model(self): 
        """ Creates a DNN mapping state to continous actions. Maps 

        :return: A constructed keras model.
        """
        if not self.graph: 
          self.graph = tf.Graph()
        with self.graph.as_default():
          self.state = tf.placeholder(tf.float32, shape=(1, self.env.observation_space.shape[0]))
          x = tf.layers.dense(self.state, 8, activation=tf.nn.relu)
          x = tf.layers.dense(x, 4, activation=tf.nn.relu)
          x = tf.layers.dense(x, self.env.action_space.shape[0], activation=tf.nn.sigmoid)
          
          # Convert to continous_action_space_domain. 
          # TODO: checks for dim > 1 outputs 
          a,b = self.env.action_space.low, self.env.action_space.high       
          model = float(b-a) * x + float(a)
        return tf.identity(model, name="output")


    def choose_action(self, env, observation, session=None):
        """ Selects an action to apply to the environment given the
            observation of the previous state.

        :param env: The environment the actor is in. 
        :param observation: The previous observations from the environment.

        :return: An action to choose from the given environment.
        
        """
        if session: 
          return session.run(self.model, feed_dict={self.state: np.reshape(observation, [1, -1])})[0]

        else: 
          with tf.Session(graph=self.graph) as sess:
            return session.run(self.model, feed_dict={self.state: observation})[0]
 

    def mutate(self, other):
        """ Returns an offspring that is a mutation between self and an other actor.

        :param other: Another actor of the same class type
        :return: A separate, new actor. 

        """
        actor = random.choice([self, other])    
        new_graph = deep_copy_graph(actor.graph)
        with tf.Session(graph=new_graph) as sess:
          sess.run(tf.global_variables_initializer())
          graph = tf.get_default_graph()
          for t_var in tf.trainable_variables():
            add_random_noise(t_var, stddev=MutationContinousActor.CHILD_e)
        
        model = new_graph.get_tensor_by_name(MutationContinousActor.OUTPUT_LAYER_NAME)
        return MutationContinousActor(self.env.unwrapped.spec.id, model=model, graph=new_graph)


### Utilities ###

#################


def deep_copy_graph(g): 
    new_g = tf.Graph()
    tf.contrib.graph_editor.copy(g, new_g)
    return new_g


def add_random_noise(w, mean=0.0, stddev=1.0):
    variables_shape = tf.shape(w)
    noise = tf.random_normal(
        variables_shape,
        mean=mean,
        stddev=stddev,
        dtype=tf.float32,
    )
    return tf.assign_add(w, noise)
