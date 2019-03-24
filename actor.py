import random

import numpy as np
from keras.models import Sequential
from keras.layers import Dense


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
        action = env.action_space.sample()
        return action

    def get_reward(self, observation, reward, done, info):
        """ Returns the reward received from the previous action.

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


class PendulumDNNActor(BaseActor):

    def __init__(self, model=None):
        self.name = "Pendulum DNN Actor"
        self.e = 0.00005        
        if model:
            self.model = model
        else:
            self.model = self.create_model()

    def create_model(self): 
        """ Creates a Keras Deep Neural Network model. Takes a vector of
            length 3 as input and outputs a scalar on the domain [-2,2]. 

        :return: A constructed keras model.
        """
        model = Sequential()
        model.add(Dense(8, input_dim=3, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model


    def choose_action(self, env, observation):
        """ Selects an action to apply to the environment given the
            observation of the previous state.

        :param env: The environment the actor is in. 
        :param observation: The previous observations from the environment.

        :return: An action to choose from the given environment.
        
        """
        a = self.model.predict(np.array([observation]))
        return a[0]

    def mutate_weights(self, w1, w2):
        """ Mutates the weights of two, same shaped, model weights. 

        """
        options = [w1, w2]
        random_w = lambda: options[random.random() > 0.5]
        weights = []

        # Input weights have no bias
        input_w = []
        for j in range(len(w1[0])):
            w = random_w()
            input_w.append(w[0][j][:]+ self.e*np.random.rand(w[0].shape[-1])) 
        weights.append(input_w)

        # Do middle layers
        # To randomise neurons, keep bias and weights together
        for i in range(1, len(w1)-1, 2):
            weight = []
            bias = []

            # Go through each neuron
            for j in range(w1[i].shape[0]):
                w = random_w()
                weight.append(w[i+1][j][:]) 
                bias.append(w[i][j])
            
            # Add noise to weights and biases
            weight = weight + self.e*np.random.rand(len(weight), weight[0].shape[0])
            bias = bias + self.e*np.random.rand(len(bias))
            weights.extend([bias, weight])

        # add final output bias
        weights.append(random_w()[-1] + self.e*np.random.rand(1))
        return weights


    def mutate(self, other):
        """ Returns an offspring that is a mutation between self and an other actor.

        :param other: Another actor of the same class type
        :return: A separate, new actor. 

        """
        config = self.model.get_config()

        weights = self.model.get_weights()
        other_weights = other.model.get_weights()
        new_weights = self.mutate_weights(weights, other_weights)

        model = Sequential.from_config(config)
        model.set_weights(new_weights)

        return PendulumDNNActor(model=model)


