# shiny-telegram
Some fun in OpenAI Gym using NeuroEvolution &amp; other RL

## Usage
To train an exisiting actor in a compatible environment: 
```bash
python -m initialiser \
  --env="Pendulum-v0" \
  --logdir=".//" \
  --actors=200 \
  --generations=200 \
  --episodes=10 \
  --duration=20 \
  --actor-type="PMutationContinousActor" \
```

### Actors
Currently, the following actors have been implemented: 
  * BaseActor: A random actor. 
  * MutationContinousActor: An actor that operates in a continous action and environment space. Actor improvements are based on a standard mutation only genetic algorithm. The fitness function is purely defined on expected reward. 

## Literature
Relevant ideas from RL literature 

* [Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning
](https://arxiv.org/pdf/1712.06567.pdf) 
  * Discusses Neuroevolution. Does not provide a DNN encoding scheme. Only mutation, no crossover used. 
  * Uses a fixed seed random fn generator to encode the progression of DNNs. The ith DNN can then be calculated from all the prior DNNs recursively. (which in essence is just the addition of a list of random noise vectors)

* [Malthusian Reinforcement Learning](https://arxiv.org/pdf/1812.07019.pdf)
  * Species sharing the same policy network. 
  * Species simultaneously updates the same network. 
  * Population growth of a species within a specific sub-domain (an island / archipelago) is proportional to success of species
  * Could be reimagined so that population growth dynamics can be used for NAS. 
