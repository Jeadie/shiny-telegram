# shiny-telegram
Some fun in OpenAI Gym using NeuroEvolution &amp; other RL

## Literature
* [Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning
](https://arxiv.org/pdf/1712.06567.pdf) 
  * Discusses Neuroevolution. Does not provide a DNN encoding scheme. Only mutation, no crossover used. 
  * Uses a fixed seed random fn generator to encode the progression of DNNs. The ith DNN can then be calculated from all the prior DNNs recursively. (which in essence is just the addition of a list of random noise vectors)
