import argparse
from typing import Type, Dict

from actor import (
    BaseActor,
    PendulumDNNActor
)
from errors import (
    ActorException, 
    ActorNameException,
    ActorEnvironmentException,
)
 
ACTOR_MAPPING = {
    "BaseActor" : (["Pendulum-v0"], BaseActor), 
    "PendulumDNNActor" : (["Pendulum-v0"], PendulumDNNActor), 
}


def setup_args()->Dict[str, object]:
    """ Parses the user input arguments and sanitises them.

    :return: A namespace over the inputted arguments. Compatibliity between the
             actor and environment has been checked.

    """
    parser = argparse.ArgumentParser(description='Neuroevolution for Reinforcement Learning.')
    parser.add_argument('--logdir',
            help="path to save tensorboard outputs.",
            type=str,
            default='./',
    )
    parser.add_argument("--env", 
        help= "Environment to run actors in. Environment and actors must be compatible.", 
        required=True, 
        type=str, 
    )
    parser.add_argument("--actors", 
            help="Number of actors to use at once.", 
            type=int, 
            required=True,
    )
    parser.add_argument("--generations", 
            help="Number of generations to run the selection for.", 
            type=int, 
            required=True,
    )
    parser.add_argument("--episodes", 
            help="Number of training episodes an actor gets per generation.", 
            type=int, 
            required=True,
    )
    parser.add_argument("--duration", 
            help="The maximum number of steps in a single episode.", 
            type=int, 
            required=True,
    )
    parser.add_argument("--actor-type", 
            help="The name of the actor to use in the experiment. Must be compatible with selected environment.", 
            type=str, 
            required=True,
    )
    args = parser.parse_args()
     
    if args.actors <= 1:
        raise ActorException("An experiment requires at least 2 actors.")

    args.actor_type = get_actor_class(args.actor_type, args.env)
    return vars(args)

def get_actor_class(actor_name:str, env:str)->Type[BaseActor]:
    """ Validates that the actor and environment are valid and converts the
        actor name to a class.

        :param actor_name: Name of the actor.
        :param env: Name of the environment
        :return: The class type of the actor if it is valid name and can
                operate in the environment.

    """
    actor_value = ACTOR_MAPPING.get(actor_name)
    if not actor_value:
        raise ActorNameException(f"Actor: {actor_name}, is not implemented.")

    env_list, actor_class = actor_value

    if env not in env_list:
        raise ActorEnvironmentException(f"Actor {actor_name} cannot operate in {env}, only in {env_list}.")

    return actor_class
   



