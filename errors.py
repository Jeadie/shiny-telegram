class ActorNameException(Exception):
    """ Raised if the user inputs an invalid actor name.
    
    """
    pass

class ActorEnvironmentException(Exception):
    """Raised if the actor and environment are not compatible.

    """
    pass

class ActorException(Exception):
    """ A general exception for failures related to actors.

    """
    pass
