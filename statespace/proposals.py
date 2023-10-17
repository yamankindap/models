import numpy as np

from primitive.proposals import ProposalModule

# Particles class is a container of weights and values during a single iteration of a particle filtering algorithm.

class Particles:

    def __init__(self, config=None, N=1):
        """Each particle is defined by a weight and a number of attributes. The Particles class is used as a container of particle weights and values in a particle filtering algorithm.

        The values may be multidimensional, as in the marginalised particle filtering case where the first dimension is a mean vector and the second dimension is a 
        covariance matrix. In this case separate attributes are created for each dimension of the particle. A NumPy array with shape (N, d, 1) called means and an
        array with shape (N, d, d) called covs will be initialised with NaNs.

        Here N is the number of particles and d is the number of dimensions of the state. Hence weights must have shape (N, 1). 

        The config argument is a dictionary with keys that name an attribute and values that define the shape of each attribute.
        """

        self.attributes = []
        self.schema = {}

        # Initialise weights
        weights = np.empty((N, 1))
        weights[:] = np.nan

        self.weights = weights

        self.attributes.append("weights")

        # Initialise values
        for key, shape in config.items():
            
            value = np.empty((N, shape[0], shape[1]))
            value[:] = np.nan

            setattr(self, key, value)
            self.attributes.append(key)
            self.schema = self.schema | {key:value.shape}

        self.schema = self.schema | {"weights":(N, 1)}

    def __dir__(self):
        """Returns an alphabetically sorted version of the attributes.
        """
        return self.attributes
    
    def get_particles(self):
        particles = {}
        for attr in self.attributes:
            particles = particles | {attr:getattr(self, attr)}
        return particles
        
    def update_particles(self, **kwargs):

        # Filter any particle key value pairs that were not initialised:
        particles = {key: kwargs[key] for key in self.attributes if key in kwargs}

        for key, value in particles.items():

            assert self.schema[key] == value.shape
            setattr(self, key, value)
    

class ParticleProposalModule(ProposalModule):
    parameter_keys = None

    def __init__(self, particles, model, **kwargs):
        # Set particles object and model object.
        self.particles = particles
        self.model = model

        # Set parameters.
        super().__init__(**kwargs)

    def likelihood(self, x_prime, x):
        """ The density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        pass

    def log_likelihood(self, x_prime, x):
        """ The log density g( \dot | \theta ) evaluated at \theta^\prime.
        """
        pass

    def propose(self, x, shape):
        pass


class SSMParticleProposalGenerator(ParticleProposalModule):

    def propose(self, dt):
        return np.array([self.model.A(dt) @ vals + self.model.I(s=0, t=dt) for vals in self.particles.values])