import numpy as np

from primitive.priors import PriorModule
from primitive.proposals import ProposalModule

# Base inference module class:


#############
# DEV NOTES #
#############
# The key-value structure of the parameters is reflected nicely in the class. Some basic tests for matching the model parameters is implemented.
# The general API of the model object is not sufficiently specificed. We know that it must possess a .get_parameter_values() method. There is no other dependence at this stage.
# It looks like set_training_variables method is not necessary at this abstract stage.

class InferenceModule:

    def __init__(self, model, prior, proposal):
        """A model defines the parameters to be learned and returns a log likelihood value (and possibly a vector of gradients) given a set of parameters and measurements.
        A prior is a PriorModule instance or a dictionary of parameter keys and PriorModule instances.
        A proposal is a ProposalModule instance or a dictionary of parameter keys and ProposalModule instances.
        """

        # Set model:
        self.model = model

        # Initalise instance parameters to be learned.
        self.parameters = self.model.get_parameter_values()

        # Set prior and proposal objects.
        self.prior = prior

        # There may be an option to use the model itself as the proposal distribution.
        # This will allow us to have subordinators of NGPs that propose samples.
        self.proposal = proposal

        # Check if each parameter has an associated PriorModule and ProposalModule.
        # Check if prior and proposal keys match parameter keys.
        self.valid = self.validate_prior() and self.validate_proposal()

    def validate_prior(self):
        """There are two types of priors in an InferenceModule: the prior may be a PriorModule object or the prior may be a dictionary of PriorModule objects associated
        individually with model parameters. The latter type is required for approximate methods such as Gibbs sampling.
        """

        if isinstance(self.prior, PriorModule):
            return True
        else:
            if isinstance(self.prior, dict):
                for key, value in self.prior.items():
                    if isinstance(value, PriorModule):
                        # Check if prior key matches a key in model parameters.
                        if key in self.parameters:
                            continue
                        else:
                            print("Prior keys do not match model parameters.")
                            return False
                    else:
                        print("The prior configuration is not valid.")
                        return False
            else:
                print("The prior configuration is not valid.")
                return False
        
        return True
            
    def validate_proposal(self):
        """There are two types of priors in an InferenceModule: the prior may be a PriorModule object or the prior may be a dictionary of PriorModule objects associated
        individually with model parameters. The latter type is required for approximate methods such as Gibbs sampling.
        """

        if isinstance(self.proposal, ProposalModule):
            return True
        else:
            if isinstance(self.proposal, dict):
                for key, value in self.proposal.items():
                    if isinstance(value, ProposalModule):
                        # Check if proposal key matches a key in model parameters.
                        if key in self.parameters:
                            continue
                        else:
                            print("Proposal keys do not match model parameters.")
                            return False
                    else:
                        print("The proposal configuration is not valid.")
                        return False
            else:
                print("The proposal configuration is not valid.")
                return False
        
        return True

    def set_training_variables(self, y, X, Xeval):
        self.y = y
        self.X = X

        if Xeval is None:
            self.Xeval = self.X
        else:
            self.Xeval = Xeval

    def initialise(self):
        pass

    def sample(self):
        pass
    
    def fit(self, y, X, n_samples=10):
        pass