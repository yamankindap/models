# Parameterised abstract class or protocol:

## The ParameterInterface class implements the functionality that is assumed to exist in parameterised classes. It should contain
## the interface api and implementations for passing parameters between classes.

class ParameterInterface:
    parameter_keys = None

    def __init__(self, **kwargs):

        # Set parameter configuration.
        self.parameters = kwargs
        self.set_parameter_values(**self.parameters)

        # Binary variable to check if parameters are valid.
        self.valid = self.validate_parameters()

    def get_parameter_values(self):
        # Return a dictionary of parameters.
        return self.parameters
    
    def set_parameter_values(self, **kwargs):
        
        # Filter any parameter key value pairs not related to specific module:
        new_parameters = {key: kwargs[key] for key in self.__class__.parameter_keys if key in kwargs}

        for key, value in new_parameters.items():
            # Create/update dictionary of parameters.
            self.parameters[key] = value

            # Create/modify instance parameters named after the key and stores value.
            setattr(self, key, value)

    def validate_parameters(self):
        # If any key in parameter_keys are not found in the initialised parameters dictionary:
        if [parameter_key for parameter_key in self.__class__.parameter_keys if parameter_key not in self.parameters.keys()]:
            print("Parameter values are not initialised.")
            return False
        else:
            return True