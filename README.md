# Probabilistic Modelling and Inference Tools:

## Models:

* Gaussian process
* Non-Gaussian process
* State-space models

## API Reference:

**ParameterInterface**

Implements the functionality that is assumed to exist in parameterised classes.

Class variables:

* parameter_keys

    A list of parameter keys required to instantiate the class.

Instance variables:

* parameters

    A dictionary of parameter key and value pairs.

* parameter_schema

    A dictionary of dictionaries containing the type and shape of each parameter key, value pair.

Class methods:

* get_parameter_values()

    Return a dictionary of parameters.

* initialise_parameter_values(**kwargs)

    Create parameter attributes given a parameter dictionary.

* set_parameter_values(**kwargs)

    Modify parameters dictionary instance given a new parameter dictionary.

* set_parameter_schema()

    Create the parameter schema dictionary in initialisation.

* check_parameter_key_initialisation()

    Check if the keys in parameter_keys are provided in the initalisation of an instance.

* validate_parameters(key, value)

    Check if the parameter schema is satisfied when using set_parameter_values method.