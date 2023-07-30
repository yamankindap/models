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

Class methods:

* get_parameter_values()

    Return a dictionary of parameters.

* set_parameter_values(**kwargs)

    Create/modify instance variables given a parameter dictionary.