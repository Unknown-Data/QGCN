# Boost.python wrapper files

For efficiency, it is advisable to wrap each code module by itself, and only then add all the wrappings in the main file.
In each of the files in this directory are the wrapper function/functions for a module of code (either a feature or a set of them) in each the "def" function of Boost::python is called on each of the functions we want to expose in the python module.
These functions also perform the wrapping code, i.e. any preproccessing and\or postproccessing before and after calculating the actual features.
