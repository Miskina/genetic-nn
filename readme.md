# Genetic-NN
Implementation of a neural network (ANN) optimized using a K-tournament genetic algorithm.

## Requirements

- Any compiler supporting the C++17 standard
- OpenMP support
- OpenBLAS

## Implementation
The implementation uses a K-tournament genetic algorithm with 3 mutation operators picked at random with a given ratio for each and 3 randomly picked crossover operators each time a genetic solution is modified.

### Mutation
- The first type of mutation operator adds randomly from a *N(0, stddev)* distribution, where `stddev` is specified in the configuration.
- The second type of mutation operator replaces the value of the genetic solution with a random value from a *N(0, stddev)* distribution, where `stddev` is again specified in the configuration.

The first two mutation operators, *M1* and *M2* are both of the first type, while the last operator, *M3*, is of the second type.

### Crossover
For each crossover a random crossover operator is picked from one of the three:
- *Simple arithmetic* - Picks a random fixed point and copies values from one parent up until to the point and values from the other parent pass the point
- *Discrete* - Each value of the child is picked from one of the parents randomly
- *Complete arithmetic* - Averages out values of both parent -- `value_child = (value_parent_1 + value_parent_2) / 2`

## Running the application
The app requires a single command line parameter which is the path to a configuration file, an example of which can 
is [cfg.txt](cfg.txt). 

## Configuration
A configuration file consists of a list of parameters specified in the form of *variable_name*=*variable_value*.
The variables are the following:

- input_features - Number of input features of the dataset
- output_features - Number of output features for the dataset
- dataset_loc - File from which the dataset will be read
- epochs - How many training epochs to run
- log_point - Log the error of the neural network every *n=log_point* iterations
- hidden_layers - Number of fully connected layers to put before the output layer, seperated by 'x'
- K - The K-tournament genetic algorithm parameter
- mortality - How many solutions are replaced in each iteration
- population_size - The size of the population for the genetic algorithm
- T - Ratio at which each of the 3 mutations is picked, in the form of: `T=[M1_T:M2_T:M3_T]` - The higher the value, the higher the chance the operator will be picked.
- pm - Mutation probability for each of the mutation operators consecutively, in the form of: `pm=[P_M1:P_M2:P_M3]`
- stddev - The standard deviation of each of the mutation operator's normal distributions used (*N(0, stddev)*), in the form of: `stddev=[STD_M1:STD_M2:STD_M3]`

