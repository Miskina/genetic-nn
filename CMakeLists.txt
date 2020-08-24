cmake_minimum_required(VERSION 3.14)
project(genetic-nn)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(gentic-nn
        activation_functions.cpp
        activation_functions.h
        config.cpp
        config.h
        crossover.cpp
        crossover.h
        dataset.cpp
        dataset.h
        fully_connected_layer.cpp
        fully_connected_layer.h
        genetic_algorithm_optimizer.cpp
        genetic_algorithm_optimizer.h
        genetic_solution.cpp
        genetic_solution.h
        gradient_desc_optimizer.cpp
        gradient_desc_optimizer.h
        initializers.cpp
        initializers.h
        layer.cpp
        layer.h
        main.cpp
        matrix.cpp
        matrix.h
        mutations.cpp
        mutations.h
        nn_model.cpp
        nn_model.h
        population.cpp
        population.h
        quadratic_loss.cpp
        quadratic_loss.h
        selection.cpp
        selection.h
        solution_pool.cpp
        solution_pool.h
        util.cpp
        util.h
        tourney_layer.h
        tourney_layer.cpp)

target_link_libraries(genetic-nn openBLAS)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(genetic OpenMP::OpenMP_CXX)
endif()