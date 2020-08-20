#include <iostream>

#include "config.h"
#include "dataset.h"
#include "nn_model.h"
#include "genetic_algorithm_optimizer.h"
#include "quadratic_loss.h"
#include "population.h"
#include "mutations.h"
#include "crossover.h"
#include "tourney_layer.h"
#include "fully_connected_layer.h"
#include "activation_functions.h"
#include "initializers.h"

using namespace nn;

template<typename Loss>
void init_model(const config& cfg, nn_model<Loss>& model)
{
	const auto& sizes = cfg.weight_bias_sizes();
	model << std::make_unique<tourney_layer>(matrix(sizes[0].first, sizes[0].second, nullptr));
	
	for(size_t i = 2, n = sizes.size(); i < n; i+=2)
	{
		model << std::make_unique<fully_connected_layer<activation::sigmoid>>(activation::sigmoid{},
																			  matrix(sizes[i].first, sizes[i].second, nullptr),
																			  matrix(sizes[i + 1].first, sizes[i + 1].second, nullptr));
	}
}

template<typename Loss>
void init_population(const config& cfg, population& pop, nn_model<Loss>& model, const dataset& set)
{
	const auto& sizes = cfg.weight_bias_sizes();
	
	std::vector<population::init_pair> inits;

	inits.push_back(population::init_pair{sizes[0].first * sizes[0].second, initializers::random_init(-1, 1)});
	
	for(size_t i = 2, n = sizes.size(); i < n; i+=2)
	{
		inits.push_back(population::init_pair{sizes[i].first * sizes[i].second, initializers::xavier_init(sizes[i].first, sizes[i].second)});
		inits.push_back(population::init_pair{sizes[i + 1].first * sizes[i + 1].second, initializers::zero_init});
	}
	
	population::initialize_params(inits, pop);
	population::init_penalties(pop, model, set);
	
}

genetic_algorithm_optimizer make_gen_alg_optimizer(const config& cfg)
{
    const auto& t_vec = cfg.ts();
    std::vector<double> v(t_vec.size());
    double sum = 0;
    for(size_t i = 0, n = t_vec.size(); i < n; ++i)
    {
        sum += t_vec[i];
    }

    for(size_t i = 0, n = v.size(); i < n; ++i)
    {
        v[i] = t_vec[i] / sum;
    }
	
	const auto& pm_vec = cfg.pm();
	const auto& stddev = cfg.stddev();
    std::vector<mutation_t> mutations(stddev.size());
    
    size_t mut_num = mutations.size() - 1;
    for(size_t i = 0; i < mut_num; ++i)
    {
    	mutations[i] = mutations::make_mutation(pm_vec[i], stddev[i], [](double& solution_elem, double rand)
																	  {
																			solution_elem += rand;
																	  });
	}
	mutations[mut_num] = mutations::make_mutation(pm_vec[mut_num], stddev[mut_num], [](double& solution_elem, double rand)
																					{
																						solution_elem = rand;	
																					});
	
	
	std::vector<crossover_t> crossovers(3);
	crossovers[0] = crossover::simple_arithmetic(cfg.solution_size());
	crossovers[1] = crossover::discrete;
	crossovers[2] = crossover::complete_arithmetic;
	
	return genetic_algorithm_optimizer(std::move(mutations), std::move(crossovers), std::move(v), cfg.k());
}

template<typename Loss>
void clear_model(nn_model<Loss>& model)
{
    for(size_t i = 0, n = model.layer_num(); i < n; ++i)
    {
        auto& layer_ptr = model[i];
        layer_ptr->get_weights().data() = nullptr;
        layer_ptr->get_bias().data() = nullptr;
    }
}

void round_predictions(matrix& predicted)
{
	for(size_t i = 0, n = predicted.rows(); i < n; ++i)
	{
		size_t max_pos = 0;
		for(size_t j = 1, m = predicted.columns(); j < m; ++j)
		{
			if(predicted[i][j] > predicted[i][max_pos])
			{
				predicted[i][max_pos] = 0.0;
				max_pos = j;
			}
			else
			{
				predicted[i][j] = 0.0;
			}
		}
		predicted[i][max_pos] = 1;
	}
}

int diff(const matrix& true_values, const matrix& predicted)
{
	int diff = 0;
	for(size_t i = 0, n = true_values.rows(); i < n; ++i)
	{
		for(size_t j = 0, m = predicted.columns(); j < m; ++j)
		{
			int tr_val = static_cast<int>(true_values[i][j]);
			int pred_val = static_cast<int>(predicted[i][j]);
			printf("%d ", pred_val);
			if(tr_val != pred_val)
			{
				++diff;
				break;
			}
		}
		
		for(size_t j = 0, m = true_values.columns(); j < m; ++j)
		{
			printf("%d ", static_cast<int>(true_values[i][j]));
		}
		printf("\n");
	}
	
	return diff;
}

int main(int argc, char** argv) 
{
	
	if(argc < 2)
	{
		fprintf(stderr, "Must specify a path to a config file as parameter!\nExiting...");
		exit(1);
	}
	
	std::ios_base::sync_with_stdio(false);

	printf("Loading config file %s\n", argv[1]);
	config cfg(argv[1]);
	printf("Finished config load\n\n");


	printf("Loading dataset: %s\n", cfg.data_file().data());
    const dataset data_set = dataset::load(cfg.data_file().data(), 2, 3);
	printf("Dataset load finished\n\n");
	
    nn_model<quadratic_loss> model;

	population pop(cfg.population_size(), cfg.solution_size());
	
	printf("Initializing neural network model\n");
    init_model(cfg, model);
    printf("Neural network model initialization finshed\n\n");
    
    printf("Initializing population of size %d\n", cfg.population_size());
    init_population(cfg, pop, model, data_set);
    printf("Finished population initialization\n\n");
	
    auto opt = make_gen_alg_optimizer(cfg);
    
    cfg.clear();
    
    double error = 999999;
    printf("Starting %ld epochs of training...\n\n", cfg.epochs());
    for(size_t i = 0, n = cfg.epochs(); i < n && error > cfg.precision(); ++i)
    {
    	error = opt.train(model, data_set.samples(), data_set.labels(), cfg.mortality(), pop);
    	
    	if((i + 1) % cfg.log_point() == 0)
    		printf("Iteration: %d, Error %.8lf\n", i + 1, error);
	}
    printf("\nFinished!\nBest solution with error %.8lf\n", pop.best->penalty);
    
    auto predicted = model.predict(data_set.samples());
    
    round_predictions(predicted);
    
    printf("\n\nDIFF %d", diff(data_set.labels(), predicted));
    
    clear_model(model);
	return 0;
}
