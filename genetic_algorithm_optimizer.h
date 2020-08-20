#ifndef GENETIC_ALGORITHM_OPTIMIZER_H
#define GENETIC_ALGORITHM_OPTIMIZER_H

#include <vector>

#include "genetic_solution.h"
#include "population.h"
#include "selection.h"
#include "nn_model.h"
#include "dataset.h"
#include "util.h"
#include "config.h"

namespace nn
{
	using mutation_t = std::function<void(genetic_solution&, size_t)>;
	using crossover_t = std::function<void(const genetic_solution&, const genetic_solution&, genetic_solution&, size_t)>;
	
	struct genetic_algorithm_optimizer
	{
		
		genetic_algorithm_optimizer(std::vector<mutation_t>&& mutations,
									std::vector<crossover_t>&& crossovers,
									std::vector<double>&& vs,
									int k = 3) : mutate_(std::move(mutations)),
												 crossover_(std::move(crossovers)),
												 vs_(std::move(vs)),
												 k_(k)
		{
			crossover_dist = std::uniform_int_distribution<size_t>(0, crossover_.size() - 1);
			
			double sum = 0;
			for(double v : vs_)
			{
				sum += v;
			}
			
			mut_distribution = std::uniform_real_distribution<double>(0, sum);
		}
		
		template<typename Loss>
		double train(nn_model<Loss>& model,
					 const matrix& samples,
					 const matrix& labels,
					 int mortality,
					 population& pop)
		{
			
			for(int i = 0; i < mortality; ++i)
			{
				auto tourney_res = selection::k_tournament(pop, k_);
				random_crossover(*tourney_res.mom, *tourney_res.dad, *tourney_res.dead, pop.single_solution_size());
				
				auto& child = *tourney_res.dead;
				
				random_mutation(child, pop.single_solution_size());
				
				util::set_solution(child, model);
				child.penalty = model.error(labels, samples);
				
				if(child.penalty < pop.best->penalty) pop.best = &child;
			}
			
			util::set_solution(*pop.best, model);
			return pop.best->penalty;
		}
		
		private:
			std::vector<mutation_t> mutate_;
			std::vector<crossover_t> crossover_;
			std::vector<double> vs_;
			std::uniform_real_distribution<double> mut_distribution;
			std::uniform_int_distribution<size_t> crossover_dist;
			int k_;
			
			
			void random_crossover(const genetic_solution&, const genetic_solution&, genetic_solution&, size_t);
			
			void random_mutation(genetic_solution&, size_t);
			
	};

//    genetic_algorithm_optimizer make_gen_alg_optimizer(config&, size_t);
};

#endif
