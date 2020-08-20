#include "genetic_algorithm_optimizer.h"

namespace nn
{
	void genetic_algorithm_optimizer::random_crossover(const genetic_solution& mom, const genetic_solution& dad, genetic_solution& child, size_t size)
	{
		crossover_[crossover_dist(util::rnd::mt_eng)](mom, dad, child, size);
	}
	
	void genetic_algorithm_optimizer::random_mutation(genetic_solution& solution, size_t size)
	{
		auto mut_rand = mut_distribution(util::rnd::mt_eng);
		double prob_sum = 0.0;
		for(int i = 0, n = vs_.size(); i < n; ++i)
		{
			prob_sum += vs_[i];
			if(prob_sum >= mut_rand)
			{
				mutate_[i](solution, size);
				break;
			}
		}
	}

//	genetic_algorithm_optimizer make_gen_alg_optimizer(config& cfg, size_t solution_size)
//    {
//	    const auto& t_vec = cfg.ts();
//	    std::vector<double> v(t_vec.size());
//	    double sum = 0;
//	    for(size_t i = 0, n = t_vec.size(); i < n; ++i)
//        {
//	        sum += t_vec[i];
//        }
//
//	    for(size_t i = 0, n = v.size(); i < n; ++i)
//        {
//	        v[i] = t_vec[i] / sum;
//        }
//		
//		const auto& pm_vec = cfg.pm();
//		const auto& stddev = cfg.stddev();
//	    std::vector<mutation_t> mutations(stddev.size());
//	    
//	    size_t mut_num = mutations.size() - 1;
//	    for(size_t i = 0; i < mut_num; ++i)
//	    {
//	    	mutations[i] = mutations::make_mutation(pm_vec[i], stddev[i], [](double& solution_elem, double rand)
//																		  {
//																				solution_elem += rand;
//																		  });
//		}
//		mutations[mut_num] = mutations::make_mutation(pm_vec[mut_num], stddev[mut_num], [](double& solution_elem, double rand)
//																						{
//																							solution_elem = rand;	
//																						});
//		
//		
//		std::vector<crossover_t> crossovers(3);
//		crossovers[0] = crossover::simple_arithmetic(solution_size);
//		crossovers[1] = crossover::discrete;
//		crossovers[2] = crossover::complete_arithmetic;
//		
//		return genetic_algorithm_optimizer(std::move(mutations), std::move(crossovers), std::move(v), cfg.k());
//    }
};
