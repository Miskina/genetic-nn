#ifndef MUTATIONS_H
#define MUTATIONS_H

#include <random>

#include "genetic_solution.h"
#include "util.h"

namespace mutations
{	
	
	template<typename Op>
	struct generic_mutate
	{
		generic_mutate(double mutation_prob, double stddev, Op op) : d(0.0, stddev), prob_(mutation_prob), op_(std::move(op)) {}
		
		void operator()(genetic_solution& s, size_t size)
		{
			for(size_t i = 0; i < size; ++i)
			{
				if(util::rnd::zero_one_rnd() <= prob_)
				{
					op_(s[i], d(util::rnd::mt_eng));
				}
			}
		}
		
		private:
			std::normal_distribution<> d;
			double prob_;
			Op op_;
	};

	template<typename Operation>
	generic_mutate<Operation> make_mutation(double mutation_prob, double stddev, Operation op)
    {
	    return generic_mutate<Operation>(mutation_prob, stddev, std::move(op));
    }
	
};

#endif
