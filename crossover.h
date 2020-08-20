#ifndef CROSSOVER_H
#define CROSSOVER_H

#include <random>

#include "genetic_solution.h"
#include "util.h"

namespace crossover
{
	struct simple_arithmetic
	{
		simple_arithmetic(size_t solution_size) : d(0, solution_size) {}
		
		void operator()(const genetic_solution&, const genetic_solution&, genetic_solution&, size_t);
		
		private:
			std::uniform_int_distribution<size_t> d;
	};
	
	void discrete(const genetic_solution&, const genetic_solution&, genetic_solution&, size_t);
	
	void complete_arithmetic(const genetic_solution&, const genetic_solution&, genetic_solution&, size_t);
};

#endif
