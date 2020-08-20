#ifndef SOLUTION_POOL_H
#define SOLUTION_POOL_H

#include <vector>

#include "genetic_solution.h"


struct solution_pool
{
	explicit solution_pool(size_t solution_size,
						   size_t population_size) : data_pool_(new double[solution_size * population_size]),
													 solution_pool_(new genetic_solution[population_size]),
													 pool_size_(population_size),
													 solution_size_(solution_size)
	{
		for(size_t i = 0; i < population_size; ++i)
		{
			solution_pool_[i] = genetic_solution(data_pool_ + i * solution_size);
		}
	}
	
	~solution_pool() noexcept;
	
	genetic_solution * data() noexcept;
	
	const genetic_solution * data() const noexcept;
	
	size_t pool_size() const noexcept;
	
	size_t solution_size() const noexcept;
	
	private:
		double * data_pool_;
		genetic_solution * solution_pool_;
		size_t pool_size_;
		size_t solution_size_;
};

#endif
