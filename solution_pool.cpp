#include "solution_pool.h"

solution_pool::~solution_pool() noexcept
{
	if(solution_pool_) delete [] solution_pool_;
	
	if(data_pool_) delete [] data_pool_;
}

genetic_solution * solution_pool::data() noexcept
{
	return solution_pool_;
}

const genetic_solution * solution_pool::data() const noexcept
{
	return solution_pool_;
}

size_t solution_pool::pool_size() const noexcept
{
	return pool_size_;
}

size_t solution_pool::solution_size() const noexcept
{
	return solution_size_;
}
