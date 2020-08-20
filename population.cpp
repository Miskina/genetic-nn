#include "population.h"

#include <cassert>

//void population::initialize_params(const std::vector<init_pair>& initalizers)
//{
//	for(size_t i = 0, n = solutions_.pool_size(); i < n; ++i)
//	{
//		auto& solution = solutions_.data()[i];
//		size_t pos = 0;
//		for(auto& [n, init] : initializers)
//		{
//			init(solution.data() + pos, n);
//			pos += n;
//		}
//	}
//}

genetic_solution& population::operator[](size_t pos)
{
	assert(pos < solutions_.pool_size() && "Cannot specify position out of bounds of population");
	
	return solutions_.data()[pos];
}

const genetic_solution& population::operator[](size_t pos) const
{
	assert(pos < solutions_.pool_size() && "Cannot specify position out of bounds of population");
	
	return solutions_.data()[pos];
}

std::size_t population::size() const noexcept
{
	return solutions_.pool_size();
}

genetic_solution& population::random_solution()
{
	return solutions_.data()[solution_d(util::rnd::mt_eng)];
}

//const genetic_solution& population::random_solution()
//{
//	return solutions_.data()[solution_d(util::rnd::mt_eng)];
//}

size_t population::single_solution_size() const noexcept
{
    return solutions_.solution_size();
}
