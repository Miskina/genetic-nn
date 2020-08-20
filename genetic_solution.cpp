#include "genetic_solution.h"

	
double genetic_solution::operator[](size_t pos) const
{
	return data_[pos];
}

double& genetic_solution::operator[](size_t pos)
{
	return data_[pos];
}

double * genetic_solution::data() noexcept
{
	return data_;
}

const double * genetic_solution::data() const noexcept
{
	return data_;
}

