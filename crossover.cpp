#include "crossover.h"

#include <openBLAS/cblas.h>

namespace crossover
{
	
	void simple_arithmetic::operator()(const genetic_solution& mom, const genetic_solution& dad, genetic_solution& child, size_t n)
	{
		auto pos = d(util::rnd::mt_eng);
		
		cblas_dcopy(n, mom.data(), 1, child.data(), 1);
		cblas_daxpy(n - pos, 1.0, dad.data() + pos, 1, child.data() + pos, 1);
		cblas_dscal(n - pos, 0.5, child.data() + pos, 1);
		
	}
	
	void discrete(const genetic_solution& mom, const genetic_solution& dad, genetic_solution& child, size_t n)
	{
		
		for(size_t i = 0; i < n; ++i)
		{
			if(util::rnd::zero_one_rnd() > 0.5)
			{
				child[i] = dad[i];
			}
			else
			{
				child[i] = mom[i];
			}
		}
	}
	
	void complete_arithmetic(const genetic_solution& mom, const genetic_solution& dad, genetic_solution& child, size_t n)
	{
		cblas_dcopy(n, mom.data(), 1, child.data(), 1);
		cblas_daxpy(n, 1.0, dad.data(), 1, child.data(), 1);
		cblas_dscal(n, 0.5, child.data(), 1);
	}
		
};
