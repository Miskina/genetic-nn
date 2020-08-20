#ifndef GENETIC_SOLUTION_H
#define GENETIC_SOLUTION_H

#include <cstddef>

struct genetic_solution
{
	genetic_solution() noexcept : data_(nullptr) {}
	
	genetic_solution(double * data) noexcept : data_(data) {}
	
	double& operator[](size_t);
	
	double operator[](size_t) const;
	
	double * data() noexcept;
	
	const double * data() const noexcept;
	
	double penalty = 0.0;		
	
	private:
		double * data_;
		
};

#endif
