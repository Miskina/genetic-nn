#ifndef POPULATION_H
#define POPULATION_H

#include "util.h"
#include "genetic_solution.h"
#include "solution_pool.h"

#include <vector>
#include <utility>
#include <functional>


struct population
{
	
	explicit population(size_t population_size, size_t solution_size) : solutions_(solution_size, population_size),
																		solution_d(0, population_size - 1) {}
	
	genetic_solution& operator[](size_t);
	
	const genetic_solution& operator[](size_t) const;
	
	std::size_t size() const noexcept;
	
	genetic_solution& random_solution();

	std::size_t single_solution_size() const noexcept;

//	const genetic_solution& random_solution() const;
	
	genetic_solution * best;
	
	template<typename Loss>
	static void init_penalties(population& pop, nn::nn_model<Loss>& model, const nn::dataset& set)
	{
		genetic_solution * min = nullptr;
		for(size_t i = 0, n = pop.size(); i < n; ++i)
		{
			util::set_solution(pop[i], model);
			pop[i].penalty = model.error(set.labels(), set.samples());
			if(min == nullptr || pop[i].penalty < min->penalty)
			{
				min = &pop[i];
			}
		}
		pop.best = min;
	}
	
	using init_pair = std::pair<size_t, std::function<void(double *, size_t)>>;
	
	static void initialize_params(const std::vector<init_pair>& initializers, population& pop)
	{
		for(size_t i = 0, n = pop.size(); i < n; ++i)
		{
			auto& solution = pop[i];
			size_t pos = 0;
			for(auto& [size, init] : initializers)
			{
				init(solution.data() + pos, size);
				pos += size;
			}
		}
	}
	
	private:
		solution_pool solutions_;
		std::uniform_int_distribution<size_t> solution_d;
		
};

#endif
