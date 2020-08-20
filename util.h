#ifndef UTIL_H
#define UTIL_H

#include <random>
#include <limits>

#include "nn_model.h"
#include "genetic_solution.h"
#include "dataset.h"

namespace util
{
	
	namespace rnd
	{
		static inline std::mt19937 mt_eng{std::random_device{}()};
		static inline std::uniform_real_distribution<> zero_one_dstrb(0.0, 1.0);
		
		double zero_one_rnd();
	};
	
	template<typename Loss>
	void set_solution(genetic_solution& solution, nn::nn_model<Loss>& model)
	{
		size_t pos = 0;
		for(size_t i = 0, n = model.layer_num(); i < n; ++i)
		{
			auto& layer_ptr = model[i];
			auto& layer_weights = layer_ptr->get_weights();
			auto& layer_bias = layer_ptr->get_bias();
			
			layer_weights.data() = nullptr;
			layer_bias.data() = nullptr;
			
			layer_weights = matrix(layer_weights.rows(), layer_weights.columns(), solution.data() + pos);
			pos += layer_weights.size();
			
			if(layer_bias.size() == 0) continue;
			layer_bias = matrix(layer_bias.rows(), layer_bias.columns(), solution.data() + pos);
			pos += layer_bias.size();
		}
	}
	
};

#endif
