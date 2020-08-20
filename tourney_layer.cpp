#include "tourney_layer.h"

#include <cassert>
#include <cmath>

namespace nn
{
	matrix tourney_layer::infer(const matrix& input) const
	{
		matrix result(input.rows(), weights_.columns());
		size_t inputs = input.rows();
		size_t half = weights_.rows() / 2;
		size_t columns = result.columns();
		#pragma omp parallel for collapse(2)
		for(size_t input_i = 0; input_i < inputs; ++input_i)
		{
			for(size_t column = 0; column < columns; ++column)
			{
				double sum = 0.0;
				#pragma omp simd reduction(+:sum)
				for(size_t i = 0; i < half; ++i)
				{
					sum += std::abs(input[input_i][i] - weights_[i][column]) / std::abs(weights_[i + half][column]);
				}
				
				result[input_i][column] = 1.0 / (1.0 + sum);
			}
		}
		
		return result;
	}
	
	matrix tourney_layer::forward(matrix input)
	{
		return infer(input);
	}
	
	matrix tourney_layer::backward_inputs(matrix)
	{
//		assert(false && "No backward_inputs method for tourney layer - The function has no gradient");
		return matrix();
	}
	
	std::pair<matrix, matrix> tourney_layer::backward()
	{
//		assert(false && "No backward method for tourney layer - The function has no gradient");
		return {matrix(), matrix()};
	}
	
	matrix& tourney_layer::get_bias()
	{
		return bias_;
	}
	
	matrix& tourney_layer::get_weights()
	{
		return weights_;
	}
	
	size_t tourney_layer::outputs() const noexcept
	{
		return weights_.columns();
	}
};
