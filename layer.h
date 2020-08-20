#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

#include <utility>

namespace nn
{
	
	struct layer
	{
		virtual matrix infer(const matrix&) const = 0;
		
		virtual matrix forward(matrix) = 0;
		
		virtual matrix backward_inputs(matrix) = 0;
		
		virtual std::pair<matrix, matrix> backward() = 0;

		virtual matrix& get_bias() = 0;
		
		virtual matrix& get_weights() = 0;
		
		virtual size_t outputs() const noexcept = 0;
		
		virtual ~layer() = default;
	};
	
};

#endif
