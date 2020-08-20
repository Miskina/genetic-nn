#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include "matrix.h"

namespace nn
{
	
	namespace activation
	{
		struct sigmoid
		{
            matrix operator()(matrix) const noexcept;
			
			matrix grad(matrix) const noexcept;

		};
	};
	
};


#endif
