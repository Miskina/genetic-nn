#ifndef QUADRATIC_LOSS_H
#define QUADRATIC_LOSS_H

#include "matrix.h"


namespace nn
{
	
	struct quadratic_loss
	{
		matrix operator()(const matrix&, matrix) const noexcept;
		
		matrix grad(const matrix&, matrix) const noexcept;
	};
	
};

#endif
