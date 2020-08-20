#ifndef TOURNEY_LAYER_H
#define TOURNEY_LAYER_H

#include "layer.h"

namespace nn
{
	struct tourney_layer : public layer
	{
		template<typename WeightInitializer>
	    tourney_layer(int inputs,
	                  int outputs,
	                  WeightInitializer init_weights) : weights_(inputs, outputs)
	    {
	        init_weights(weights_);
	    }
	    
	    tourney_layer(matrix&& weights) noexcept : weights_(std::move(weights)) {}
	
	    tourney_layer(tourney_layer&& o) noexcept : weights_(std::move(o.weights_)) {}
		
		matrix infer(const matrix&) const override;
		
		matrix forward(matrix) override;
		
		matrix backward_inputs(matrix) override;
		
		std::pair<matrix, matrix> backward() override;
		
		matrix& get_bias() override;
		
		matrix& get_weights() override;
		
		size_t outputs() const noexcept override;
		
		private:
			matrix weights_;
			matrix bias_{};
	};
};

#endif
