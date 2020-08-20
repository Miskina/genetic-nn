#ifndef NN_MODEL_H
#define NN_MODEL_H

#include "layer.h"
#include "dataset.h"

#include <memory>
#include <vector>

namespace nn
{
	
	template<typename LossFunction>
	struct nn_model
	{
	    
		matrix predict(const matrix& input) const
		{
			matrix output = input;
			for(size_t i = 0, n = this->layers.size(); i < n; ++i)
			{
				output = this->layers[i]->infer(std::move(output));
			}

			return output;
		}
		
		
		matrix forward_pass(matrix input)
		{
			for(size_t i = 0, n = this->layers.size(); i < n; ++i)
			{
				input = this->layers[i]->forward(std::move(input));
			}

			return input;
		}
		
		
		size_t layer_num() const noexcept
		{
			return layers.size();
		}
		
		const std::unique_ptr<layer>& operator[](int n) const
		{
			return layers[n];
		}
		
		std::unique_ptr<layer>& operator[](int n)
		{
			return layers[n];
		}
		
		
		matrix loss_grad(const matrix& y_true, matrix y_predicted) noexcept
		{
			return loss.grad(y_true, std::move(y_predicted));
		}
		
		double error(const matrix& true_values, const matrix& samples)
		{
			return (1.0 / samples.rows()) * matrix::asum(loss(true_values, predict(samples)));
		}

		double cost(const matrix& true_values, const matrix& predicted)
        {
            return (1.0 / predicted.rows()) * matrix::asum(loss(true_values, predicted));
        }
		
		nn_model& operator<<(std::unique_ptr<layer>&& layer_to_add)
		{
			layers.push_back(std::move(layer_to_add));
			return *this;
		}

        std::vector<std::unique_ptr<layer>>::reverse_iterator layers_rbegin() noexcept
        {
		    return layers.rbegin();
        }

        std::vector<std::unique_ptr<layer>>::reverse_iterator layers_rend() noexcept
        {
            return layers.rend();
        }
        
        void print_all_weights_and_bias() const noexcept
        {
        	for(int i = 0, n = layers.size(); i < n; ++i)
        	{
        		printf("\nWeights for layer %d\n", i + 1);
        		print(layers[i]->get_weights());
        		printf("\nBiases for layer %d\n", i + 1);
        		print(layers[i]->get_bias());
        		printf("\n");
			}
		}
		
		private:
			std::vector<std::unique_ptr<layer>> layers;
			LossFunction loss;
	};

};

#endif
