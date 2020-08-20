#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "matrix.h"
#include "layer.h"

namespace nn
{

    template<typename ActivationFunction>
    struct fully_connected_layer : public layer
    {

        template<typename WeightInitializer, typename BiasInitializer>
        fully_connected_layer(int inputs,
                              int outputs,
                              ActivationFunction activation,
                              WeightInitializer init_weights,
                              BiasInitializer init_bias) : f(std::move(activation)),
                                                           weights_(inputs, outputs),
							  							   bias_(1, outputs),
                                                           inputs_()
        {
            init_weights(weights_);
            init_bias(bias_);
        }
        
        fully_connected_layer(ActivationFunction func,
							  matrix&& weights,
							  matrix&& bias) noexcept : weights_(std::move(weights)),
							  							bias_(std::move(bias)),
														f(std::move(func)),
														inputs_() {}

        fully_connected_layer(fully_connected_layer&& o) noexcept : bias_(std::move(o.bias_)),
                                                                    weights_(std::move(o.weights_)), f(std::move(o.f)),
                                                                    inputs_(std::move(o.inputs_)) {}

        fully_connected_layer(const fully_connected_layer&) = delete;

        fully_connected_layer& operator=(fully_connected_layer&& o) noexcept
        {
            weights_ = std::move(o.weights_);
            inputs_ = std::move(o.inputs_);
            bias_ = std::move(o.bias_);
            f = std::move(o.f);
        }

        matrix infer(const matrix& input) const override
        {
            return f(matrix::fma(input, weights_, bias_));
        }

        matrix forward(matrix input) override
        {
            inputs_ = std::move(input);
            f_grads_ = f(matrix::fma(inputs_, weights_, bias_));
            return f_grads_;
        }

        matrix backward_inputs(matrix delta) override
        {
        	f_grads_ = f.grad(std::move(f_grads_));
            f_grads_ *= delta;
            return matrix::matmul<false, true>(f_grads_, weights_);
        }

        std::pair<matrix, matrix> backward() override
        {
            matrix delta_w = matrix::matmul<true, false>(inputs_, f_grads_);
            matrix delta_b = matrix::sum<0>(f_grads_);
            // Free unnecessary memory.
            inputs_ = matrix();
            f_grads_ = matrix();
            return {std::move(delta_w), std::move(delta_b)};
        }

        matrix& get_bias() override
        {
            return this->bias_;
        }

        matrix& get_weights() override
        {
            return this->weights_;
        }

        size_t outputs() const noexcept override
        {
            return bias_.rows();
        }

    private:
        ActivationFunction f;
        matrix weights_;
        matrix bias_;
        matrix inputs_;
        matrix f_grads_;
    };

};
#endif
