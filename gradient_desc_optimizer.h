#ifndef GRADIENT_DESC_OPTIMIZER_H
#define GRADIENT_DESC_OPTIMIZER_H

#include <cstddef>

#include "nn_model.h"
#include "dataset.h"

namespace nn
{

    struct gradient_desc_optimizer
    {
        constexpr gradient_desc_optimizer(double learning_rate, size_t batch_size = 1) : batch(batch_size),
                                                                                         learn_rate(learning_rate) {}

        constexpr void set_batch_size(size_t batch_size) noexcept
        {
            batch = batch_size;
        }

        constexpr size_t batch_size() const noexcept
        {
            return batch;
        }

        constexpr void set_learn_rate(double learning_rate) noexcept
        {
            learn_rate = learning_rate;
        }

        constexpr double learning_rate() const noexcept
        {
            return learn_rate;
        }

        template<typename Loss>
        double train(nn_model<Loss>& model, const matrix& samples, const matrix& labels)
        {
            auto output = model.forward_pass(samples);

            auto& last_layer = *model.layers_rbegin();

            auto loss = model.cost(labels, output);

            //EAo -> y_true - y_pred
            auto delta = model.loss_grad(labels, std::move(output));

            // delta(net)/delta(h)
            // h - izlaz iz trenutnog sloja
            delta = last_layer->backward_inputs(std::move(delta));

            // (delta(h)/delta(w), delta(h)/delta(b))
            auto[w_grads, b_grads] = last_layer->backward();

            for (auto curr_layer_it = ++model.layers_rbegin(), prev_layer_it = model.layers_rbegin();
                 curr_layer_it != model.layers_rend();
                 ++curr_layer_it, ++prev_layer_it)
            {
                auto& curr_layer = *curr_layer_it;
                auto& prev_layer = *prev_layer_it;
                delta = curr_layer->backward_inputs(std::move(delta));
                auto curr_grad_pair = curr_layer->backward();

                prev_layer->get_weights().add(w_grads, -learn_rate);
                prev_layer->get_bias().add(b_grads, -learn_rate);

                w_grads = std::move(curr_grad_pair.first);
                b_grads = std::move(curr_grad_pair.second);
            }

            model[0]->get_weights().add(w_grads, -learn_rate);
            model[0]->get_bias().add(b_grads, -learn_rate);
			
            return loss;
        }

    private:
        size_t batch;
        double learn_rate;
    };

};

#endif
