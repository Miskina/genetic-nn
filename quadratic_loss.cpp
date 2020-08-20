#include "quadratic_loss.h"

namespace nn
{
    matrix quadratic_loss::operator()(const matrix& y_true, matrix y_pred) const noexcept
    {

        for (size_t i = 0, n = y_pred.size(); i < n; ++i)
        {
//            printf("Predicted: %f, Expected: %f\n", y_pred.data()[i], y_true.data()[i]);
            y_pred.data()[i] -= y_true.data()[i];
            y_pred.data()[i] *= y_pred.data()[i];
        }

        return y_pred;
    }

    matrix quadratic_loss::grad(const matrix& y_true, matrix y_pred) const noexcept
    {
        return (1.f / y_pred.rows()) * (y_pred -= y_true);
    }
};
