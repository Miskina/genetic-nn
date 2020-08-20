#include "activation_functions.h"

#include <cmath>

namespace nn
{
    namespace activation
    {

        matrix sigmoid::operator()(matrix m) const noexcept
        {
            for (size_t i = 0, n = m.size(); i < n; ++i)
            {
                m.data()[i] = 1.0 / (1.0 + std::exp(-m.data()[i]));
            }
            return m;
        }

        matrix sigmoid::grad(matrix m) const noexcept
        {
            for (size_t i = 0, n = m.size(); i < n; ++i)
            {
                m.data()[i] = m.data()[i] * (1.0 - m.data()[i]);
            }
            
            return m;
        }
    };
};
