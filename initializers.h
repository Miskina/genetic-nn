#ifndef INITIALIZERS_H
#define INITIALIZERS_H

#include <random>
#include <cmath>

#include "matrix.h"
#include "util.h"

namespace nn
{
	
	namespace initializers
	{
//		void zero_init(matrix&);
		
		void zero_init(double *, size_t);

		struct random_init
        {
		    random_init(double upper, double lower) : d(upper, lower) {}

		    void operator()(matrix&) noexcept;
		    
		    void operator()(double *, size_t);

            private:
		    std::uniform_real_distribution<double> d;
        };
        
        struct xavier_init
        {
        	xavier_init(int input_nodes, int output_nodes) : d(0.0, std::sqrt(1.0 / ((input_nodes + output_nodes) / 2.0))) {}
			
			void operator()(matrix&) noexcept;
			
			void operator()(double *, size_t);
			
			private:
				std::normal_distribution<double> d;
		};
	};
	
};

#endif
