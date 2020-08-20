#include "initializers.h"

namespace nn
{
	namespace initializers
	{
//		void zero_init(matrix& m)
//		{
//			cblas_dscal(m.size(), 0.0, m.data(), 1);
//		}
		
		void zero_init(double * data, size_t size)
		{
			cblas_dscal(size, 0.0, data, 1);
		}
		
		void random_init::operator()(matrix& mtrx) noexcept
        {
	        for(size_t i = 0, n = mtrx.rows(); i < n; ++i)
            {
	            for(size_t j = 0, m = mtrx.columns(); j < m; ++j)
                {
	                mtrx[i][j] = d(util::rnd::mt_eng);
                }
            }
        }
        
        void random_init::operator()(double * data, size_t size)
        {
        	for(size_t i = 0; i < size; ++i)
        	{
        		data[i] = d(util::rnd::mt_eng);
			}
		}
        
        void xavier_init::operator()(matrix& mtrx) noexcept
		{
			for(size_t i = 0, n = mtrx.rows(); i < n; ++i)
			{
				for(size_t j = 0, m = mtrx.columns(); j < m; ++j)
				{
					mtrx[i][j] = d(util::rnd::mt_eng);
				}
			}
		}
		
		void xavier_init::operator()(double * data, size_t size)
		{
			for(size_t i = 0; i < size; ++i)
			{
				data[i] = d(util::rnd::mt_eng);
			}
		}
	};	
};
