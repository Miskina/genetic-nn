#include "dataset.h"
#include "util.h"

namespace nn
{

    dataset::batch_iterator dataset::get_batch_iterator(size_t batch_size) const noexcept
    {
        return batch_iterator(*this, batch_size);
    }
    
    dataset::random_batch_iterator dataset::get_random_iterator(size_t batch_size) const noexcept
    {
    	return random_batch_iterator(*this, batch_size);
	}

    const matrix& dataset::samples() const noexcept
    {
        return samples_;
    }

    const matrix& dataset::labels() const noexcept
    {
        return labels_;
    }

    size_t dataset::size() const noexcept
    {
        return samples_.rows();
    }

    size_t dataset::features() const noexcept
    {
        return samples_.columns();
    }


    bool dataset::batch_iterator::has_next() const noexcept
    {
        return this->data.size() > this->processed;
    }

    std::pair<matrix, matrix> dataset::batch_iterator::next_batch()
    {
        size_t left_samples = data.size() - processed;
        size_t batch_size = left_samples < batch ? left_samples : batch;

        auto& samples = data.samples();
        auto& labels = data.labels();

        matrix sample_batch(batch_size, samples.columns());
        matrix label_batch(batch_size, labels.columns());

        cblas_dcopy(sample_batch.columns() * batch_size, samples.data() + processed * sample_batch.columns(), 1, sample_batch.data(), 1);
        cblas_dcopy(label_batch.columns() * batch_size, labels.data() + processed * label_batch.columns(), 1, label_batch.data(), 1);

        processed += batch_size;

        return {std::move(sample_batch), std::move(label_batch)};
    }
    
    std::pair<matrix, matrix> dataset::random_batch_iterator::next_batch()
    {
    	auto& samples = set.samples();
    	auto& labels = set.labels();
    	
//    	if(batch == samples.rows())
//    	{
//    		return {samples, labels};
//		}
    	
    	matrix sample_batch(batch, samples.columns());
    	matrix label_batch(batch, labels.columns());
    	
    	for(size_t i = 0; i < batch; ++i)
    	{
    		size_t row = d(util::rnd::mt_eng);
    		cblas_dcopy(sample_batch.columns(), samples.data() + row * samples.columns(), 1, sample_batch.data() + i * sample_batch.columns(), 1);
    		cblas_dcopy(label_batch.columns(), labels.data() + row * labels.columns(), 1, label_batch.data() + i * label_batch.columns(), 1);
		}
		
		
		return {std::move(sample_batch), std::move(label_batch)};
	}

};
