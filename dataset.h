#ifndef DATASET_H
#define DATASET_H

#include <utility>
#include <random>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <openBLAS/cblas.h>
#include <sstream>

#include "matrix.h"

namespace nn
{

    struct dataset
    {

        dataset(std::initializer_list<std::pair<std::initializer_list<double>, std::initializer_list<double>>> l)
                : samples_(l.size(), l.begin()->first.size()), labels_(l.size(), l.begin()->second.size())
        {
            size_t i = 0;
            for (auto& [in, out] : l)
            {
                cblas_dcopy(samples_.columns(), in.begin(), 1, samples_.data() + i * samples_.columns(), 1);
                cblas_dcopy(labels_.columns(), out.begin(), 1, labels_.data() + i * labels_.columns(), 1);
                ++i;
            }
        }

        dataset(matrix&& samples, matrix&& labels) : samples_(std::move(samples)), labels_(std::move(labels)) {}

		static dataset load(const char * file_name, size_t sample_features, size_t label_features)
		{
			std::ifstream file(file_name);
			
			if(!file || !file.is_open())
			{
				throw std::runtime_error("Could not open specified dataset file!\n");
			}
			
			std::vector<std::vector<double>> sample_vec;
			std::vector<std::vector<double>> label_vec;
			
			std::string line;
			
			while(std::getline(file, line))
			{
				std::vector<double> sample(sample_features);
				std::vector<double> label(label_features);
				
				if(line.empty()) break;
				
				std::istringstream ss(line);
				
				float val;
				for(size_t i = 0; i < sample_features; ++i)
				{
					ss >> val;
					sample[i] = val;
				}
				
				for(size_t i = 0; i < label_features; ++i)
				{
					ss >> val;
					label[i] = val;
				}
				
				sample_vec.push_back(std::move(sample));
				label_vec.push_back(std::move(label));
			}
			
			if(sample_vec.empty() || label_vec.empty()) throw std::runtime_error("No data in given file, failed to load dataset\n");
			if(sample_vec.size() != label_vec.size()) throw std::runtime_error("Missing samples or labels in given dataset file, failed to load dataset\n");
			
			matrix sample_m(sample_vec.size(), sample_features);
			matrix label_m(label_vec.size(), label_features);
			
			for(size_t i = 0, n = sample_vec.size(); i < n; ++i)
			{
				cblas_dcopy(sample_m.columns(), sample_vec[i].data(), 1, sample_m.data() + i * sample_m.columns(), 1);
				cblas_dcopy(label_m.columns(), label_vec[i].data(), 1, label_m.data() + i * label_m.columns(), 1);
			}
			
			return dataset(std::move(sample_m), std::move(label_m));
		}
		
        struct batch_iterator
        {
            batch_iterator(const dataset& data_set, size_t batch_size) : data(data_set), batch(batch_size)
            {}

//			~batch_iterator() noexcept;

            bool has_next() const noexcept;

            operator bool() const noexcept
            {
                return has_next();
            }

            std::pair<matrix, matrix> next_batch();

        private:
            const dataset& data;
            size_t processed = 0;
            size_t batch;
        };
        
        struct random_batch_iterator
        {
        	random_batch_iterator(const dataset& data_set, size_t batch_size) : set(data_set), batch(batch_size), d(0, batch_size) {}
			
			std::pair<matrix, matrix> next_batch();
			
			private:
				const dataset& set;
				size_t batch;
				std::uniform_int_distribution<size_t> d;
		};
        
        batch_iterator get_batch_iterator(size_t batch_size) const noexcept;
		
		random_batch_iterator get_random_iterator(size_t batch_size) const noexcept;
		
        size_t size() const noexcept;

        size_t features() const noexcept;

//		std::pair<matrix, matrix> get_sample_batch(size_t) const;

        const matrix& samples() const noexcept;

        const matrix& labels() const noexcept;

    private:
        matrix samples_;
        matrix labels_;
    };


};
#endif
