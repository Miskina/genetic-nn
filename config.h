#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <string_view>
#include <vector>

struct config
{	
	config(const char * file);
	
	size_t input_features() const noexcept;
	
	size_t output_features() const noexcept;
	
	size_t epochs() const noexcept;
	
	const std::vector<int>& hidden_layer_vec() const noexcept;
	
	const std::string& data_file() const noexcept;
	
	int mortality() const noexcept;
	
	int k() const noexcept;
	
	size_t population_size() const noexcept;
	
	std::vector<double>& ts() noexcept;
	
	const std::vector<double>& ts() const noexcept;
	
	std::vector<double>& pm() noexcept;
	
	const std::vector<double>& pm() const noexcept;
	
	std::vector<double>& stddev() noexcept;
	
	const std::vector<double>& stddev() const noexcept;
	
	std::size_t solution_size() const noexcept;
	
	const std::vector<std::pair<size_t, size_t>>& weight_bias_sizes() const noexcept;
	
	void clear();
	
	double precision() const noexcept;

	size_t log_point() const noexcept;
	
	private:
		size_t input_features_;
		size_t output_features_;
		size_t epochs_{10000};
		double precision_;
		std::vector<int> hidden_layers;
		std::string dataset_;
		int k_;
		int mortality_;
		size_t population_size_;
		std::vector<double> ts_;
		std::vector<double> pm_;
		std::vector<double> stddev_;
		std::size_t solution_size_;
		std::vector<std::pair<size_t, size_t>> sizes_;
		size_t log_point_;
	
};

#endif
