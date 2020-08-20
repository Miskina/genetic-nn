#include "config.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <regex>
#include <cstdlib>
#include <unordered_map>

config::config(const char * file_name)
{
	std::ifstream file(file_name);
	if(!file || !file.is_open()) throw std::runtime_error("Could not open config file!");
	
	std::unordered_map<std::string, std::string> helper_map;
	
	std::string line;
	
	while(std::getline(file, line))
	{
		if(line.empty() || line[0] == '#') continue;
		
		if(line[0] == '#') continue;
		
		size_t start_pos = 0;
		while(line[start_pos] == ' ')
		{
			++start_pos;
		}
		
		size_t end = line.length() - 1;
		
		while(line[end] == ' ') --end;
		
		size_t current_pos = start_pos + 1;
		while(current_pos <= end)
		{
			
			if(line[current_pos] == '=')
			{
				helper_map[line.substr(start_pos, current_pos - start_pos)] = line.substr(current_pos + 1, end - current_pos);
				break;
			}
			++current_pos;
		}
	}
	
	
	input_features_ = std::stoull(helper_map.at("input_features"));
	output_features_ = std::stoull(helper_map.at("output_features"));
	
	dataset_ = helper_map.at("dataset_loc");
	
	auto epoch_it = helper_map.find("epochs");
	
	if(epoch_it != helper_map.end())
	{
		epochs_ = std::stoull(epoch_it->second);
	}
	
	auto pop_size_it = helper_map.find("population_size");
	
	if(pop_size_it != helper_map.end())
	{
		population_size_ = std::stoull(pop_size_it->second);
	}
	else
	{
		population_size_ = 50;
	}
	
	auto& hidden_layers_str = helper_map.at("hidden_layers");
	
	const static std::regex num_rgx("\\d+");
	
	for(auto it = std::sregex_token_iterator(hidden_layers_str.begin(), hidden_layers_str.end(), num_rgx);
		it != std::sregex_token_iterator();
		++it)
	{

		hidden_layers.push_back(std::stoi(it->str()));
	}
	
	k_ = 3;
	auto find_k = helper_map.find("K");
	if(find_k != helper_map.end())
	{
		k_ = std::stoi(find_k->second);
	}
	
	mortality_ = std::stoi(helper_map.at("mortality"));
	
	const static std::regex real_num_rgx("(\\d+\\.\\d*|\\.\\d+|\\d+)");
	
	auto& t_str = helper_map.at("T");
	
	for(auto it = std::sregex_token_iterator(t_str.begin(), t_str.end(), real_num_rgx);
		it != std::sregex_token_iterator();
		++it)
	{
		ts_.push_back(std::stod(it->str()));
	}
	
	auto& pm_str = helper_map.at("pm");
	
	for(auto it = std::sregex_token_iterator(pm_str.begin(), pm_str.end(), real_num_rgx);
		it != std::sregex_token_iterator();
		++it)
	{
		pm_.push_back(std::stod(it->str()));
	}
	
	auto& stddev_str = helper_map.at("stddev");
	
	for(auto it = std::sregex_token_iterator(stddev_str.begin(), stddev_str.end(), real_num_rgx);
		it != std::sregex_token_iterator();
		++it)
	{
		stddev_.push_back(std::stod(it->str()));
	}
	
	if(stddev_.size() != pm_.size()) throw std::runtime_error("The config file must specify as many stddev variables as pm variables");
	
	solution_size_ = 0;
	solution_size_ += 2 * input_features_ * hidden_layers[0];
	sizes_.push_back({2 * input_features_, hidden_layers[0]});
	sizes_.push_back({0, 0});
	
	size_t n_layers = hidden_layers.size();
	
	for(size_t i = 1; i < n_layers; ++i)
	{
		solution_size_ += hidden_layers[i - 1] * hidden_layers[i];
		solution_size_ += hidden_layers[i];
		
		sizes_.push_back({hidden_layers[i - 1], hidden_layers[i]});
		sizes_.push_back({1, hidden_layers[i]});
	}
	
	solution_size_ += hidden_layers[n_layers - 1] * output_features_;
	solution_size_ += output_features_;
	
	sizes_.push_back({hidden_layers[n_layers - 1], output_features_});
	sizes_.push_back({1, output_features_});
	
	precision_ = 1e-7;
	auto precision_it = helper_map.find("precision");
	if(precision_it != helper_map.end())
	{
		precision_ = std::stod(precision_it->second);
	}

	log_point_ = 100;
	auto log_it = helper_map.find("log_point");
	if(log_it != helper_map.end())
    {
	    log_point_ = std::stoull(log_it->second);
    }
}

size_t config::input_features() const noexcept
{
	return this->input_features_;
}

size_t config::output_features() const noexcept
{
	return this->output_features_;
}

size_t config::epochs() const noexcept
{
	return this->epochs_;
}

const std::string& config::data_file() const noexcept
{
	return this->dataset_;
}

int config::mortality() const noexcept
{
	return mortality_;
}

int config::k() const noexcept
{
	return k_;
}

size_t config::population_size() const noexcept
{
	return population_size_;
}

std::vector<double>& config::ts() noexcept
{
	return ts_;
}

const std::vector<double>& config::ts() const noexcept
{
	return ts_;
}

std::vector<double>& config::pm() noexcept
{
	return pm_;
}

const std::vector<double>& config::pm() const noexcept
{
	return pm_;
}

std::vector<double>& config::stddev() noexcept
{
	return stddev_;
}

const std::vector<double>& config::stddev() const noexcept
{
	return stddev_;
}

size_t config::solution_size() const noexcept
{
	return solution_size_;
}

const std::vector<std::pair<std::size_t, std::size_t>>& config::weight_bias_sizes() const noexcept
{
	return sizes_;
}

void config::clear()
{
	hidden_layers.clear();
	hidden_layers.shrink_to_fit();
	dataset_.clear();
	ts_.clear();
	ts_.shrink_to_fit();
	pm_.clear();
	pm_.shrink_to_fit();
	stddev_.clear();
	stddev_.shrink_to_fit();
	sizes_.clear();
	sizes_.shrink_to_fit();
}

double config::precision() const noexcept
{
	return precision_;
}

size_t config::log_point() const noexcept
{
    return log_point_;
}
