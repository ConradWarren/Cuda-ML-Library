#pragma once

#include <vector>

enum class activation_functions{Linear ,Rectified_Linear, Sigmoid, Softmax};

class layer {

public:

	size_t neurons;
	size_t inputs;
	size_t batch_size;
	double* forward_output;
	
	void virtual forward(const std::vector<std::vector<double>>& batched_inputs) = 0;
	void virtual forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) = 0;
	void virtual forward(double* batched_inputs, size_t input_size, size_t batch_size) = 0;
	void virtual forward(const layer* prev_layer) = 0;

	double virtual loss(std::vector<std::vector<double>>& batched_targets) const = 0;
	double virtual loss(std::vector<int>& batched_targets) const = 0;
	double virtual loss(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const = 0;

};

class dense_layer : public layer {
	
public:

	double* weights;
	double* bias;
	
	dense_layer();
	dense_layer(size_t _inputs, size_t _neurons);
	~dense_layer();

	void virtual forward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual forward(double* batched_inputs, size_t input_size, size_t batch_size) override;
	void virtual forward(const layer* prev_layer) override;

	double virtual loss(std::vector<std::vector<double>>& batched_targets) const override;
	double virtual loss(std::vector<int>& batched_targets) const override;
	double virtual loss(std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const override;
};

