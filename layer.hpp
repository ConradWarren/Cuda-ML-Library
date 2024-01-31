#pragma once

#include <vector>

enum class activation_functions{Linear ,Rectified_Linear, Sigmoid, Softmax};

class layer {

public:

	size_t neurons;
	size_t inputs;
	size_t batch_size;
	double* forward_output;
	double* backward_input;
	activation_functions layer_activation_function;

	void virtual forward(const std::vector<std::vector<double>>& batched_inputs) = 0;
	void virtual forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) = 0;
	void virtual forward(double* batched_inputs, size_t input_size, size_t _batch_size) = 0;
	void virtual forward(const layer* prev_layer) = 0;

	double virtual loss(const std::vector<std::vector<double>>& batched_targets) const = 0;
	double virtual loss(const std::vector<int>& batched_targets) const = 0;
	double virtual loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const = 0;

	void virtual init_back_propigation(const std::vector<std::vector<double>>& batched_targets) = 0;
	void virtual init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) = 0;
	void virtual init_back_propigation(double* batched_targets, size_t input_size, size_t _batch_size) = 0;

	void virtual backward(const std::vector<std::vector<double>>& batched_inputs) = 0;
	void virtual backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) = 0;
	void virtual backward(double* batched_inputs, size_t input_size, size_t _batch_size) = 0;
	void virtual backward(layer* prev_layer) = 0;
};

class dense_layer : public layer {
	
public:

	double* weights;
	double* bias;
	double* d_weights;
	double* d_bias;
	
	dense_layer();
	dense_layer(size_t _inputs, size_t _neurons);
	~dense_layer();

	void virtual forward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual forward(double* batched_inputs, size_t input_size, size_t batch_size) override;
	void virtual forward(const layer* prev_layer) override;

	double virtual loss(const std::vector<std::vector<double>>& batched_targets) const override;
	double virtual loss(const std::vector<int>& batched_targets) const override;
	double virtual loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const override;

	void virtual init_back_propigation(const std::vector<std::vector<double>>& batched_targets) override;
	void virtual init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) override;
	void virtual init_back_propigation(double* batched_targets, size_t input_size, size_t batch_size) override;

	void virtual backward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual backward(double* batched_inputs, size_t input_size, size_t _batch_size) override;
	void virtual backward(layer* prev_layer) override;
};

