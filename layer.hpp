#pragma once

#include <vector>

enum class activation_functions{Linear ,Rectified_Linear, Sigmoid, Softmax};
enum class pooling_type{Max, Average};

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
	void virtual forward(double* batched_inputs, size_t _input_size, size_t _batch_size) = 0;
	void virtual forward(const layer* prev_layer) = 0;

	double virtual loss(const std::vector<std::vector<double>>& batched_targets) const = 0;
	double virtual loss(const std::vector<int>& batched_targets) const = 0;
	double virtual loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const = 0;

	void virtual init_back_propigation(const std::vector<std::vector<double>>& batched_targets) = 0;
	void virtual init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) = 0;
	void virtual init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) = 0;

	void virtual backward(const std::vector<std::vector<double>>& batched_inputs) = 0;
	void virtual backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) = 0;
	void virtual backward(double* batched_inputs, size_t _input_size, size_t _batch_size) = 0;
	void virtual backward(layer* prev_layer) = 0;

	void virtual update_paramters(double learning_rate) = 0;
};

class dense_layer : public layer {
	
public:

	double* weights;
	double* bias;
	double* d_weights;
	double* d_bias;
	
	dense_layer();
	dense_layer(size_t _inputs, size_t _neurons, activation_functions _layer_activation_function);
	~dense_layer();

	void virtual forward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual forward(double* batched_inputs, size_t _input_size, size_t _batch_size) override;
	void virtual forward(const layer* prev_layer) override;

	double virtual loss(const std::vector<std::vector<double>>& batched_targets) const override;
	double virtual loss(const std::vector<int>& batched_targets) const override;
	double virtual loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const override;

	void virtual init_back_propigation(const std::vector<std::vector<double>>& batched_targets) override;
	void virtual init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) override;
	void virtual init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) override;

	void virtual backward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual backward(double* batched_inputs, size_t _input_size, size_t _batch_size) override;
	void virtual backward(layer* prev_layer) override;

	void virtual update_paramters(double learning_rate) override;
};

class convolutional_layer : public layer {

public:
	double* weights;
	double* bias;
	double* d_weights;
	double* d_bias;
	size_t kernals;
	size_t kernal_size;
	size_t channels;
	size_t padding;
	size_t stride;
	size_t input_size;
	size_t output_size;

	convolutional_layer();
	convolutional_layer(size_t _input_size, size_t _channels, size_t _kernals, size_t _kernal_size, size_t _stride, size_t _padding, activation_functions _layer_activation_function);
	~convolutional_layer();

	void virtual forward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual forward(double* batched_inputs, size_t _input_size, size_t _batch_size) override;
	void virtual forward(const layer* prev_layer) override;

	double virtual loss(const std::vector<std::vector<double>>& batched_targets) const override;
	double virtual loss(const std::vector<int>& batched_targets) const override;
	double virtual loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const override;

	void virtual init_back_propigation(const std::vector<std::vector<double>>& batched_targets) override;
	void virtual init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) override;
	void virtual init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) override;

	void virtual backward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual backward(double* batched_inputs, size_t _input_size, size_t _batch_size) override;
	void virtual backward(layer* prev_layer) override;

	void virtual update_paramters(double learning_rate) override;
};

class pooling_layer : public layer {


public:
	size_t input_size;
	size_t channels;
	size_t kernal_size;
	size_t stride;
	size_t output_size;
	pooling_type pooling_layer_type;

	pooling_layer();
	pooling_layer(size_t _input_size, size_t _channels, size_t _kernal_size, size_t stride, pooling_type layer_type);
	~pooling_layer();

	void virtual forward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual forward(double* batched_inputs, size_t _input_size, size_t _batch_size) override;
	void virtual forward(const layer* prev_layer) override;

	double virtual loss(const std::vector<std::vector<double>>& batched_targets) const override;
	double virtual loss(const std::vector<int>& batched_targets) const override;
	double virtual loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const override;

	void virtual init_back_propigation(const std::vector<std::vector<double>>& batched_targets) override;
	void virtual init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) override;
	void virtual init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) override;

	void virtual backward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual backward(double* batched_inputs, size_t _input_size, size_t _batch_size) override;
	void virtual backward(layer* prev_layer) override;

	void virtual update_paramters(double learning_rate) override;
};
