#pragma once

#include <vector>

//debuging functions will delete later.
void Print_Cuda_Forward_Output(double* input_arr, size_t batch_size, size_t neurons);
void Modify_Cuda_Weight(double* input_arr, int parameter_idx, double esp);

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
	void virtual forward(double* batched_inputs_1, double* batched_inputs_2, size_t _input_size, size_t _batch_size) = 0;
	void virtual forward(const layer* prev_layer) = 0;
	void virtual forward(const layer* prev_layer, const layer* residual_layer) = 0;

	double virtual loss(const std::vector<std::vector<double>>& batched_targets) const = 0;
	double virtual loss(const std::vector<unsigned int>& batched_targets) const = 0;
	double virtual loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const = 0;

	void virtual init_back_propigation(const std::vector<unsigned int>& batched_targets) = 0;
	void virtual init_back_propigation(const std::vector<std::vector<double>>& batched_targets) = 0;
	void virtual init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) = 0;
	void virtual init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) = 0;
	void virtual init_back_propigation(unsigned int* batched_targets, size_t _batch_size) = 0;

	void virtual backward(const std::vector<std::vector<double>>& batched_inputs) = 0;
	void virtual backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) = 0;
	void virtual backward(double* batched_inputs, size_t _input_size, size_t _batch_size) = 0;
	void virtual backward(layer* prev_layer) = 0;
	void virtual backward(layer* prev_layer, layer* residual_layer) = 0;

	void virtual update_paramters_stochastic_gradient_descent(double learning_rate) = 0;
	void virtual update_paramters_stochastic_gradient_descent_with_momentum(double learning_rate, double sgd_mass) = 0;
	void virtual update_paramters_adaptive_gradient(double learning_rate) = 0;
	void virtual update_paramters_root_mean_squared_propagation(double learning_rate, double rho) = 0;
	void virtual update_paramters_adaptive_momentum(double learning_rate, double sgd_mass, double rho) = 0;
};

class dense_layer : public layer {
	
public:

	double* weights;
	double* bias;
	double* d_weights;
	double* d_bias;
	double* weight_momentums;
	double* bias_momentums;
	double* weight_adagrad_cache;
	double* bias_adagrad_cache;
	double* weight_rms_cache;
	double* bias_rms_cache;
	
	dense_layer();
	dense_layer(size_t _inputs, size_t _neurons, activation_functions _layer_activation_function);
	~dense_layer();

	void virtual forward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual forward(double* batched_inputs, size_t _input_size, size_t _batch_size) override;
	void virtual forward(double* batched_inputs_1, double* batched_inputs_2, size_t _input_size, size_t _batch_size) override;
	void virtual forward(const layer* prev_layer) override;
	void virtual forward(const layer* prev_layer, const layer* residual_layer) override;

	double virtual loss(const std::vector<std::vector<double>>& batched_targets) const override;
	double virtual loss(const std::vector<unsigned int>& batched_targets) const override;
	double virtual loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const override;

	void virtual init_back_propigation(const std::vector<unsigned int>& batched_targets) override;
	void virtual init_back_propigation(const std::vector<std::vector<double>>& batched_targets) override;
	void virtual init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) override;
	void virtual init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) override;
	void virtual init_back_propigation(unsigned int* batched_targets, size_t _batch_size) override;

	void virtual backward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual backward(double* batched_inputs, size_t _input_size, size_t _batch_size) override;
	void virtual backward(layer* prev_layer) override;
	void virtual backward(layer* prev_layer, layer* residual_layer) override;

	void virtual update_paramters_stochastic_gradient_descent(double learning_rate) override;
	void virtual update_paramters_stochastic_gradient_descent_with_momentum(double learning_rate, double sgd_mass) override;
	void virtual update_paramters_adaptive_gradient(double learning_rate) override;
	void virtual update_paramters_root_mean_squared_propagation(double learning_rate, double rho) override;
	void virtual update_paramters_adaptive_momentum(double learning_rate, double sgd_mass, double rho) override;
};

class convolutional_layer : public layer {

public:
	double* weights;
	double* bias;
	double* d_weights;
	double* d_bias;
	double* weight_momentums;
	double* bias_momentums;
	double* weight_adagrad_cache;
	double* bias_adagrad_cache;
	double* weight_rms_cache;
	double* bias_rms_cache;
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
	void virtual forward(double* batched_inputs_1, double* batched_inputs_2, size_t _input_size, size_t _batch_size) override;
	void virtual forward(const layer* prev_layer) override;
	void virtual forward(const layer* prev_layer, const layer* residual_layer) override;

	double virtual loss(const std::vector<std::vector<double>>& batched_targets) const override;
	double virtual loss(const std::vector<unsigned int>& batched_targets) const override;
	double virtual loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const override;

	void virtual init_back_propigation(const std::vector<unsigned int>& batched_targets) override;
	void virtual init_back_propigation(const std::vector<std::vector<double>>& batched_targets) override;
	void virtual init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) override;
	void virtual init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) override;
	void virtual init_back_propigation(unsigned int* batched_targets, size_t _batch_size) override;

	void virtual backward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual backward(double* batched_inputs, size_t _input_size, size_t _batch_size) override;
	void virtual backward(layer* prev_layer) override;
	void virtual backward(layer* prev_layer, layer* residual_layer) override;

	void virtual update_paramters_stochastic_gradient_descent(double learning_rate) override;
	void virtual update_paramters_stochastic_gradient_descent_with_momentum(double learning_rate, double sgd_mass) override;
	void virtual update_paramters_adaptive_gradient(double learning_rate) override;
	void virtual update_paramters_root_mean_squared_propagation(double learning_rate, double rho) override;
	void virtual update_paramters_adaptive_momentum(double learning_rate, double sgd_mass, double rho) override;
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
	pooling_layer(size_t _input_size, size_t _channels, size_t _kernal_size, size_t stride, pooling_type layer_type, activation_functions _layer_activation_function);
	~pooling_layer();

	void virtual forward(const std::vector<std::vector<double>>& batched_inputs) override;
	void virtual forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override;
	void virtual forward(double* batched_inputs, size_t _input_size, size_t _batch_size) override;
	void virtual forward(double* batched_inputs_1, double* batched_inputs_2, size_t _input_size, size_t _batch_size) override;
	void virtual forward(const layer* prev_layer) override;
	void virtual forward(const layer* prev_layer, const layer* residual_layer) override;

	double virtual loss(const std::vector<std::vector<double>>& batched_targets) const override;
	double virtual loss(const std::vector<unsigned int>& batched_targets) const override;
	double virtual loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const override;

	void virtual init_back_propigation(const std::vector<unsigned int>& batched_targets) override;
	void virtual init_back_propigation(const std::vector<std::vector<double>>& batched_targets) override;
	void virtual init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) override;
	void virtual init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) override;
	void virtual init_back_propigation(unsigned int* batched_targets, size_t _batch_size) override;

	void virtual backward(const std::vector<std::vector<double>>& batched_inputs) override {}
	void virtual backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) override {}
	void virtual backward(double* batched_inputs, size_t _input_size, size_t _batch_size) override {}
	void virtual backward(layer* prev_layer) override;
	void virtual backward(layer* prev_layer, layer* residual_layer) override;

	void virtual update_paramters_stochastic_gradient_descent(double learning_rate) override {}
	void virtual update_paramters_stochastic_gradient_descent_with_momentum(double learning_rate, double sgd_mass) override {}
	void virtual update_paramters_adaptive_gradient(double learning_rate) override {}
	void virtual update_paramters_root_mean_squared_propagation(double learning_rate, double rho) override {}
	void virtual update_paramters_adaptive_momentum(double learning_rate, double sgd_mass, double rho) override {}
};
