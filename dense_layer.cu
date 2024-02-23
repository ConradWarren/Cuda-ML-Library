#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "layer.hpp"
#include <iostream>
#include <stdio.h>
#include <random>

__global__ static void Cuda_Dense_Layer_Forward_Pass(double* batched_inputs, double* weights, double* bias, double* forward_output, size_t inputs, size_t neurons, size_t batch_size) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		
		forward_output[batch_idx * neurons + neuron_idx] = bias[neuron_idx];
		for (size_t i = 0; i < inputs; i++) {
			forward_output[batch_idx*neurons + neuron_idx] += weights[neuron_idx*inputs + i] * batched_inputs[batch_idx * inputs + i];
		}
	}
}

__global__ static void Cuda_Dense_Layer_Init_Back_Propigation(double* batched_targets, double* forward_ouput, double* backward_input, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		backward_input[batch_idx * neurons + neuron_idx] = 2.0 * (forward_ouput[batch_idx * neurons + neuron_idx] - batched_targets[batch_idx * neurons + neuron_idx]) / (double)(batch_size * neurons);
	}
}

__global__ static void Cuda_Dense_Layer_Init_Cross_Catigorial_Loss_Back_Propigation(unsigned int* batched_targets, double* forward_output, double* backward_input, size_t batch_size, size_t neurons) {
	
	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		backward_input[batch_idx * neurons + neuron_idx] = forward_output[batch_idx * neurons + neuron_idx]/(double)batch_size;
		if (neuron_idx == batched_targets[batch_idx]) backward_input[batch_idx * neurons + neuron_idx] -= 1.0 / (double)batch_size;
	}
}

__global__ static void Cuda_Dense_Layer_First_Backward_Pass(double* batched_inputs, double* backward_input, double* d_weights, size_t batch_size, size_t neurons, size_t inputs){
	
	size_t neuron_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t input_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (neuron_idx < neurons && input_idx < inputs) {

		d_weights[neuron_idx * inputs + input_idx] = 0.0;
		for (size_t i = 0; i < batch_size; i++) {
			d_weights[neuron_idx * inputs + input_idx] += batched_inputs[i * inputs + input_idx] * backward_input[i * neurons + neuron_idx];
		}
	}
}

__global__ static void Cuda_Dense_Layer_Second_Backward_Pass(double* backward_input, double* d_bias, size_t batch_size, size_t neurons) {

	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (neuron_idx < neurons) {
		d_bias[neuron_idx] = 0.0;
		for (size_t i = 0; i < batch_size; i++) {
			d_bias[neuron_idx] += backward_input[(i * neurons) + neuron_idx];
		}
	}
}

__global__ static void Cuda_Sigmoid_Activation_Forward_Pass(double* forward_output, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		forward_output[batch_idx * neurons + neuron_idx] = 1.0 / (1.0 + std::powf(2.71828182846, -forward_output[batch_idx * neurons + neuron_idx]));
	}
}

__global__ static void Cuda_Sigmoid_Activation_Backward_Pass(double* backward_input, double* forward_output, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		backward_input[batch_idx * neurons + neuron_idx] *= forward_output[batch_idx * neurons + neuron_idx] * (1.0 - forward_output[batch_idx * neurons + neuron_idx]);
	}
}

__global__ static void Cuda_Rectified_Linear_Activation_Forward_Pass(double* forward_output, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons && forward_output[batch_idx * neurons + neuron_idx] < 0.0) {
		forward_output[batch_idx * neurons + neuron_idx] = 0.0;
	}
}

__global__ static void Cuda_Rectified_Linear_Activation_Backward_Pass(double* backward_input, double* forward_input, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons && forward_input[batch_idx * neurons + neuron_idx] == 0.0) {
		backward_input[batch_idx * neurons + neuron_idx] = 0.0;
	}
}

__global__ static void Cuda_Softmax_Activation_Forward_Pass(double* forward_output, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		forward_output[batch_idx * neurons + neuron_idx] = std::powf(2.71828182846, forward_output[batch_idx * neurons + neuron_idx]);
	}

	__syncthreads();

	double sum = 0;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		for (int i = 0; i < neurons; i++) {
			sum += forward_output[batch_idx * neurons + i];
		}
	}
	
	__syncthreads();

	if (batch_idx < batch_size && neuron_idx < neurons) {
		forward_output[batch_idx * neurons + neuron_idx] /= sum;
	}
}

__global__ static void Cuda_Softmax_Activation_Bakcward_Pass(double* forward_output, double* backward_input, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	double sum = 0.0;
	if (batch_idx < batch_size && neuron_idx < neurons) {

		for (int i = 0; i < neurons; i++) {
			sum += backward_input[batch_idx * neurons + i] * forward_output[batch_idx * neurons + i];
		}
	}

	__syncthreads();

	if (batch_idx < batch_size && neuron_idx < neurons) {
		backward_input[batch_idx * neurons + neuron_idx] = (backward_input[batch_idx * neurons + neuron_idx] - sum) * forward_output[batch_idx * neurons + neuron_idx];
	}
}

__global__ static void Cuda_Partial_Derivitive_of_Loss(double* backward_input, double* weights, double* prev_layer_backward_input, size_t batch_size, size_t inputs, size_t neurons) {
	
	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t input_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && input_idx < inputs) {
		for (int i = 0; i < neurons; i++) {
			prev_layer_backward_input[batch_idx * inputs + input_idx] += backward_input[batch_idx * neurons + i] * weights[i * inputs + input_idx];
		}
	}
}

__global__ static void Cuda_Matix_Addition(double* residual_batched_inputs, double* forward_output, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (batch_idx < batch_size && neuron_idx < neurons) {
		forward_output[batch_idx * neurons + neuron_idx] += residual_batched_inputs[batch_idx * neurons + neuron_idx];
	}
}

__global__ static void Cuda_Graident_Decent(double* d_weights, double* d_bias, double* weights, double* bias,double learning_rate, size_t neurons, size_t inputs) {

	size_t neuron_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t input_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (neuron_idx < neurons && input_idx < inputs) {
		weights[neuron_idx * inputs + input_idx] -= d_weights[neuron_idx * inputs + input_idx] * learning_rate;
		if (input_idx == 0) {
			bias[neuron_idx] -= d_bias[neuron_idx] * learning_rate;
		}
	}
	
}

dense_layer::dense_layer() {
	neurons = 0;
	inputs = 0;
	weights = nullptr;
	bias = nullptr;
	forward_output = nullptr;
	backward_input = nullptr;
	d_weights = nullptr;
	d_bias = nullptr;
	layer_activation_function = activation_functions::Linear;
}

dense_layer::dense_layer(size_t _inputs, size_t _neurons, activation_functions _layer_activation_function) {

	neurons = _neurons;
	inputs = _inputs;
	batch_size = 0;
	layer_activation_function = _layer_activation_function;
	cudaError error_code;

	error_code = cudaMalloc((void**)&d_weights, neurons * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemset(d_weights, 0, neurons * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemset failed in dnese_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&d_bias, neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemset(d_bias, 0, neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemset failed in dense_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&weights, neurons * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&bias, neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
		exit(error_code);
	}

	forward_output = nullptr;
	backward_input = nullptr;

	double* temp_weights = (double*)malloc(neurons * inputs * sizeof(double));
	double* temp_bias = (double*)malloc(neurons * sizeof(double));

	if (temp_weights == nullptr || temp_bias == nullptr) {
		std::cerr << "Error: Failed to allocate memory in dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	std::mt19937 generator;

	for (size_t i = 0; i < neurons; i++) {
		temp_bias[i] = distribution(generator);
		for (size_t j = 0; j < inputs; j++) {
			temp_weights[i * inputs + j] = distribution(generator);
		}
	}

	error_code = cudaMemcpy(weights, temp_weights, neurons * inputs * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed in dense_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(bias, temp_bias, neurons * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed in dense_layer" << std::endl;
		exit(error_code);
	}

	free(temp_weights);
	free(temp_bias);

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed in dense_layer" << std::endl;
		exit(error_code);
	}
}

dense_layer::~dense_layer() {
	cudaFree(weights);
	cudaFree(d_weights);
	cudaFree(bias);
	cudaFree(d_bias);
	cudaFree(forward_output);
	cudaFree(backward_input);
}

void dense_layer::forward(const std::vector<std::vector<double>>& batched_inputs) {

	double* input_arr = nullptr;
	cudaError error_code;

	error_code = cudaMalloc((void**)&input_arr, batched_inputs.size() * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
		exit(error_code);
	}
	
	for (size_t i = 0; i < batched_inputs.size(); i++) {

		if (batched_inputs[i].size() != inputs) {
			std::cerr << "Error: batched_inputs of invalid shape" << std::endl;
			exit(EXIT_FAILURE);
		}
		error_code = cudaMemcpy(input_arr + i * inputs, batched_inputs[i].data(), inputs * sizeof(double), cudaMemcpyHostToDevice);
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemcpy to device failed in dense_layer" << std::endl;
			exit(error_code);
		}
	}

	forward(input_arr, inputs, batched_inputs.size());
	cudaFree(input_arr);
}
void dense_layer::forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {
	
	double* input_arr = nullptr;
	cudaError error_code;

	error_code = cudaMalloc((void**)&input_arr, batched_inputs.size() * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
		exit(error_code);
	}

	unsigned int current_size = 0;

	for (int i = 0; i < batched_inputs.size(); i++) {
		for (int j = 0; j < batched_inputs[j].size(); j++) {
			for (int y = 0; y < batched_inputs[i][j].size(); y++) {

				if (current_size + batched_inputs[i][j][y].size() > inputs) {
					std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
					exit(EXIT_FAILURE);
				}

				error_code = cudaMemcpy(input_arr + i * inputs + current_size, batched_inputs[i][j][y].data(), batched_inputs[i][j][y].size() * sizeof(double), cudaMemcpyHostToDevice);
				if (error_code != cudaError::cudaSuccess) {
					std::cerr << "Error: cudaMemcpy to device failed in dense_layer" << std::endl;
					exit(error_code);
				}
				
				current_size += batched_inputs[i][j][y].size();
			}
		}

		if (current_size != inputs) {
			std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
			exit(EXIT_FAILURE);
		}
		current_size = 0;
	}
	
	forward(input_arr, inputs, batch_size);
	cudaFree(input_arr);
}

void dense_layer::forward(double* batched_inputs, size_t _input_size, size_t _batch_size) {

	if (_input_size != inputs) {
		std::cerr << "Error: Incompatible input for dense layer of shape " << inputs << " " << neurons << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (_batch_size != batch_size || forward_output == nullptr) {

		cudaFree(forward_output);
		cudaFree(backward_input);
		backward_input = nullptr;

		error_code = cudaMalloc((void**)&forward_output, neurons * _batch_size * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
			exit(error_code);
		}

		batch_size = _batch_size;
	}
	
	dim3 blocks(neurons / 16 + 1, batch_size / 16 + 1);
	dim3 threads(16, 16);

	if (backward_input != nullptr) {
		error_code = cudaMemset(backward_input, 0, batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemset failed in dense_layer" << std::endl;
			exit(error_code);
		}
	}

	Cuda_Dense_Layer_Forward_Pass<<<blocks, threads>>>(batched_inputs, weights, bias, forward_output, inputs, neurons, batch_size);
	
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch forward pass kernal in dense_layer" << std::endl;
		exit(error_code);
	}

	if (layer_activation_function == activation_functions::Sigmoid) {
		Cuda_Sigmoid_Activation_Forward_Pass<<<blocks, threads>>>(forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Rectified_Linear) {
		Cuda_Rectified_Linear_Activation_Forward_Pass<<<blocks, threads>>>(forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Softmax) {
		Cuda_Softmax_Activation_Forward_Pass<<<blocks, threads>>>(forward_output, batch_size, neurons);
	}

	if (layer_activation_function != activation_functions::Linear && (error_code = cudaGetLastError()) != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch forward activation kernal in dense_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
}

void dense_layer::forward(double* batched_inputs, double* residual_batched_inputs, size_t _input_size, size_t _batch_size) {

	if (_input_size != inputs) {
		std::cerr << "Error: Batched_inputs of invalid shape to connect to dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (batch_size != _batch_size || forward_output == nullptr) {
		cudaFree(forward_output);
		cudaFree(backward_input);
		backward_input = nullptr;
		
		error_code = cudaMalloc((void**)&forward_output, _batch_size * neurons * sizeof(double));
		
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
			exit(error_code);
		}

		batch_size = _batch_size;
	}

	dim3 blocks(neurons / 16 + 1, batch_size / 16 + 1);
	dim3 threads(16, 16);

	if (backward_input != nullptr) {
		error_code = cudaMemset(backward_input, 0, batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemset failed in dense_layer" << std::endl;
			exit(error_code);
		}
	}

	Cuda_Dense_Layer_Forward_Pass<<<blocks, threads>>>(batched_inputs, weights, bias, forward_output, inputs, neurons, batch_size);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch forward pass kernal" << std::endl;
		exit(error_code);
	}

	Cuda_Matix_Addition<<<blocks, threads>>>(residual_batched_inputs, forward_output, batch_size, neurons);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch matrix addition kernal" << std::endl;
		exit(error_code);
	}
	
	if (layer_activation_function == activation_functions::Sigmoid) {
		Cuda_Sigmoid_Activation_Forward_Pass<<<blocks, threads>>>(forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Rectified_Linear) {
		Cuda_Rectified_Linear_Activation_Forward_Pass<<<blocks, threads>>>(forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Softmax) {
		Cuda_Softmax_Activation_Forward_Pass<<<blocks, threads>>>(forward_output, batch_size, neurons);
	}

	if (layer_activation_function != activation_functions::Linear && (error_code = cudaGetLastError()) != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch activation function forward pass kernal" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
}

void dense_layer::forward(const layer* prev_layer) {

	if (prev_layer->neurons != inputs) {
		std::cerr << "Error: Prev_layer of invalid input shape to connect to dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	forward(prev_layer->forward_output, prev_layer->neurons, prev_layer->batch_size);
}

void dense_layer::forward(const layer* prev_layer, const layer* residual_layer) {

	if (prev_layer->neurons != inputs || residual_layer->neurons != neurons || prev_layer->batch_size != residual_layer->batch_size) {
		std::cerr << "Error: Prev_layer_1 and prev_layer_2 of invalid input shape to connect to dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	forward(prev_layer->forward_output, residual_layer->forward_output, prev_layer->neurons, prev_layer->batch_size);
}

double dense_layer::loss(const std::vector<std::vector<double>>& batched_targets) const {

	if (forward_output == nullptr) {
		std::cerr << "Error: No forwad output in dense_layer to calculate loss" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Incompatible batch size, cannot calculate loss" << std::endl;
		exit(EXIT_FAILURE);
	}

	double result = 0.0;
	double* host_forward_output = (double*)malloc(batch_size * neurons * sizeof(double));

	if (host_forward_output == nullptr) {
		std::cerr << "Error: Failed to allocate memory in dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code = cudaMemcpy(host_forward_output, forward_output, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to host failed in dense_layer" << std::endl;
		exit(error_code);
	}

	for (int i = 0; i < batch_size; i++) {

		if (batched_targets[i].size() != neurons) {
			std::cerr << "Error: Invalid input shape" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (int j = 0; j < neurons; j++) {
			result += ((host_forward_output[i * neurons + j] - batched_targets[i][j]) * (host_forward_output[i * neurons + j] - batched_targets[i][j])) / (double)(batch_size * neurons);
		}
	}
	free(host_forward_output);
	return result;
}

double dense_layer::loss(const std::vector<int>& batched_targets) const {

	//ALL of this will be moved to another function when I start renaming things. 

	if (layer_activation_function != activation_functions::Softmax) {
		std::cerr << "Error: Not a classification model" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (forward_output == nullptr) {
		std::cerr << "Error: No forwad output in dense layer to calculate loss" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Incompatible batch size, cannot calculate loss" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	double* host_forward_output = (double*)malloc(batch_size * neurons * sizeof(double));
	if (host_forward_output == nullptr) {
		std::cerr << "Error: Failed to allocate memory in dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code = cudaMemcpy(host_forward_output, forward_output, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpyDeviceToHost failed in dense_layer" << std::endl;
		exit(error_code);
	}

	double result = 0;

	for (int i = 0; i < batch_size; i++) {

		if (batched_targets[i] >= neurons) {
			std::cerr << "Error: invalid batched_tagets input" << std::endl;
			exit(EXIT_FAILURE);
		}
		host_forward_output[batched_targets[i]] = (host_forward_output[batched_targets[i]] > 1e-7) ? host_forward_output[batched_targets[i]] : 1e-7;
		host_forward_output[batched_targets[i]] = (host_forward_output[batched_targets[i]] < 1 - 1e-7) ? host_forward_output[batched_targets[i]] : 1 - 1e-7;
		result += -std::log(host_forward_output[batched_targets[i]])/(double)(batch_size);   
	}
	free(host_forward_output);
	return result;
}
double dense_layer::loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const {
	
	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	double result = 0.0;
	int idx = 0;

	double* host_forward_output = (double*)malloc(batch_size * neurons * sizeof(double));
	if (host_forward_output == nullptr) {
		std::cerr << "Error: Failed to allocate memory in dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code = cudaMemcpy(host_forward_output, forward_output, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to host failed in dense_layer" << std::endl;
		exit(error_code);
	}

	for (int i = 0; i < batch_size; i++) {

		for (int j = 0; j < batched_targets[i].size(); j++) {
			for (int y = 0; y < batched_targets[i][j].size(); y++) {

				if (idx + batched_targets[i][j][y].size() > neurons) {
					std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
					exit(EXIT_FAILURE);
				}

				for (int x = 0; x < batched_targets[i][j][y].size(); x++) {
					result += (host_forward_output[batch_size * neurons + idx] - batched_targets[i][j][y][x]) * (host_forward_output[batch_size * neurons + idx] - batched_targets[i][j][y][x]) / (double)(batch_size * neurons);
					idx++;
				}
			}
		}

		if (idx != neurons) {
			std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
			exit(EXIT_FAILURE);
		}

		idx = 0;
	}
	free(host_forward_output);
	return result;
}

void dense_layer::init_back_propigation(const std::vector<unsigned int>& batched_targets) {

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	unsigned int* input_arr = nullptr;

	for (int i = 0; i < batch_size; i++) {
		if (batched_targets[i] >= neurons) {
			std::cerr << "Error: Invalid batched_targets inputs" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(input_arr, batched_targets.data(), batch_size * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpyHostToDevice failed in dense_layer" << std::endl;
		exit(error_code);
	}

	init_back_propigation(input_arr, batch_size);
	cudaFree(input_arr);
}

void dense_layer::init_back_propigation(const std::vector<std::vector<double>>& batched_targets) {
	
	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
		exit(error_code);
	}

	for (size_t i = 0; i < batch_size; i++) {
		
		if (batched_targets[i].size() != neurons) {
			std::cerr << "Error: batched targets of invalid shape" << std::endl;
			exit(EXIT_FAILURE);
		}
		
		error_code = cudaMemcpy(input_arr + i * neurons, batched_targets[i].data(), neurons * sizeof(double), cudaMemcpyHostToDevice);
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemcpy to device failed in dense_layer" << std::endl;
			exit(error_code);
		}
	}

	init_back_propigation(input_arr, neurons, batch_size);

	cudaFree(input_arr);
}

void dense_layer::init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {
	
	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
		exit(error_code);
	}

	unsigned int current_size = 0;
	
	for (int i = 0; i < batch_size; i++) {

		for (int j = 0; j < batched_targets[i].size(); j++) {

			for (int y = 0; y < batched_targets[i][j].size(); y++) {

				if (current_size + batched_targets[i][j][y].size() > neurons) {
					std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
					exit(EXIT_FAILURE);
				}
				
				memcpy(input_arr + i * neurons +  current_size, batched_targets[i][j][y].data(), batched_targets[i][j][y].size() * sizeof(double));
				error_code = cudaMemcpy(input_arr + i * neurons + current_size, batched_targets[i][j][y].data(), batched_targets[i][j][y].size() * sizeof(double), cudaMemcpyHostToDevice);
				if (error_code != cudaError::cudaSuccess) {
					std::cerr << "Error: cudaMemcpy to device failed in dense_layer" << std::endl;
					exit(error_code);
				}
				current_size += batched_targets[i][j][y].size();
			}
		}

		if (current_size != neurons) {
			std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
			exit(EXIT_FAILURE);
		}

		current_size = 0;
	}

	init_back_propigation(input_arr, neurons, batch_size);

	cudaFree(input_arr);
}

void dense_layer::init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) {
	
	if (batch_size != _batch_size || _input_size != neurons) {
		std::cerr << "Error: Invalid input size for dense layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (backward_input == nullptr) {
		error_code = cudaMalloc((void**)&backward_input, batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
			exit(error_code);
		}
	}

	dim3 blocks(neurons/16 + 1, batch_size/16 + 1);
	dim3 threads(16, 16);

	Cuda_Dense_Layer_Init_Back_Propigation<<<blocks, threads>>>(batched_targets, forward_output, backward_input, batch_size, neurons);
	
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch init back propigation kernal in dense_layer" << std::endl;
		exit(error_code);
	}

	if (layer_activation_function == activation_functions::Sigmoid) {
		Cuda_Sigmoid_Activation_Backward_Pass<<<blocks, threads>>>(backward_input, forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Rectified_Linear) {
		Cuda_Rectified_Linear_Activation_Backward_Pass<<<blocks, threads>>>(backward_input, forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Softmax) {
		Cuda_Softmax_Activation_Bakcward_Pass<<<blocks, threads>>>(forward_output, backward_input, batch_size, neurons);
	}

	if (layer_activation_function != activation_functions::Linear && (error_code = cudaGetLastError()) != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch backward activation function kernal" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDevice Synchronize failed in dense_layer" << std::endl;
		exit(error_code);
	}
}

void dense_layer::init_back_propigation(unsigned int* batched_targets, size_t _batch_size) {

	if (batch_size != _batch_size) {
		std::cerr << "Error: Invalid input size for dense layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (layer_activation_function != activation_functions::Softmax) {
		std::cerr << "Error: Invalid activation_function for Init_Cross_Catagorial_Loss in dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (backward_input == nullptr) {
		error_code = cudaMalloc((void**)&backward_input, batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
			exit(error_code);
		}
	}

	dim3 blocks(neurons / 16 + 1, batch_size / 16 + 1);
	dim3 threads(16, 16);

	Cuda_Dense_Layer_Init_Cross_Catigorial_Loss_Back_Propigation<<<blocks, threads>>>(batched_targets, forward_output, backward_input, batch_size, neurons);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to Launch Init_Cross_Catigorial_Loss_Back_Propigation in dense_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed in dense_layer" << std::endl;
		exit(error_code);
	}
}

void dense_layer::backward(const std::vector<std::vector<double>>& batched_inputs) {
	
	if (batch_size != batched_inputs.size()) {
		std::cerr << "Error: Incompatible batch size" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * inputs * sizeof(double));

	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
		exit(error_code);
	}

	for (int i = 0; i < batch_size; i++) {
		if (batched_inputs[i].size() != inputs) {
			std::cerr << "Error: Batched inputs of invalid shape" << std::endl;
			exit(EXIT_FAILURE);
		}

		error_code = cudaMemcpy(input_arr + i * inputs, batched_inputs[i].data(), inputs * sizeof(double), cudaMemcpyHostToDevice);
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemcpy to device failed in dense_layer" << std::endl;
			exit(error_code);
		}
	}

	backward(input_arr, inputs, batch_size);
	cudaFree(input_arr);
}

void dense_layer::backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {
	
	if (batched_inputs.size() != batch_size) {
		std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = (double*)malloc(batch_size * inputs * sizeof(double));
	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
		exit(error_code);
	}

	unsigned int current_size = 0;

	for (int i = 0; i < batch_size; i++) {

		for (int j = 0; j < batched_inputs[i].size(); j++) {
			for (int y = 0; y < batched_inputs[i][j].size(); y++) {

				if (current_size + batched_inputs[i][j][y].size() > inputs) {
					std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
					exit(EXIT_FAILURE);
				}

				error_code = cudaMemcpy(input_arr + i * inputs + current_size, batched_inputs[i][j][y].data(), batched_inputs[i][j][y].size() * sizeof(double), cudaMemcpyHostToDevice);
				if (error_code != cudaError::cudaSuccess) {
					std::cerr << "Error: cudaMemcpy to device failed in dense_layer" << std::endl;
					exit(error_code);
				}
				
				current_size += batched_inputs[i][j][y].size();
			}
		}

		if (current_size != inputs) {
			std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
			exit(EXIT_FAILURE);
		}
		current_size = 0;
	}

	backward(input_arr, inputs, batch_size);
	cudaFree(input_arr);
}

void dense_layer::backward(double* batched_inputs, size_t _input_size, size_t _batch_size) {

	if (_batch_size != batch_size || _input_size != inputs) {
		std::cerr << "Error: Invalid input size for backward pass" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (backward_input == nullptr) {
		std::cerr << "Error: Dense_layer not intialized for backward pass" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	cudaError error_code;
	dim3 blocks(inputs/16 + 1, neurons/16 + 1);
	dim3 threads(16, 16);
	
	Cuda_Dense_Layer_First_Backward_Pass<<<blocks, threads>>>(batched_inputs, backward_input, d_weights, batch_size, neurons, inputs);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: First backward pass kernal failed to launch" << std::endl;
		exit(error_code);
	}
	
	Cuda_Dense_Layer_Second_Backward_Pass<<<neurons / 16 + 1, 16>>>(backward_input, d_bias, batch_size, neurons);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Second backward pass kernal failed to launch" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}

}

void dense_layer::backward(layer* prev_layer) {
	
	if (prev_layer->batch_size != batch_size || prev_layer->neurons != inputs) {
		std::cerr << "Error: Prev_layer of invalid input shape or batch size to connect to dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	backward(prev_layer->forward_output, prev_layer->neurons, batch_size);
	
	cudaError error_code;
	dim3 blocks(inputs / 16 + 1, batch_size / 16 + 1);
	dim3 threads(16, 16);

	if (prev_layer->backward_input == nullptr) {
		
		error_code = cudaMalloc((void**)&prev_layer->backward_input, batch_size * inputs * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
			exit(error_code);
		}

		error_code = cudaMemset(prev_layer->backward_input, 0, batch_size * inputs * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemset failed in dense_layer" << std::endl;
			exit(error_code);
		}

		error_code = cudaDeviceSynchronize();
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaDeviceSynchronize failed in dense_layer" << std::endl;
			exit(error_code);
		}
	}

	Cuda_Partial_Derivitive_of_Loss<<<blocks, threads>>>(backward_input, weights, prev_layer->backward_input, batch_size, inputs, neurons);
	
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to Launch Partial Derivitive of Loss Kernal" << std::endl;
		exit(error_code);
	}

	if (prev_layer->layer_activation_function == activation_functions::Sigmoid) {
		Cuda_Sigmoid_Activation_Backward_Pass<<<blocks, threads>>>(prev_layer->backward_input, prev_layer->forward_output, batch_size, inputs);
	}
	else if (prev_layer->layer_activation_function == activation_functions::Rectified_Linear) {
		Cuda_Rectified_Linear_Activation_Backward_Pass<<<blocks, threads>>>(prev_layer->backward_input, prev_layer->forward_output, batch_size, inputs);
	}
	else if (prev_layer->layer_activation_function == activation_functions::Softmax) {
		Cuda_Softmax_Activation_Bakcward_Pass<<<blocks, threads>>>(prev_layer->forward_output, prev_layer->backward_input, batch_size, inputs);
	}

	if (prev_layer->layer_activation_function != activation_functions::Linear && (error_code = cudaGetLastError()) != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch activation function backward pass kernals" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
}

void dense_layer::backward(layer* prev_layer, layer* residual_layer) {

	if (prev_layer->neurons != inputs || residual_layer->neurons != neurons || prev_layer->batch_size != batch_size || residual_layer->batch_size != batch_size) {
		std::cerr << "Error: Prev_layer_1 or residual_layer of invalid input shape or batch size to connect to dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (residual_layer->backward_input == nullptr) {
		error_code = cudaMalloc((void**)&residual_layer->backward_input, batch_size * inputs * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in dense_layer" << std::endl;
			exit(error_code);
		}
	}

	error_code = cudaMemcpy(residual_layer->backward_input, backward_input, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy device to device failed in dense_layer" << std::endl;
		exit(error_code);
	}
	
	error_code = cudaDeviceSynchronize();
	if (error_code != cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed in dense_layer" << std::endl;
		exit(error_code);
	}

	backward(prev_layer);
}

void dense_layer::update_paramters(double learning_rate) {
	
	dim3 blocks(inputs/16 + 1, neurons/16 + 1);
	dim3 threads(16, 16);

	Cuda_Graident_Decent<<<blocks, threads>>>(d_weights, d_bias, weights, bias, learning_rate, neurons, inputs);

	cudaError error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch Graident Decent kernal in dense_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
}

//debugging function will delete later. 
void Print_Cuda_Forward_Output(double* input_arr, size_t batch_size, size_t neurons) {

	double* host_input_arr = (double*)malloc(batch_size * neurons * sizeof(double));
	cudaMemcpy(host_input_arr, input_arr, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	
	for (int i = 0; i < batch_size; i++) {

		std::cout << "{";
		for (int j = 0; j < neurons; j++) {
			std::cout << i * neurons + j << " = > " << host_input_arr[i * neurons + j];
			if (j + 1 < neurons) std::cout << ", ";
		}
		std::cout << "}\n";
	}
	std::cout << "\n";
}