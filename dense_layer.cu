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

__global__ static void Cuda_Partial_Derivitive_of_Loss(double* backward_input, double* weights, double* prev_layer_backward_input, size_t batch_size, size_t inputs, size_t neurons) {
	
	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t input_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && input_idx < inputs) {

		prev_layer_backward_input[batch_idx * inputs + input_idx] = 0.0;
		for (int i = 0; i < neurons; i++) {
			prev_layer_backward_input[batch_idx * inputs + input_idx] += backward_input[batch_idx * neurons + i] * weights[i * inputs + input_idx];
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

	weights = (double*)malloc(inputs * neurons * sizeof(double));
	bias = (double*)malloc(neurons * sizeof(double));
	d_weights = (double*)malloc(inputs * neurons * sizeof(double));
	d_bias = (double*)malloc(neurons * sizeof(double));

	forward_output = nullptr;
	backward_input = nullptr;

	if (bias == nullptr || weights == nullptr || d_weights == nullptr || d_bias == nullptr) {
		std::cerr << "Error: Could not allocate memory in dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	std::mt19937 generator;

	for (size_t i = 0; i < neurons; i++) {
		bias[i] = distribution(generator);
		for (size_t j = 0; j < inputs; j++) {
			weights[i * inputs + j] = distribution(generator);
		}
	}

}

dense_layer::~dense_layer() {
	free(weights);
	free(d_weights);
	free(bias);
	free(d_bias);
	free(forward_output);
	free(backward_input);
}

void dense_layer::forward(const std::vector<std::vector<double>>& batched_inputs) {

	double* input_arr = (double*)malloc(batched_inputs.size() * inputs * sizeof(double));

	if (input_arr == nullptr) {
		std::cerr << "Error: Could not allocate memory in dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (size_t i = 0; i < batched_inputs.size(); i++) {

		if (batched_inputs[i].size() != inputs) {
			std::cerr << "Error: batched_inputs of invalid shape" << std::endl;
			exit(EXIT_FAILURE);
		}
		memcpy(input_arr + (inputs * i), batched_inputs[i].data(), inputs * sizeof(double));
	}

	forward(input_arr, inputs, batched_inputs.size());

	free(input_arr);
}
void dense_layer::forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {
	
	double* input_arr = (double*)malloc(batched_inputs.size() * inputs * sizeof(double));
	unsigned int current_size = 0;

	if (input_arr == nullptr) {
		std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < batched_inputs.size(); i++) {
		for (int j = 0; j < batched_inputs[j].size(); j++) {
			for (int y = 0; y < batched_inputs[i][j].size(); y++) {

				if (current_size + batched_inputs[i][j][y].size() > inputs) {
					std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
					exit(EXIT_FAILURE);
				}

				memcpy(input_arr + i * inputs + current_size, batched_inputs[i][j][y].data(), batched_inputs[i][j][y].size() * sizeof(double));

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
	free(input_arr);
}

void dense_layer::forward(double* batched_inputs, size_t _input_size, size_t _batch_size) {

	if (_input_size != inputs) {
		std::cerr << "Error: Incompatible input for dense layer of shape " << inputs << " " << neurons << std::endl;
		exit(EXIT_FAILURE);
	}

	if (_batch_size != batch_size || forward_output == nullptr) {

		if (forward_output != nullptr) free(forward_output);
		if (backward_input != nullptr) {
			free(backward_input);
			backward_input = nullptr;
		}
		forward_output = (double*)malloc(neurons * _batch_size * sizeof(double));

		if (forward_output == nullptr) {
			std::cerr << "Error: Could not allocate memory for dense_layer output" << std::endl;
			exit(EXIT_FAILURE);
		}
		batch_size = _batch_size;
	}

	double* cuda_batched_inputs = nullptr;
	double* cuda_weights = nullptr;
	double* cuda_bias = nullptr;
	double* cuda_forward_output = nullptr;
	cudaError error_code;

	error_code = cudaMalloc((void**)&cuda_batched_inputs, batch_size * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&cuda_weights, neurons * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&cuda_bias, neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&cuda_forward_output, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(cuda_batched_inputs, batched_inputs, batch_size * inputs * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(cuda_weights, weights, neurons * inputs * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(cuda_bias, bias, neurons * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed" << std::endl;
		exit(error_code);
	}
	
	
	dim3 blocks(neurons/16 + 1, batch_size/16 + 1);
	dim3 threads(16, 16);

	Cuda_Dense_Layer_Forward_Pass<<<blocks, threads>>>(cuda_batched_inputs, cuda_weights, cuda_bias, cuda_forward_output, inputs, neurons, batch_size);
	
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Forward Pass Kernal Launch failed" << std::endl;
		exit(error_code);
	}

	if (layer_activation_function == activation_functions::Sigmoid) {
		Cuda_Sigmoid_Activation_Forward_Pass<<<blocks, threads>>>(cuda_forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Rectified_Linear) {
		Cuda_Rectified_Linear_Activation_Forward_Pass<<<blocks, threads>>>(cuda_forward_output, batch_size, neurons);
	}

	if (layer_activation_function != activation_functions::Linear && (error_code = cudaGetLastError()) != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch forward activation kernal" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(forward_output, cuda_forward_output, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to host failed" << std::endl;
		exit(error_code);
	}

	cudaFree(cuda_batched_inputs);
	cudaFree(cuda_weights);
	cudaFree(cuda_bias);
	cudaFree(cuda_forward_output);
}

void dense_layer::forward(const layer* prev_layer) {

	if (prev_layer->neurons != inputs) {
		std::cerr << "Error: Prev_layer of invalid input shape to connect to dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	forward(prev_layer->forward_output, prev_layer->neurons, prev_layer->batch_size);
}

double dense_layer::loss(const std::vector<std::vector<double>>& batched_targets) const {

	if (forward_output == nullptr) {
		std::cerr << "Error: No forwad output in dense layer to calculate loss" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Incompatible batch size, cannot calculate loss" << std::endl;
		exit(EXIT_FAILURE);
	}

	double result = 0.0;

	for (int i = 0; i < batch_size; i++) {

		if (batched_targets[i].size() != neurons) {
			std::cerr << "Error: Invalid input shape" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (int j = 0; j < neurons; j++) {
			result += ((forward_output[i * neurons + j] - batched_targets[i][j]) * (forward_output[i * neurons + j] - batched_targets[i][j])) / (double)(batch_size * neurons);
		}
	}

	return result;
}

double dense_layer::loss(const std::vector<int>& batched_targets) const {

	//TODO: activation flag check

	if (forward_output == nullptr) {
		std::cerr << "Error: No forwad output in dense layer to calculate loss" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Incompatible batch size, cannot calculate loss" << std::endl;
		exit(EXIT_FAILURE);
	}
	//TODO: need to check math here again.

	return 0;
}
double dense_layer::loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const {
	
	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	double result = 0.0;
	int idx = 0;

	for (int i = 0; i < batch_size; i++) {

		for (int j = 0; j < batched_targets[i].size(); j++) {
			for (int y = 0; y < batched_targets[i][j].size(); y++) {

				if (idx + batched_targets[i][j][y].size() > neurons) {
					std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
					exit(EXIT_FAILURE);
				}

				for (int x = 0; x < batched_targets[i][j][y].size(); x++) {
					result += (forward_output[batch_size * neurons + idx] - batched_targets[i][j][y][x]) * (forward_output[batch_size * neurons + idx] - batched_targets[i][j][y][x]) / (double)(batch_size * neurons);
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

	return result;
}

void dense_layer::init_back_propigation(const std::vector<std::vector<double>>& batched_targets) {
	
	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = (double*)malloc(batch_size * neurons * sizeof(double));

	if (input_arr == nullptr) {
		std::cerr << "Error: Could not allocated memory in dense layer for backpropigation" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (size_t i = 0; i < batch_size; i++) {
		
		if (batched_targets[i].size() != neurons) {
			std::cerr << "Error: batched targets of invalid shape" << std::endl;
			exit(EXIT_FAILURE);
		}
		memcpy(input_arr + i * neurons, batched_targets[i].data(), neurons * sizeof(double));
	}

	init_back_propigation(input_arr, neurons, batch_size);

	free(input_arr);
}

void dense_layer::init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {
	
	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = (double*)malloc(batch_size * neurons * sizeof(double));
	unsigned int current_size = 0;
	
	if (input_arr == nullptr) {
		std::cerr << "Error: Could not allocate memory for backpropigation in dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < batch_size; i++) {

		for (int j = 0; j < batched_targets[i].size(); j++) {

			for (int y = 0; y < batched_targets[i][j].size(); y++) {

				if (current_size + batched_targets[i][j][y].size() > neurons) {
					std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
					exit(EXIT_FAILURE);
				}
				
				memcpy(input_arr + i * neurons +  current_size, batched_targets[i][j][y].data(), batched_targets[i][j][y].size() * sizeof(double));

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

	free(input_arr);
}

void dense_layer::init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) {
	
	//probally not worth it too load this into a kenral but were going to do it anyway.

	if (batch_size != _batch_size || _input_size != neurons) {
		std::cerr << "Error: Invalid input size for dense layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (backward_input == nullptr) {
		backward_input = (double*)malloc(batch_size * neurons * sizeof(double));
		if (backward_input == nullptr) {
			std::cerr << "Error: Could not allocate memory in dense layer for backpropigation" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	double* cuda_backward_input = nullptr;
	double* cuda_batched_targets = nullptr;
	double* cuda_forward_output = nullptr;
	cudaError error_code;

	error_code = cudaMalloc((void**)&cuda_backward_input, neurons * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&cuda_batched_targets, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}
	 
	error_code = cudaMalloc((void**)&cuda_forward_output, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(cuda_batched_targets, batched_targets, batch_size * neurons * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to Device failed" << std::endl;
		exit(error_code);
	}
	error_code = cudaMemcpy(cuda_forward_output, forward_output, batch_size * neurons * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to Device failed" << std::endl;
		exit(error_code);
	}

	dim3 blocks(neurons/16 + 1, batch_size/16 + 1);
	dim3 threads(16, 16);

	Cuda_Dense_Layer_Init_Back_Propigation<<<blocks, threads>>>(cuda_batched_targets, cuda_forward_output, cuda_backward_input, batch_size, neurons);
	
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch init back propigation kernal in dense_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}

	if (layer_activation_function == activation_functions::Sigmoid) {
		Cuda_Sigmoid_Activation_Backward_Pass<<<blocks, threads>>>(cuda_backward_input, cuda_forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Rectified_Linear) {
		Cuda_Rectified_Linear_Activation_Backward_Pass<<<blocks, threads>>>(cuda_backward_input, cuda_forward_output, batch_size, neurons);
	}

	if (layer_activation_function != activation_functions::Linear && (error_code = cudaGetLastError()) != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch backward activation function kernal" << std::endl;
		exit(error_code);
	}

	if (layer_activation_function != activation_functions::Linear && (error_code = cudaDeviceSynchronize()) != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(backward_input, cuda_backward_input, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcy to Host failed" << std::endl;
	}
	
	cudaFree(cuda_backward_input);
	cudaFree(cuda_batched_targets);
	cudaFree(cuda_forward_output);
}

void dense_layer::backward(const std::vector<std::vector<double>>& batched_inputs) {
	
	if (batch_size != batched_inputs.size()) {
		std::cerr << "Error: Incompatible batch size" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = (double*)malloc(batch_size * inputs * sizeof(double));

	if (input_arr == nullptr) {
		std::cerr << "Error: Could not allocate memory in dense_layer for backward pass" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < batch_size; i++) {
		if (batched_inputs[i].size() != inputs) {
			std::cerr << "Error: Batched inputs of invalid shape" << std::endl;
			exit(EXIT_FAILURE);
		}
		memcpy(input_arr + i * inputs, batched_inputs[i].data(), inputs * sizeof(double));
	}

	backward(input_arr, inputs, batch_size);
	
	free(input_arr);
}

void dense_layer::backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {
	
	if (batched_inputs.size() != batch_size) {
		std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = (double*)malloc(batch_size * inputs * sizeof(double));
	unsigned int current_size = 0;

	if (input_arr == nullptr) {
		std::cerr << "Error: Unable to allocate memory in dense layer for backward pass" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < batch_size; i++) {

		for (int j = 0; j < batched_inputs[i].size(); j++) {
			for (int y = 0; y < batched_inputs[i][j].size(); y++) {

				if (current_size + batched_inputs[i][j][y].size() > inputs) {
					std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
					exit(EXIT_FAILURE);
				}

				memcpy(input_arr + i * inputs + current_size, batched_inputs[i][j][y].data(), batched_inputs[i][j][y].size());

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
	free(input_arr);
}

void dense_layer::backward(double* batched_inputs, size_t _input_size, size_t _batch_size) {

	if (_batch_size != batch_size || _input_size != inputs) {
		std::cerr << "Error: Invalid input size for backward pass" << std::endl;
	}

	if (backward_input == nullptr) {
		std::cerr << "Error: Dense_layer not intialized" << std::endl;
	}
	
	double* cuda_batched_inputs = nullptr;
	double* cuda_backward_inputs = nullptr;
	double* cuda_d_weights = nullptr;
	double* cuda_d_bias = nullptr;
	cudaError error_code;

	error_code = cudaMalloc((void**)&cuda_batched_inputs, batch_size * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&cuda_backward_inputs, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&cuda_d_weights, neurons * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&cuda_d_bias, neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}
	
	error_code = cudaMemcpy(cuda_batched_inputs, batched_inputs, batch_size * inputs * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(cuda_backward_inputs, backward_input, batch_size * neurons * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed" << std::endl;
		exit(error_code);
	}

	int block_size = (batch_size > neurons && batch_size > inputs) ? ((int)batch_size + 16 - 1) / 16 : ((neurons > inputs) ? ((int)neurons + 16 - 1) / 16 : ((int)inputs + 16 - 1) / 16);
	dim3 blocks(block_size, block_size);
	dim3 threads(16, 16);
	
	Cuda_Dense_Layer_First_Backward_Pass<<<blocks, threads>>>(cuda_batched_inputs, cuda_backward_inputs, cuda_d_weights, batch_size, neurons, inputs);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: First backward pass kernal failed to launch" << std::endl;
		exit(error_code);
	}
	
	Cuda_Dense_Layer_Second_Backward_Pass<<<neurons/16 + 1, 16>>>(cuda_backward_inputs, cuda_d_bias, batch_size, neurons);

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

	error_code = cudaMemcpy(d_weights, cuda_d_weights, neurons * inputs * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to host failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(d_bias, cuda_d_bias, neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to host failed" << std::endl;
		exit(error_code);
	}

	cudaFree(cuda_batched_inputs);
	cudaFree(cuda_backward_inputs);
	cudaFree(cuda_d_weights);
	cudaFree(cuda_d_bias);
}

void dense_layer::backward(layer* prev_layer) {
	
	if (prev_layer->batch_size != batch_size || prev_layer->neurons != inputs) {
		std::cerr << "Error: Prev_layer of invalid input shape or batch size to connect to dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	backward(prev_layer->forward_output, prev_layer->neurons, batch_size);
	
	if (prev_layer->backward_input == nullptr) {
		prev_layer->backward_input = (double*)malloc(batch_size * inputs * sizeof(double));
		if (prev_layer->backward_input == nullptr) {
			std::cerr << "Error: Could not allocated memory in dense layer for backpropigation" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	double* cuda_weights = nullptr;
	double* cuda_backward_input = nullptr;
	double* cuda_prev_layer_backward_input = nullptr;
	double* cuda_prev_layer_forward_output = nullptr;
	cudaError error_code;

	error_code = cudaMalloc((void**)&cuda_weights, neurons * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&cuda_backward_input, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}
	
	error_code = cudaMalloc((void**)&cuda_prev_layer_backward_input, batch_size * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(cuda_weights, weights, neurons * inputs * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(cuda_backward_input, backward_input, batch_size * neurons * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed" << std::endl;
		exit(error_code);
	}

	int block_size = (batch_size > inputs) ? ((int)batch_size + 16 - 1) / 16 : ((int)inputs + 16 - 1) / 16;
	dim3 blocks(block_size, block_size);
	dim3 threads(16, 16);

	Cuda_Partial_Derivitive_of_Loss<<<blocks, threads>>>(cuda_backward_input, cuda_weights, cuda_prev_layer_backward_input, batch_size, inputs, neurons);
	
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to Launch Partial Derivitive of Loss Kernal" << std::endl;
		exit(error_code);
	}

	if (prev_layer->layer_activation_function != activation_functions::Linear) {
		error_code = cudaMalloc((void**)&cuda_prev_layer_forward_output, batch_size * inputs * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed" << std::endl;
			exit(error_code);
		}

		error_code = cudaMemcpy(cuda_prev_layer_forward_output, prev_layer->forward_output ,batch_size * inputs * sizeof(double), cudaMemcpyHostToDevice);
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemcpy to device failed" << std::endl;
			exit(error_code);
		}
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}

	if (prev_layer->layer_activation_function == activation_functions::Sigmoid) {
		Cuda_Sigmoid_Activation_Backward_Pass<<<blocks, threads>>>(cuda_prev_layer_backward_input, cuda_prev_layer_forward_output, batch_size, inputs);
	}
	else if (prev_layer->layer_activation_function == activation_functions::Rectified_Linear) {
		Cuda_Rectified_Linear_Activation_Backward_Pass<<<blocks, threads>>>(cuda_prev_layer_backward_input, cuda_prev_layer_forward_output, batch_size, inputs);
	}

	if (prev_layer->layer_activation_function != activation_functions::Linear) {
		error_code = cudaGetLastError();
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: Failed to launch activation function backward pass kernals" << std::endl;
			exit(error_code);
		}
		error_code = cudaDeviceSynchronize();
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
			exit(error_code);
		}
	}

	error_code = cudaMemcpy(prev_layer->backward_input, cuda_prev_layer_backward_input, batch_size * inputs * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to host failed" << std::endl;
		exit(error_code);
	}

	if (cuda_prev_layer_forward_output != nullptr) cudaFree(cuda_prev_layer_forward_output);
	cudaFree(cuda_prev_layer_backward_input);
	cudaFree(cuda_weights);
	cudaFree(cuda_backward_input);
}

void dense_layer::update_paramters(double learning_rate) {

	for (size_t i = 0; i < neurons * inputs; i++) {
		weights[i] -= d_weights[i] * learning_rate;
	}

	for (size_t i = 0; i < neurons; i++) {
		bias[i] -= d_bias[i] * learning_rate;
	}
}