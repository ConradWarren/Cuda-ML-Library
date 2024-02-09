#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "layer.hpp"

#include <iostream>

__global__ static void Cuda_Max_Pooling_Layer_Forward_Pass(double* batched_inputs, double* forward_output, 
	size_t batch_size, size_t channels, size_t input_size, size_t kernal_size, size_t output_size, size_t stride) {
	
	size_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t position_idx = blockIdx.z * blockDim.z + threadIdx.z;

	if (batch_idx < batch_size && channel_idx < channels && position_idx < (output_size * output_size)) {

		int idx = batch_idx * channels * output_size * output_size + channel_idx * output_size * output_size + position_idx;
		int input_position_y = (position_idx / output_size) * stride;
		int input_position_x = (position_idx % output_size) * stride;

		forward_output[idx] = batched_inputs[batch_idx * channels * input_size * input_size + channel_idx * input_size * input_size + input_position_y * input_size + input_position_x];
		
		for (int i = 0; i < kernal_size; i++) {
			for (int j = 0; j < kernal_size; j++) {
				if (batched_inputs[batch_idx * channels * input_size * input_size + channel_idx * input_size * input_size + (input_position_y + i) * input_size + (input_position_x + j)] > forward_output[idx]) {
					forward_output[idx] = batched_inputs[batch_idx * channels * input_size * input_size + channel_idx * input_size * input_size + (input_position_y + i) * input_size + (input_position_x + j)];
				}
			}
		}
	}
}

__global__ static void Cuda_Average_Pooling_Layer_Forward_Pass(double* batched_inputs, double* forward_output,
	size_t batch_size, size_t channels, size_t input_size, size_t kernal_size, size_t output_size, size_t stride) {
	
	size_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t position_idx = blockIdx.z * blockDim.z + threadIdx.z;

	if (batch_idx < batch_size && channel_idx < channels && position_idx < (output_size * output_size)) {

		int idx = batch_idx * channels * output_size * output_size + channel_idx * output_size * output_size + position_idx;
		int input_position_y = (position_idx / output_size) * stride;
		int input_position_x = (position_idx % output_size) * stride;

		forward_output[idx] = 0.0;

		for (int i = 0; i < kernal_size; i++) {
			for (int j = 0; j < kernal_size; j++) {
				forward_output[idx] += batched_inputs[batch_idx * channels * input_size * input_size + channel_idx * input_size * input_size + (input_position_y + i) * kernal_size + input_position_x + j] / (double)(kernal_size * kernal_size);
			}
		}
	}
}

__global__ static void Cuda_Pooling_Layer_Init_Backpropigation(double* forward_output, double* batched_targets, double* backward_input, size_t batch_size, size_t neurons) {

	size_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t neuron_idx = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (batch_idx < batch_size && neuron_idx < neurons) {
		backward_input[batch_idx * neurons + neuron_idx] = (2 * (forward_output[batch_idx * neurons + neuron_idx] - batched_targets[batch_idx * neurons + neuron_idx])) / (double)(batch_size * neurons);
	}
}

__global__ static void Cuda_Max_Pooling_Layer_Partial_Derivitive_of_Loss(double* prev_layer_forward_output, double* backward_input, double* prev_layer_backward_input,
												size_t batch_size, size_t channels, size_t output_size, size_t input_size, size_t kernal_size, size_t stride) {

	size_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t position_idx = blockIdx.z * blockDim.z + threadIdx.z;

	if (batch_idx < batch_size && channel_idx < channels && position_idx < (input_size * input_size)) {

		int idx = batch_idx * channels * input_size * input_size + channel_idx * input_size * input_size + position_idx;
		int position_y_idx = position_idx/input_size;
		int position_x_idx = position_idx % input_size;
		prev_layer_backward_input[idx] = 0.0;
		
		for (int y = 0; y < kernal_size; y++) {
			
			if (position_y_idx - y < 0 || (position_y_idx - y) % stride != 0 || position_y_idx - y + kernal_size - 1 >= input_size) {
				continue;
			}
			
			for (int x = 0; x < kernal_size; x++) {

				if (position_x_idx - x < 0 || (position_x_idx - x) % stride != 0 || position_x_idx - x + kernal_size - 1 >= input_size) {
					continue;
				}

				int corner_pos_y = position_y_idx - y;
				int corner_pos_x = position_x_idx - x;
				int max_idx = batch_idx * channels * input_size * input_size + channel_idx * input_size * input_size + corner_pos_y * input_size + corner_pos_x;
				for (int i = 0; i < kernal_size * kernal_size; i++) {
					int test = batch_idx * channels * input_size * input_size + channel_idx * input_size * input_size + ((corner_pos_y + (i / (int)kernal_size)) * input_size) + (corner_pos_x + (i % (int)kernal_size));
					if (prev_layer_forward_output[test] > prev_layer_forward_output[max_idx]) {
						max_idx = test;
					}
				}
				
				if (max_idx == idx) {
					int output_idx = (corner_pos_y / stride) * output_size + corner_pos_x / stride;
					prev_layer_backward_input[idx] += backward_input[batch_idx * channels * output_size * output_size + channel_idx * output_size * output_size + output_idx];
				}
			}
		}
	}
}

__global__ static void Cuda_Average_Pooling_Layer_Partial_Derivitive_of_Loss(double* prev_layer_forward_output, double* backward_input, double* prev_layer_backward_input,
	size_t batch_size, size_t channels, size_t output_size, size_t input_size, size_t kernal_size, size_t stride) {

	size_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t position_idx = blockIdx.z * blockDim.z + threadIdx.z;

	if (batch_idx < batch_size && channel_idx < channels && position_idx < (input_size * input_size)) {
		
		int idx = batch_idx * channels * input_size * input_size + channel_idx * input_size * input_size + position_idx;
		int position_idx_y = position_idx / input_size;
		int position_idx_x = position_idx % input_size;
		prev_layer_backward_input[idx] = 0.0;

		for (int y = 0; y < kernal_size; y++) {
			
			if ((position_idx_y - y) < 0 || (position_idx_y - y - 1 + kernal_size) >= input_size || (position_idx_y - y) % stride != 0) {
				continue;
			}
			
			for (int x = 0; x < kernal_size; x++) {

				if ((position_idx_x - x) < 0 || (position_idx_x - x - 1 + kernal_size) >= input_size || (position_idx_x - x) % stride != 0) {
					continue;
				}

				int output_idx = ((position_idx_y - y) / stride) * output_size + (position_idx_x - x)/stride;
				prev_layer_backward_input[idx] += backward_input[batch_idx * channels * output_size * output_size + channel_idx * output_size * output_size + output_idx] / (double)(kernal_size * kernal_size);
			}
		}

	}

}

pooling_layer::pooling_layer() {
	batch_size = 0;
	input_size = 0;
	channels = 0;
	kernal_size = 0;
	stride = 0;
	inputs = 0;
	output_size = 0;
	neurons = 0;

	forward_output = nullptr;
	backward_input = nullptr;
	layer_activation_function = activation_functions::Linear;
	pooling_layer_type = pooling_type::Max;
}

pooling_layer::pooling_layer(size_t _input_size, size_t _channels, size_t _kernal_size, size_t _stride, pooling_type layer_type) {

	batch_size = 0;
	input_size = _input_size;
	channels = _channels;
	kernal_size = _kernal_size;
	stride = _stride;

	inputs = channels * input_size * input_size;
	output_size = (input_size - kernal_size) / stride + 1;
	neurons = output_size * output_size * channels;

	forward_output = nullptr;
	backward_input = nullptr;
	layer_activation_function = activation_functions::Linear;
	pooling_layer_type = layer_type;
}

pooling_layer::~pooling_layer() {
	free(forward_output);
	free(backward_input);
}

void pooling_layer::forward(const std::vector<std::vector<double>>& batched_inputs) {

	double* input_arr = (double*)malloc(batched_inputs.size() * inputs * sizeof(double));

	if (input_arr == nullptr) {
		std::cerr << "Error: Unable to allocated memory in pooling layer for forward pass" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < batched_inputs.size(); i++) {

		if (batched_inputs[i].size() != inputs) {
			std::cerr << "Error: Batched_inputs of invalid input shape for pooling layer" << std::endl;
			exit(EXIT_FAILURE);
		}
		
		memcpy(input_arr + i * inputs, batched_inputs[i].data(), inputs * sizeof(double));
	}

	forward(input_arr, inputs, batched_inputs.size());
	free(input_arr);
}
void pooling_layer::forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {

	double* input_arr = (double*)malloc(batched_inputs.size() * channels * input_size * input_size * sizeof(double));

	if (input_arr == nullptr) {
		std::cerr << "Error: Unable to allocated memory in pooling layer for forward pass" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < batched_inputs.size(); i++) {

		if (batched_inputs[i].size() != channels) {
			std::cerr << "Error: Batched_inputs of invalid input shape for pooling layer" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (int j = 0; j < channels; j++) {

			if (batched_inputs[i][j].size() != input_size) {
				std::cerr << "Error: Batched_inputs of invalid input shape for pooling layer" << std::endl;
				exit(EXIT_FAILURE);
			}

			for (int y = 0; y < input_size; y++) {

				if (batched_inputs[i][j][y].size() != input_size) {
					std::cerr << "Error: Batched_inputs of invalid input shape for pooling layer" << std::endl;
					exit(EXIT_FAILURE);
				}

				memcpy(input_arr + i * inputs + j * input_size * input_size + y * input_size, batched_inputs[i][j][y].data(), input_size * sizeof(double));
			}
		}
	}
	forward(input_arr, inputs, batched_inputs.size());
	free(input_arr);
}

void pooling_layer::forward(double* batched_inputs, size_t _input_size, size_t _batch_size) {

	if (_input_size != inputs) {
		std::cerr << "Error: Batched_inputs of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (_batch_size != batch_size) {

		if (forward_output != nullptr) free(forward_output);
		if (backward_input != nullptr) free(backward_input);

		forward_output = (double*)malloc(_batch_size * inputs * sizeof(double));

		if (forward_output == nullptr) {
			std::cerr << "Error: Unable to allocate memory in pooling_layer for forward pass" << std::endl;
			exit(EXIT_FAILURE);
		}

		batch_size = _batch_size;
	}

	double* cuda_batched_input = nullptr;
	double* cuda_forward_output = nullptr;
	cudaError error_code;

	error_code = cudaMalloc((void**)&cuda_batched_input, batch_size * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}
	
	error_code = cudaMalloc((void**)&cuda_forward_output, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(cuda_batched_input, batched_inputs, batch_size * inputs * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed" << std::endl;
		exit(error_code);
	}

	dim3 blocks(batch_size / 6 + 1, channels / 6 + 1, (output_size * output_size) / 6 + 1);
	dim3 threads(6, 6, 6);

	if(pooling_layer_type == pooling_type::Max){
		Cuda_Max_Pooling_Layer_Forward_Pass<<<blocks, threads>>>(cuda_batched_input, cuda_forward_output, batch_size, channels, input_size, kernal_size, output_size, stride);
	}
	else if (pooling_layer_type == pooling_type::Average) {
		Cuda_Average_Pooling_Layer_Forward_Pass<<<blocks, threads>>>(cuda_batched_input, cuda_forward_output, batch_size, channels, input_size, input_size, output_size, stride);
	}
	
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to lauch forward pass kernal in pooling_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(forward_output, cuda_forward_output, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to host failed" << std::endl;
		exit(error_code);
	}

	cudaFree(cuda_forward_output);
	cudaFree(cuda_batched_input);
}
void pooling_layer::forward(const layer* prev_layer) {
	forward(prev_layer->forward_output, prev_layer->neurons, prev_layer->batch_size);
}

double pooling_layer::loss(const std::vector<std::vector<double>>& batched_targets) const {
	
	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	double result = 0.0;

	for (size_t i = 0; i < batch_size; i++) {
		
		if (batched_targets[i].size() != neurons) {
			std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (size_t j = 0; j < neurons; j++) {
			result += ((forward_output[i * neurons + j] - batched_targets[i][j]) * (forward_output[i * neurons + j] - batched_targets[i][j])) / (double)(batch_size * neurons);
		}
	}
	
	return result;
}
double pooling_layer::loss(const std::vector<int>& batched_targets) const {
	//need to check softmax math here. 
	return 0.0;
}
double pooling_layer::loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const {

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	double result = 0.0;

	for (size_t i = 0; i < batch_size; i++) {

		if (batched_targets[i].size() != channels) {
			std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (size_t j = 0; j < channels; j++) {

			if (batched_targets[i][j].size() != output_size) {
				std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
				exit(EXIT_FAILURE);
			}

			for (size_t y = 0; y < output_size; y++) {

				if (batched_targets[i][j][y].size() != output_size) {
					std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
					exit(EXIT_FAILURE);
				}

				for (size_t x = 0; x < output_size; x++) {
					result += ((forward_output[i * channels * output_size * output_size + j * output_size * output_size + y * output_size + x] * batched_targets[i][j][y][x]) * (forward_output[i * channels * output_size * output_size + j * output_size * output_size + y * output_size + x] * batched_targets[i][j][y][x])) / (double)(batch_size * neurons);
				}

			}
		}
	}

	return result;
}

void pooling_layer::init_back_propigation(const std::vector<std::vector<double>>& batched_targets) {
	
	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	double* input_arr = (double*)malloc(batch_size * neurons * sizeof(double));

	if (input_arr == nullptr){
		std::cerr << "Error: Could not allocate memory for backward pass in pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (size_t i = 0; i < batch_size; i++) {

		if (batched_targets[i].size() != neurons) {
			std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
			exit(EXIT_FAILURE);
		}

		memcpy(input_arr + i * neurons, batched_targets[i].data(), neurons * sizeof(double));
	}
	
	init_back_propigation(input_arr, neurons, batch_size);
	free(input_arr);
}
void pooling_layer::init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = (double*)malloc(batch_size * neurons * sizeof(double));

	if (input_arr == nullptr) {
		std::cerr << "Error: Could not allocate memory for backward pass in pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (size_t i = 0; i < batched_targets.size(); i++) {

		if (batched_targets[i].size() != channels) {
			std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (size_t j = 0; j < channels; j++) {

			if (batched_targets[i][j].size() != output_size) {
				std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
				exit(EXIT_FAILURE);
			}

			for (size_t y = 0; y < output_size; y++) {

				if (batched_targets[i][j][y].size() != output_size) {
					std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
					exit(EXIT_FAILURE);
				}

				memcpy(input_arr + i * neurons + j * channels * output_size * output_size + y * output_size, batched_targets[i][j][y].data(), output_size * sizeof(double));
			}
		}
	}

	init_back_propigation(input_arr, neurons, batch_size);
	free(input_arr);
}
void pooling_layer::init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) {

	if (batch_size != _batch_size || _input_size != neurons) {
		std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (backward_input == nullptr) {
		backward_input = (double*)malloc(batch_size * neurons * sizeof(double));
		if (backward_input == nullptr) {
			std::cerr << "Error: Could not allocate memory for backward pass in pooling_layer" << std::endl;
			exit(EXIT_FAILURE);
		}
	
	}

	double* cuda_batched_targets = nullptr;
	double* cuda_forward_output = nullptr;
	double* cuda_backward_input = nullptr;
	cudaError error_code;

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

	error_code = cudaMalloc((void**)&cuda_backward_input, batch_size * neurons * sizeof(double));
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

	dim3 blocks(batch_size / 16 + 1, neurons / 16 + 1);
	dim3 threads(16, 16);
	
	Cuda_Pooling_Layer_Init_Backpropigation<<<blocks, threads>>>(cuda_forward_output, cuda_batched_targets, cuda_backward_input, batch_size, neurons);
	
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch init_backpropigation kernal in pooling layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
	
	error_code = cudaMemcpy(backward_input, cuda_backward_input, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to host failed" << std::endl;
		exit(error_code);
	}
	
	cudaFree(cuda_backward_input);
	cudaFree(cuda_forward_output);
	cudaFree(cuda_batched_targets);
}

void pooling_layer::backward(layer* prev_layer) {
	
	if (prev_layer->batch_size != batch_size || prev_layer->neurons != inputs) {
		std::cerr << "Error: Prev_layer of invalid input shape or batch size to connected to pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (prev_layer->backward_input == nullptr) {
		prev_layer->backward_input = (double*)malloc(batch_size * inputs * sizeof(double));
		if (prev_layer->backward_input == nullptr) {
			std::cerr << "Error: Could not allocate memory for backward pass in pooling layer" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	double* cuda_backward_input = nullptr;
	double* cuda_prev_layer_forward_output = nullptr;
	double* cuda_prev_layer_backward_input = nullptr;
	cudaError error_code;

	error_code = cudaMalloc((void**)&cuda_backward_input, batch_size * neurons * sizeof(double));
	if (error_code != cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&cuda_prev_layer_forward_output, batch_size * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&cuda_prev_layer_backward_input, batch_size * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(cuda_backward_input, backward_input, batch_size * neurons * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(cuda_prev_layer_forward_output, prev_layer->forward_output, batch_size * inputs * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to deivice failed" << std::endl;
		exit(error_code);
	}

	dim3 blocks(batch_size/6 + 1, channels/6 + 1, (input_size * input_size)/6 + 1);
	dim3 threads(6, 6, 6);
	
	if (pooling_layer_type == pooling_type::Max) {
		Cuda_Max_Pooling_Layer_Partial_Derivitive_of_Loss<<<blocks, threads>>>(cuda_prev_layer_forward_output, cuda_backward_input, cuda_prev_layer_backward_input, batch_size, channels, output_size, input_size, kernal_size, stride);
	}
	else if (pooling_layer_type == pooling_type::Average) {
		Cuda_Average_Pooling_Layer_Partial_Derivitive_of_Loss<<<blocks, threads>>>(cuda_prev_layer_forward_output, cuda_backward_input, cuda_prev_layer_backward_input, batch_size, channels, output_size, input_size, kernal_size, stride);
	}

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch Pooling layer pairtial derivite of loss kernal" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(prev_layer->backward_input, cuda_prev_layer_backward_input, batch_size * inputs * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to host failed" << std::endl;
		exit(error_code);
	}

	cudaFree(cuda_backward_input);
	cudaFree(cuda_prev_layer_forward_output);
	cudaFree(cuda_prev_layer_backward_input);
}