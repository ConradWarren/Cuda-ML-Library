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

max_pooling_layer::max_pooling_layer() {
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
}

max_pooling_layer::max_pooling_layer(size_t _input_size, size_t _channels, size_t _kernal_size, size_t _stride) {

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
}

max_pooling_layer::~max_pooling_layer() {
	free(forward_output);
	free(backward_input);
}

void max_pooling_layer::forward(const std::vector<std::vector<double>>& batched_inputs) {

	double* input_arr = (double*)malloc(batched_inputs.size() * inputs * sizeof(double));

	if (input_arr == nullptr) {
		std::cerr << "Error: Unable to allocated memory in max_pooling layer for forward pass" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < batched_inputs.size(); i++) {

		if (batched_inputs[i].size() != inputs) {
			std::cerr << "Error: Batched_inputs of invalid input shape for max_pooling layer" << std::endl;
			exit(EXIT_FAILURE);
		}
		
		memcpy(input_arr + i * inputs, batched_inputs[i].data(), inputs * sizeof(double));
	}

	forward(input_arr, inputs, batched_inputs.size());
	free(input_arr);
}
void max_pooling_layer::forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {

	double* input_arr = (double*)malloc(batched_inputs.size() * channels * input_size * input_size * sizeof(double));

	if (input_arr == nullptr) {
		std::cerr << "Error: Unable to allocated memory in max_pooling layer for forward pass" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < batched_inputs.size(); i++) {

		if (batched_inputs[i].size() != channels) {
			std::cerr << "Error: Batched_inputs of invalid input shape for max_pooling layer" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (int j = 0; j < channels; j++) {

			if (batched_inputs[i][j].size() != input_size) {
				std::cerr << "Error: Batched_inputs of invalid input shape for max_pooling layer" << std::endl;
				exit(EXIT_FAILURE);
			}

			for (int y = 0; y < input_size; y++) {

				if (batched_inputs[i][j][y].size() != input_size) {
					std::cerr << "Error: Batched_inputs of invalid input shape for max_pooling layer" << std::endl;
					exit(EXIT_FAILURE);
				}

				memcpy(input_arr + i * inputs + j * input_size * input_size + y * input_size, batched_inputs[i][j][y].data(), input_size * sizeof(double));
			}
		}
	}
	forward(input_arr, inputs, batched_inputs.size());
	free(input_arr);
}

void max_pooling_layer::forward(double* batched_inputs, size_t _input_size, size_t _batch_size) {

	if (_input_size != inputs) {
		std::cerr << "Error: Batched_inputs of invalid input shape for max_pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (_batch_size != batch_size) {

		if (forward_output != nullptr) free(forward_output);
		if (backward_input != nullptr) free(backward_input);

		forward_output = (double*)malloc(_batch_size * inputs * sizeof(double));

		if (forward_output == nullptr) {
			std::cerr << "Error: Unable to allocate memory in max_pooling_layer for forward pass" << std::endl;
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

	Cuda_Max_Pooling_Layer_Forward_Pass<<<blocks, threads>>>(cuda_batched_input, cuda_forward_output, batch_size, channels, input_size, kernal_size, output_size, stride);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to lauch forward pass kernal in max_pooling_layer" << std::endl;
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
void max_pooling_layer::forward(const layer* prev_layer) {
	forward(prev_layer->forward_output, prev_layer->neurons, prev_layer->batch_size);
}


