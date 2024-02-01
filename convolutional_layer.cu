#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "layer.hpp"

#include <iostream>

__global__ void Cuda_Convolutional_Layer_Forward_Pass(double* batched_inputs, double* weights, double* bias, double* forward_output, 
						size_t batch_size, size_t kernals,size_t kernal_size, size_t padding, size_t output_size, size_t channels, size_t input_size, size_t stride) {
	
	size_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t kernal_idx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t position_idx = blockIdx.z * blockDim.z + threadIdx.z;

	if (batch_idx < batch_size && kernal_idx < kernals && position_idx < (output_size * output_size)) {

		size_t neurons = output_size * output_size * kernals;
		forward_output[batch_idx * neurons + kernal_idx * output_size * output_size + position_idx] = 0.0;
		int starting_y_pos = (position_idx / output_size) * stride - padding;
		int starting_x_pos = (position_idx % output_size) * stride - padding;

		for (int y = 0; y < kernal_size; y++) {

			for (int x = 0; x < kernal_size; x++) {

				if (starting_y_pos + y < 0 || starting_y_pos + y >= input_size || starting_x_pos + x < 0 || starting_x_pos + x >= input_size) {
					continue;
				}
				for (int z = 0; z < channels; z++) {
					forward_output[batch_idx * neurons + kernal_idx * output_size * output_size + position_idx] += weights[kernal_idx * channels * kernal_size * kernal_size + z * kernal_size * kernal_size + y * kernal_size + x] * batched_inputs[batch_idx * channels * input_size * input_size + z * input_size * input_size + (starting_y_pos + y) * input_size + (starting_x_pos + x)];
				}	
			}
		}
	}

}

convolutional_layer::convolutional_layer() { 
	weights = nullptr;
	bias = nullptr;
	d_weights = nullptr;
	d_bias = nullptr;
	forward_output = nullptr;
	backward_input = nullptr;
	kernals = 0;
	kernal_size = 0;
	channels = 0;
	padding = 0;
	stride = 0;
	input_size = 0;
	output_size = 0;
	batch_size = 0;
	neurons = 0;
	inputs = 0;
}

convolutional_layer::convolutional_layer(size_t _input_size, size_t _channels, size_t _kernals, size_t _kernal_size, size_t _stride, size_t _padding) {

	kernals = _kernals;
	kernal_size = _kernal_size;
	channels = _channels;
	padding = _padding;
	stride = _stride;
	input_size = _input_size;
	batch_size = 0;
	
	output_size = (input_size + (2 * padding) - kernal_size) / stride + 1;
	neurons = output_size * output_size * kernals;
	inputs = input_size * input_size * channels;

	weights = (double*)malloc(kernal_size * kernal_size * channels * kernals * sizeof(double));
	d_weights = (double*)malloc(kernal_size * kernal_size * channels * kernals * sizeof(double));
	bias = (double*)malloc(kernals * sizeof(double));
	d_bias = (double*)malloc(kernals * sizeof(double));

	if (weights == nullptr || d_weights == nullptr || bias == nullptr || d_bias == nullptr) {
		std::cerr << "Error: Could not allocate memory in convolutional layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (size_t i = 0; i < kernals; i++) {
		bias[i] = (double)i;
		for (size_t j = 0; j < kernal_size * kernal_size; j++) {
			weights[i * kernal_size * kernal_size + j] = (double)(i * kernal_size * kernal_size + j);
		}
	}
}

convolutional_layer::~convolutional_layer() {
	free(weights);
	free(d_weights);
	free(bias);
	free(d_bias);
	free(forward_output);
	free(backward_input);
}

void convolutional_layer::forward(const std::vector<std::vector<double>>& batched_inputs) {
	std::cerr << "Error: Auto de-flattening not supported at this time" << std::endl;
	exit(EXIT_FAILURE);
}

void convolutional_layer::forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {
	
	double* input_arr = (double*)malloc(batched_inputs.size() * inputs * sizeof(double));

	if (input_arr == nullptr) {
		std::cerr << "Error: Could not allocate memory in convolutional_layer for forward pass" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (size_t i = 0; i < batched_inputs.size(); i++) {
		if (batched_inputs[i].size() != channels) {
			std::cerr << "Error: Inputs incomptibale shape with convolutional layer" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (size_t j = 0; j < channels; j++) {

			if (batched_inputs[i][j].size() != input_size) {
				std::cerr << "Error: Inputs incomptibale shape with convolutional layer" << std::endl;
				exit(EXIT_FAILURE);
			}

			for (size_t z = 0; z < input_size; z++) {

				if (batched_inputs[i][j][z].size() != input_size) {
					std::cerr << "Error: Inputs incomptibale shape with convolutional layer" << std::endl;
					exit(EXIT_FAILURE);
				}

				memcpy(input_arr + z * input_size + j * input_size * input_size + i * channels * input_size * input_size, batched_inputs[i][j][z].data(), input_size * sizeof(double));
			}
		}
	}

	forward(input_arr, inputs, batched_inputs.size());

	free(input_arr);
}

void convolutional_layer::forward(const layer* prev_layer) {

	if (prev_layer->neurons != inputs) {
		std::cerr << "Error: convolutional_layer of incomptibale shape with connected layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	forward(prev_layer->forward_output, prev_layer->neurons, prev_layer->batch_size);
}

void convolutional_layer::forward(double* batched_inputs, size_t _input_size, size_t _batch_size) {
	
	if (inputs != _input_size) {
		std::cerr << "Error: Incompatible input shape with convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (batch_size != _batch_size) {
		
		if (forward_output != nullptr) free(forward_output);
		forward_output = (double*)malloc(_batch_size * neurons * sizeof(double));

		if (backward_input != nullptr) {
			free(backward_input);
			backward_input = nullptr;
		}

		if (forward_output == nullptr) {
			std::cerr << "Error: Could not allocate memory in convolutional layer for forward pass" << std::endl;
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

	error_code = cudaMalloc((void**)&cuda_weights, kernals * kernal_size * kernal_size * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&cuda_bias, kernals * sizeof(double));
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

	error_code = cudaMemcpy(cuda_weights, weights, kernals * kernal_size * kernal_size * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(cuda_bias, bias, kernals * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to device failed" << std::endl;
		exit(error_code);
	}

	dim3 blocks(1, 1, 1);
	dim3 threads(16, 16, 16);
	Cuda_Convolutional_Layer_Forward_Pass<<<blocks, threads>>>(cuda_batched_inputs, cuda_weights, cuda_bias, cuda_forward_output, batch_size, kernals, kernal_size, padding, output_size, channels, input_size, stride);

	
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: failed to launch convolutional layer forward pass kernal" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
	
	//activation function kernals go here.
	 
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: failed to launch convolutional layer forward activation function kernal" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(forward_output, cuda_forward_output, batch_size * neurons, cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to host failed" << std::endl;
		exit(error_code);
	}

	cudaFree(cuda_batched_inputs);
	cudaFree(cuda_weights);
	cudaFree(cuda_bias);
	cudaFree(cuda_forward_output);
}






