#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "layer.hpp"

#include <random>
#include <iostream>

__global__ static void Cuda_Convolutional_Layer_Forward_Pass(double* batched_inputs, double* weights, double* bias, double* forward_output, 
						size_t batch_size, size_t kernals,size_t kernal_size, size_t padding, size_t output_size, size_t channels, size_t input_size, size_t stride) {
	
	size_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t kernal_idx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t position_idx = blockIdx.z * blockDim.z + threadIdx.z;
	
	if (batch_idx < batch_size && kernal_idx < kernals && position_idx < (output_size * output_size)) {
		
		size_t neurons = output_size * output_size * kernals;
		forward_output[batch_idx * neurons + kernal_idx * output_size * output_size + position_idx] = bias[kernal_idx];
		int starting_y_pos = (position_idx / output_size) * stride - padding;
		int starting_x_pos = (position_idx % output_size) * stride - padding;
		
		for (int y = 0; y < (int)kernal_size; y++) {

			if (starting_y_pos + y < 0) {
				continue;
			}

			if (starting_y_pos + y >= input_size) {
				break;
			}

			for (int x = 0; x < (int)kernal_size; x++) {

				if (starting_x_pos + x < 0){ 
					continue;
				}
				if (starting_x_pos + x >= input_size) {
					break;
				}
				for (int z = 0; z < (int)channels; z++) {
					forward_output[batch_idx * neurons + kernal_idx * output_size * output_size + position_idx] += weights[kernal_idx * channels * kernal_size * kernal_size + (z * kernal_size * kernal_size) + (y * kernal_size) + x] * batched_inputs[batch_idx * channels * input_size * input_size + z * input_size * input_size + (starting_y_pos + y) * input_size + (starting_x_pos + x)];
				}
			}
		}
	}
}

__global__ static void Cuda_Convolutional_Layer_First_Backward_Pass(double* batched_inputs, double* backward_input, double* d_weights, 
									size_t batch_size, size_t kernals, size_t channels, size_t kernal_size, size_t input_size, size_t output_size, size_t stride, size_t padding) {

	size_t kernal_idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t position_idx = blockIdx.z * blockDim.z + threadIdx.z;

	if (kernal_idx < kernals && channel_idx < channels && position_idx < (kernal_size * kernal_size)) {

		int idx = kernal_idx * channels * kernal_size * kernal_size + channel_idx * kernal_size * kernal_size + position_idx;
		d_weights[idx] = 0.0;

		int weight_y_offset = position_idx / (int)kernal_size;
		int weight_x_offset = position_idx % (int)kernal_size;
			
		for (int y = 0; y < (int)output_size; y++) {

			for (int x = 0; x < (int)output_size; x++) {

				int position_y = (y * (int)stride) - (int)padding + weight_y_offset;
				int position_x = (x * (int)stride) - (int)padding + weight_x_offset;

				if (position_y < 0 || position_y >= input_size || position_x < 0 || position_x >= input_size) {
					continue;
				}
				
				for (int i = 0; i < (int)batch_size; i++) {
					d_weights[idx] += backward_input[i * kernals * output_size * output_size + kernal_idx * output_size * output_size + y * output_size + x] * batched_inputs[i * channels * input_size * input_size + channel_idx * input_size * input_size + position_y * input_size + position_x];
				}
			}
		}
	}
}

__global__ static void Cuda_Convolutional_Layer_Second_Backward_Pass(double* backward_input, double* d_bias, size_t batch_size, size_t kernals, size_t output_size) {
	
	size_t kernal_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (kernal_idx < kernals) {

		d_bias[kernal_idx] = 0.0;

		for (size_t i = 0; i < batch_size; i++) {
			for (size_t j = 0; j < output_size * output_size; j++) {
				d_bias[kernal_idx] += backward_input[i * output_size * output_size * kernals + kernal_idx * output_size * output_size + j];
			}
		}
	}
}

__global__ static void Cuda_Convolutional_Layer_Init_Back_Propigation(double* batched_targets, double* forward_ouput, double* backward_input, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		backward_input[batch_idx * neurons + neuron_idx] = 2.0 * (forward_ouput[batch_idx * neurons + neuron_idx] - batched_targets[batch_idx * neurons + neuron_idx]) / (double)(batch_size * neurons);
	}
}

__global__ static void Cuda_Init_Cross_Catigorial_Loss_Back_Propigation(unsigned int* batched_targets, double* forward_output, double* backward_input, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		backward_input[batch_idx * neurons + neuron_idx] = forward_output[batch_idx * neurons + neuron_idx];
		if (neuron_idx == batched_targets[batch_idx]) backward_input[batch_idx * neurons + neuron_idx] -= 1.0;
		backward_input[batch_idx * neurons + neuron_idx] /= (double)(batch_size);
	}
}

__global__ static void Cuda_Convolution_Layer_Partial_Derivitive_of_Loss(double* backward_input, double* weights, double* prev_layer_backward_input,
												size_t batch_size, size_t kernals, size_t channels, size_t kernal_size, size_t input_size, size_t padding, size_t stride, size_t output_size) {
	
	size_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t position_idx = blockIdx.z * blockDim.z + threadIdx.z;

	if (batch_idx < batch_size && channel_idx < channels && position_idx < (input_size * input_size)) {

		int idx = batch_idx * channels * input_size * input_size + channel_idx * input_size * input_size + position_idx;
		int position_y_idx = position_idx / input_size;
		int position_x_idx = position_idx % input_size;
				
		for (int y = 0; y < kernal_size; y++) {
			
			if ((int)(position_y_idx - y + (int)padding) < 0 || position_y_idx + (int)kernal_size - 1 - y >= (int)input_size + (int)padding ||(int)(position_y_idx - y + (int)padding) % stride != 0) {
				continue;
			}
			
			for (int x = 0; x < kernal_size; x++) {

				if ((int)(position_x_idx - x + (int)padding) < 0 || position_x_idx + (int)kernal_size - 1 - x >= (int)input_size + (int)padding ||(int)(position_x_idx - x + (int)padding) % stride != 0) {
					continue;
				}

				int output_y = (position_y_idx - y + padding) / stride;
				int output_x = (position_x_idx - x + padding) / stride;
				
				for (int i = 0; i < kernals; i++) {
					prev_layer_backward_input[idx] += weights[i * channels * kernal_size * kernal_size + channel_idx * kernal_size * kernal_size + y * kernal_size + x] * backward_input[batch_idx * kernals * output_size * output_size + i * output_size * output_size + output_y * output_size + output_x];
				}	
			}
		}
	}
}

__global__ static void Cuda_Sigmoid_Activation_Forward_Pass(double* forward_output, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		forward_output[batch_idx * neurons + neuron_idx] = 1.0 / (1.0 + std::powf(2.71828182846, -1.0 * forward_output[batch_idx * neurons + neuron_idx]));
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
	
	double sum = 0.0;
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

__global__ static void Cuda_Softmax_Activation_Backward_Pass(double* forward_output, double* backward_input, size_t batch_size, size_t neurons){

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

__global__ static void Cuda_Matrix_Addition(double* residual_inputs, double* forward_output, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		forward_output[batch_idx * neurons + neuron_idx] += residual_inputs[batch_idx * neurons + neuron_idx];
	}
}

__global__ static void Cuda_Graident_Decent(double* d_weights, double* d_bias, double* weights, double* bias, double learning_rate, size_t kernals, size_t channels, size_t kernal_size) {
	
	size_t kernal_idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t channel_idx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t position_idx = blockIdx.z * blockDim.z + threadIdx.z;

	if (kernal_idx < kernals && channel_idx < channels && position_idx < (kernal_size * kernal_size)) {

		weights[kernal_idx * channels * kernal_size * kernal_size + channel_idx * kernal_size * kernal_size + position_idx] -= d_weights[kernal_idx * channels * kernal_size * kernal_size + channel_idx * kernal_size * kernal_size + position_idx] * learning_rate;
		if (channel_idx == 0 && position_idx == 0) {
			bias[kernal_idx] -= d_bias[kernal_idx] * learning_rate;
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

convolutional_layer::convolutional_layer(size_t _input_size, size_t _channels, size_t _kernals, size_t _kernal_size, size_t _stride, size_t _padding, activation_functions _layer_activation_function) {

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

	cudaError error_code;

	error_code = cudaMalloc((void**)&d_weights, kernals * channels * kernal_size * kernal_size * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemset(d_weights, 0, kernals * channels * kernal_size * kernal_size * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr <<"Error: cudaMemset failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&d_bias, kernals * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemset(d_bias, 0, kernals * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemset failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&bias, kernals * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMalloc((void**)&weights, kernals * channels * kernal_size * kernal_size * sizeof(double));
	if (error_code != cudaError::cudaSuccess){
		std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	forward_output = nullptr;
	backward_input = nullptr;
	layer_activation_function = _layer_activation_function;

	double* temp_weights = (double*)malloc(kernals * channels * kernal_size * kernal_size * sizeof(double));
	double* temp_bias = (double*)malloc(kernals * sizeof(double));

	if (temp_weights == nullptr || temp_bias == nullptr) {
		std::cerr << "Error: Failed to allocate memory in convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	 
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	std::mt19937 generator;

	for (size_t i = 0; i < kernals; i++) {
		temp_bias[i] = distribution(generator);
		for (size_t j = 0; j < channels * kernal_size * kernal_size; j++) {
			temp_weights[i * kernal_size * kernal_size * channels + j] = distribution(generator);
		}
	}

	error_code = cudaMemcpy(weights, temp_weights, kernals * channels * kernal_size * kernal_size * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpyHostToDeivce failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(bias, temp_bias, kernals * sizeof(double), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Errror: cudaMemcpyHostToDevice failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	free(temp_weights);
	free(temp_bias);

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed in convolutional_layer" << std::endl;
		exit(error_code);
	}
}

convolutional_layer::~convolutional_layer() {
	cudaFree(weights);
	cudaFree(d_weights);
	cudaFree(bias);
	cudaFree(d_bias);
	cudaFree(forward_output);
	cudaFree(backward_input);
}

void convolutional_layer::forward(const std::vector<std::vector<double>>& batched_inputs) {
	
	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batched_inputs.size() * inputs * sizeof(double));

	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	for (size_t i = 0; i < batched_inputs.size(); i++) {

		if (batched_inputs[i].size() != inputs) {
			std::cerr << "Error: Inputs incompatible shape with convolutional_layer" << std::endl;
			exit(EXIT_FAILURE);
		}
		
		error_code = cudaMemcpy(input_arr + i * inputs, batched_inputs[i].data(), inputs * sizeof(double), cudaMemcpyHostToDevice);
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemcpy to deivce failed in convolutional_layer" << std::endl;
			exit(error_code);
		}
	}

	forward(input_arr, inputs, batched_inputs.size());
	cudaFree(input_arr);
}

void convolutional_layer::forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {
	
	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
		exit(error_code);
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

				error_code = cudaMemcpy(input_arr + z * input_size + j * input_size * input_size + i * channels * input_size * input_size, batched_inputs[i][j][z].data(), input_size * sizeof(double), cudaMemcpyHostToDevice);
				if (error_code != cudaError::cudaSuccess) {
					std::cerr << "Error: cudaMemcpy to device failed in convolutional_layer" << std::endl;
					exit(error_code);
				}
			}
		}
	}
	forward(input_arr, inputs, batched_inputs.size());
	cudaFree(input_arr);
}

void convolutional_layer::forward(double* batched_inputs, size_t _input_size, size_t _batch_size) {
	
	if (inputs != _input_size) {
		std::cerr << "Error: Incompatible input shape with convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (batch_size != _batch_size) {
		
		cudaFree(forward_output);
		cudaFree(backward_input);
		backward_input = nullptr;

		error_code = cudaMalloc((void**)&forward_output, _batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
			exit(error_code);
		}
		
		batch_size = _batch_size;
	}

	if (backward_input != nullptr) {
		error_code = cudaMemset(backward_input, 0, batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemset failed in convolutional_layer" << std::endl;
			exit(error_code);
		}
	}

	
	dim3 blocks(batch_size/6 + 1, kernals/6 + 1, (output_size * output_size)/6 + 1);
	dim3 threads(6, 6, 6);
	Cuda_Convolutional_Layer_Forward_Pass<<<blocks, threads>>>(batched_inputs, weights, bias, forward_output, batch_size, kernals, kernal_size, padding, output_size, channels, input_size, stride);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: failed to launch convolutional layer forward pass kernal" << std::endl;
		exit(error_code);
	}

	if (layer_activation_function == activation_functions::Sigmoid) {
		dim3 blocks_2d(neurons/16 + 1, batch_size/16 + 1);
		dim3 threads_2d(16, 16);
		Cuda_Sigmoid_Activation_Forward_Pass<<<blocks_2d, threads_2d>>>(forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Rectified_Linear) {
		dim3 blocks_2d(neurons/16 + 1, batch_size/16 + 1);
		dim3 threads_2d(16, 16);
		Cuda_Rectified_Linear_Activation_Forward_Pass<<<blocks_2d, threads_2d>>>(forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Softmax) {
		dim3 blocks_2d(neurons / 16 + 1, batch_size / 16 + 1);
		dim3 threads_2d(16, 16);
		Cuda_Softmax_Activation_Forward_Pass<<<blocks_2d, threads_2d>>>(forward_output, batch_size, neurons);
	}
	
	if (layer_activation_function != activation_functions::Linear && (error_code = cudaGetLastError()) != cudaError::cudaSuccess) {
		std::cerr << "Error: failed to launch convolutional layer forward activation function kernal" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
}

void convolutional_layer::forward(double* batched_inputs, double* residual_inputs, size_t _input_size, size_t _batch_size) {

	if (_input_size != inputs) {
		std::cerr << "Error: Batched_inputs of invalid shape to connect to dense_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (_batch_size != batch_size) {
		cudaFree(forward_output);
		cudaFree(backward_input);
		backward_input = nullptr;

		error_code = cudaMalloc((void**)&forward_output, _batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
			exit(error_code);
		}

		batch_size = _batch_size;
	}

	if (backward_input != nullptr) {
		error_code = cudaMemset(backward_input, 0, batch_size * neurons * sizeof(double));
		if(error_code != cudaError::cudaSuccess){
			std::cerr << "Error: cudaMemset failed in convolutional_layer" << std::endl;
			exit(error_code);
		}
	}


	dim3 blocks(batch_size/6 + 1, kernals/6 + 1, (output_size * output_size)/6 + 1);
	dim3 threads(6, 6, 6);

	Cuda_Convolutional_Layer_Forward_Pass<<<blocks, threads>>>(batched_inputs, weights, bias, forward_output, batch_size, kernals, kernal_size, padding, output_size, channels, input_size, stride);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch forward pass kernal" << std::endl;
		exit(error_code);
	}

	dim3 blocks_2d(neurons / 16 + 1, batch_size / 16 + 1);
	dim3 threads_2d(16, 16);

	Cuda_Matrix_Addition<<<blocks_2d, threads_2d>>>(residual_inputs, forward_output, batch_size, neurons);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch matrix addition kernal" << std::endl;
		exit(error_code);
	}

	if (layer_activation_function == activation_functions::Sigmoid) {
		Cuda_Sigmoid_Activation_Forward_Pass<<<blocks_2d, threads_2d>>>(forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Rectified_Linear) {
		Cuda_Rectified_Linear_Activation_Forward_Pass<<<blocks_2d, threads_2d>>>(forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Softmax) {
		Cuda_Softmax_Activation_Forward_Pass<<<blocks_2d, threads_2d>>>(forward_output, batch_size, neurons);
	}

	if (layer_activation_function != activation_functions::Linear && (error_code = cudaGetLastError()) != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch forward activation function kernal" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronice failed" << std::endl;
		exit(error_code);
	}
}

void convolutional_layer::forward(const layer* prev_layer) {

	if (prev_layer->neurons != inputs) {
		std::cerr << "Error: Prev_layer of invalid input shape size to connect to convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	forward(prev_layer->forward_output, prev_layer->neurons, prev_layer->batch_size);
}

void convolutional_layer::forward(const layer* prev_layer, const layer* residual_layer) {

	if (prev_layer->neurons != inputs || residual_layer->neurons != neurons || prev_layer->batch_size != residual_layer->batch_size ) {
		std::cerr << "Error: Prev_layer or residual_layer of invalid input shape to connect to convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	forward(prev_layer->forward_output, residual_layer->forward_output, prev_layer->neurons, prev_layer->batch_size);
}


double convolutional_layer::loss(const std::vector<std::vector<double>>& batched_targets) const {

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Incompatible batch size, cannot calculate loss in convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	double result = 0.0;
	double* host_forward_output = (double*)malloc(batch_size * neurons * sizeof(double));

	if (host_forward_output == nullptr) {
		std::cerr << "Error: Failed to allocate memory in convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code = cudaMemcpy(host_forward_output, forward_output, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to host failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	for (size_t i = 0; i < batch_size; i++) {
		
		if (batched_targets[i].size() != neurons) {
			std::cerr << "Error: Incompatible input shape, cannot calculate loss in convolutional_layer" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (size_t j = 0; j < neurons; j++) {
			result += ((host_forward_output[i * neurons + j] - batched_targets[i][j]) * (host_forward_output[i * neurons + j] - batched_targets[i][j])) / (double)(neurons * batch_size);
		}
	}
	free(host_forward_output);
	return result;
}
double convolutional_layer::loss(const std::vector<unsigned int>& batched_targets) const {

	//TODO: move to cross catigorial loss function later.
	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Incompatible batch size, cannot calculate loss in convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (layer_activation_function != activation_functions::Softmax) {
		std::cerr << "Error: Not a classification model" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (forward_output == nullptr) {
		std::cerr << "Error: No forwad output in convolutional_layer to calculate loss" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* host_forward_ouput = (double*)malloc(batch_size * neurons * sizeof(double));
	if (host_forward_ouput == nullptr) {
		std::cerr << "Error: Could not allocate memory in convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code = cudaMemcpy(host_forward_ouput, forward_output, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpyDeviceToHost failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	double result = 0.0;

	for (int i = 0; i < batch_size; i++) {
		
		if (batched_targets[i] >= neurons) {
			std::cerr << "Error: Invalid input in batched_targets" << std::endl;
			exit(EXIT_FAILURE);
		}

		host_forward_ouput[i * neurons + batched_targets[i]] = (host_forward_ouput[i * neurons + batched_targets[i]] > 1e-7) ? host_forward_ouput[i * neurons + batched_targets[i]] : 1e-7;
		host_forward_ouput[i * neurons + batched_targets[i]] = (host_forward_ouput[i * neurons + batched_targets[i]] < 1 - 1e-7) ? host_forward_ouput[i * neurons + batched_targets[i]] : 1 - 1e-7;
		result += -std::log(host_forward_ouput[i * neurons + batched_targets[i]])/(double)batch_size;
	}

	free(host_forward_ouput);
	return result;
}

double convolutional_layer::loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const {

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Incompatible batch size, cannot calculate loss in convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	double result = 0.0;
	double* host_forward_output = (double*)malloc(batch_size * neurons * sizeof(double));
	if (host_forward_output == nullptr) {
		std::cerr << "Error: Failed to allocate memory in convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code = cudaMemcpy(host_forward_output, forward_output, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpy to host failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	for (size_t i = 0; i < batch_size; i++) {

		if (batched_targets[i].size() != kernals) {
			std::cerr << "Error: Incompatible input shape, cannot calculate loss in convolutional_layer" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (size_t j = 0; j < kernals; j++) {

			if (batched_targets[i][j].size() != output_size) {
				std::cerr << "Error: Incompatible input shape, cannot calculate loss in convolutional_layer" << std::endl;
				exit(EXIT_FAILURE);
			}

			for (size_t y = 0; y < output_size; y++) {

				if (batched_targets[i][j][y].size() != output_size) {
					std::cerr << "Error: Incompatible input shape, cannot calculate loss in convolutional_layer" << std::endl;
					exit(EXIT_FAILURE);
				}
				
				for (size_t x = 0; x < output_size; x++) {
					result += ((host_forward_output[i * neurons + j * output_size * output_size + y * output_size + x] - batched_targets[i][j][y][x]) * (host_forward_output[i * neurons + j * output_size * output_size + y * output_size + x] - batched_targets[i][j][y][x])) / (double)(neurons * batch_size);	
				}
			}
		}
	}
	free(host_forward_output);
	return result;
}

void convolutional_layer::init_back_propigation(const std::vector<unsigned int>& batched_targets) {

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	unsigned int* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * sizeof(unsigned int));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(input_arr, batched_targets.data(), batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpyHostToDevice failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	init_back_propigation(input_arr, batch_size);
	cudaFree(input_arr);
}

void convolutional_layer::init_back_propigation(const std::vector<std::vector<double>>& batched_targets) {

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	for (size_t i = 0; i < batch_size; i++) {
		
		if (batched_targets[i].size() != neurons) {
			std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
			exit(EXIT_FAILURE);
		}

		error_code = cudaMemcpy(input_arr + i * neurons, batched_targets[i].data(), neurons * sizeof(double), cudaMemcpyHostToDevice);
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemcpy to device failed in convolutional_layer" << std::endl;
			exit(error_code);
		}
	}

	init_back_propigation(input_arr, neurons, batch_size);
	cudaFree(input_arr);
}
void convolutional_layer::init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed" << std::endl;
		exit(error_code);
	}

	for (size_t i = 0; i < batch_size; i++) {

		if (batched_targets[i].size() != kernals) {
			std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (size_t j = 0; j < kernals; j++) {

			if (batched_targets[i][j].size() != output_size) {
				std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
				exit(EXIT_FAILURE);
			}

			for (size_t y = 0; y < output_size; y++) {
				
				if (batched_targets[i][j][y].size() != output_size) {
					std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
					exit(EXIT_FAILURE);
				}

				error_code = cudaMemcpy(input_arr + i * neurons + j * output_size * output_size + y * output_size, batched_targets[i][j][y].data(), output_size * sizeof(double), cudaMemcpyHostToDevice);
				if (error_code != cudaError::cudaSuccess) {
					std::cerr << "Error: cudaMemcpy to device failed in convolutional_layer" << std::endl;
					exit(error_code);
				}
			}
		}
	}

	init_back_propigation(input_arr, neurons, batch_size);
	cudaFree(input_arr);
}
void convolutional_layer::init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) {

	if (_input_size != neurons || _batch_size != batch_size) {
		std::cerr << "Error: Batched_targets of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (backward_input == nullptr) {
		error_code = cudaMalloc((void**)&backward_input, batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
			exit(error_code);
		}
	}

	dim3 blocks(neurons / 16 + 1, batch_size / 16 + 1);
	dim3 threads(16, 16);
	
	Cuda_Convolutional_Layer_Init_Back_Propigation<<<blocks, threads>>>(batched_targets, forward_output, backward_input, batch_size, neurons);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch init back propigation kernal in convolutional_layer" << std::endl;
		exit(error_code);
	}

	if (layer_activation_function == activation_functions::Sigmoid) {
		Cuda_Sigmoid_Activation_Backward_Pass<<<blocks, threads>>>(backward_input, forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Rectified_Linear) {
		Cuda_Rectified_Linear_Activation_Backward_Pass<<<blocks, threads>>>(backward_input, forward_output, batch_size, neurons);
	}
	else if (layer_activation_function == activation_functions::Softmax) {
		Cuda_Softmax_Activation_Backward_Pass<<<blocks, threads>>>(forward_output, backward_input, batch_size, neurons);
	}

	if (layer_activation_function != activation_functions::Linear && (error_code = cudaGetLastError()) != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch backward activation function kernal" << std::endl;
		exit(error_code);
	}
	
	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
}

void convolutional_layer::init_back_propigation(unsigned int* batched_targets, size_t _batch_size) {

	if (_batch_size != batch_size) {
		std::cerr << "Error: Invalid input size for convolutional_layer" << std::endl;
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
			std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
			exit(error_code);
		}
	}

	dim3 blocks(neurons / 16 + 1, batch_size / 16 + 1);
	dim3 threads(16, 16);

	Cuda_Init_Cross_Catigorial_Loss_Back_Propigation<<<blocks, threads>>>(batched_targets, forward_output, backward_input, batch_size, neurons);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to Launch Init_Cross_Catigorial_Loss_Back_Propigation in convolutional_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
}

void convolutional_layer::backward(const std::vector<std::vector<double>>& batched_inputs) {
	
	if (batched_inputs.size() != batch_size) {
		std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	for (size_t i = 0; i < batch_size; i++) {

		if (batched_inputs[i].size() != inputs) {
			std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
			exit(EXIT_FAILURE);
		}

		error_code = cudaMemcpy(input_arr + i * inputs, batched_inputs[i].data(), inputs * sizeof(double), cudaMemcpyHostToDevice);
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemcpy failed in convolutional_layer" << std::endl;
			exit(error_code);
		}
	}

	backward(input_arr, inputs, batch_size);
	cudaFree(input_arr);
}

void convolutional_layer::backward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {

	if (batched_inputs.size() != batch_size) {
		std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	for (size_t i = 0; i < batch_size; i++) {
		
		if (batched_inputs[i].size() != channels) {
			std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (size_t j = 0; j < channels; j++) {

			if (batched_inputs[i][j].size() != input_size) {
				std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
				exit(EXIT_FAILURE);
			}

			for (size_t y = 0; y < input_size; y++) {

				if (batched_inputs[i][j][y].size() != input_size) {
					std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
					exit(EXIT_FAILURE);
				}

				error_code = cudaMemcpy(input_arr + i * inputs + j * input_size * input_size + y * input_size, batched_inputs[i][j][y].data(), input_size * sizeof(double), cudaMemcpyHostToDevice);
				if (error_code != cudaError::cudaSuccess) {
					std::cerr << "Error: cudaMemcpy to deivce failed in convolutional_layer" << std::endl;
					exit(error_code);
				}
			}
		}
	}
	  
	backward(input_arr, inputs, batch_size);
	cudaFree(input_arr);
}
void convolutional_layer::backward(double* batched_inputs, size_t _input_size, size_t _batch_size) {

	if (backward_input == nullptr) {
		std::cerr << "Error: Convolutional_layer not initialized for backward pass" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (inputs != _input_size || batch_size != _batch_size) {
		std::cerr << "Error: Batched_inputs of incompatible input shape" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;
	dim3 blocks(kernals/6 + 1, channels/6 + 1, (kernal_size * kernal_size)/6 + 1);
	dim3 threads(6, 6, 6);

	Cuda_Convolutional_Layer_First_Backward_Pass<<<blocks, threads>>>(batched_inputs, backward_input, d_weights, batch_size, kernals, channels, kernal_size, input_size, output_size, stride, padding);
	
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch kernal for backward pass in convolutional_layer" << std::endl;
		exit(error_code);
	}

	Cuda_Convolutional_Layer_Second_Backward_Pass<<<kernals / 16 + 1, 16>>>(backward_input, d_bias, batch_size, kernals, output_size);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch kernal for backward pass in convolutional_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
}
void convolutional_layer::backward(layer* prev_layer) {

	if (prev_layer->batch_size != batch_size || prev_layer->neurons != inputs) {
		std::cerr << "Error: Prev_layer of invalid input shape or batch size to connect to convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	backward(prev_layer->forward_output, prev_layer->neurons, prev_layer->batch_size);

	cudaError error_code;

	if (prev_layer->backward_input == nullptr) {

		error_code = cudaMalloc((void**)&prev_layer->backward_input, batch_size * inputs * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in convolutional_layer" << std::endl;
			exit(error_code);
		}
		
		error_code = cudaMemset(prev_layer->backward_input, 0, batch_size * inputs * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemset failed in convolutional_layer" << std::endl;
			exit(error_code);
		}

		error_code = cudaDeviceSynchronize();
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaDeviceSynchronize failed in convolutional_layer" << std::endl;
			exit(error_code);
		}
	}
	
	dim3 blocks(batch_size/6 + 1, channels/6 + 1, (input_size * input_size)/6 + 1);
	dim3 threads(6, 6, 6);
	
	Cuda_Convolution_Layer_Partial_Derivitive_of_Loss<<<blocks, threads>>>(backward_input, weights, prev_layer->backward_input, batch_size, kernals, channels, kernal_size, input_size, padding, stride, output_size);
	
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch partial derivitive of loss kernal" << std::endl;
		exit(error_code);
	}
	
	if (prev_layer->layer_activation_function != activation_functions::Linear) {

		dim3 blocks_2d(inputs/16 + 1, batch_size/16 + 1);
		dim3 threads_2d(16, 16);

		if (prev_layer->forward_output == nullptr) {
			std::cerr << "Error: prev_layer not initialized" << std::endl;
			exit(EXIT_FAILURE);
		}

		if (prev_layer->layer_activation_function == activation_functions::Sigmoid) {
			Cuda_Sigmoid_Activation_Backward_Pass<<<blocks_2d, threads_2d>>>(prev_layer->backward_input, prev_layer->forward_output, batch_size, inputs);
		}
		else if (prev_layer->layer_activation_function == activation_functions::Rectified_Linear) {
			Cuda_Rectified_Linear_Activation_Backward_Pass<<<blocks_2d, threads_2d>>>(prev_layer->backward_input, prev_layer->forward_output, batch_size, inputs);
		}
		else if (prev_layer->layer_activation_function == activation_functions::Softmax) {
			Cuda_Softmax_Activation_Backward_Pass<<<blocks_2d, threads_2d>>>(prev_layer->forward_output, prev_layer->backward_input, batch_size, inputs);
		}

		error_code = cudaGetLastError();
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: Failed to launch backward activation function kernal" << std::endl;
			exit(error_code);
		}
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed in convolutional_layer" << std::endl;
		exit(error_code);
	}
}

void convolutional_layer::backward(layer* prev_layer, layer* residual_layer) {

	if (prev_layer->batch_size != batch_size || residual_layer->batch_size != batch_size || prev_layer->neurons != inputs || residual_layer->neurons != neurons) {
		std::cerr << "Error: Prev_layer or residual_layer of invalid input shape or batch size to connect to convolutional_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (residual_layer->backward_input == nullptr) {
		error_code = cudaMalloc((void**)&residual_layer->backward_input, batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed" << std::endl;
			exit(error_code);
		}
	}

	error_code = cudaMemcpy(residual_layer->backward_input, backward_input, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpyDeviceToDevice failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed in convolutional_layer" << std::endl;
		exit(error_code);
	}

	backward(prev_layer);	
}

void convolutional_layer::update_paramters(double learning_rate) {

	dim3 blocks(kernals / 6 + 1, channels / 6 + 1, (kernal_size * kernal_size) / 6 + 1);
	dim3 threads(6, 6, 6);

	Cuda_Graident_Decent<<<blocks, threads>>>(d_weights, d_bias, weights, bias, learning_rate, kernals, channels, kernal_size);

	cudaError error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch graident decent kernal in convolutional_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed in convolutional_layer" << std::endl;
		exit(error_code);
	}
}