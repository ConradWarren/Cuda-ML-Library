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
				forward_output[idx] += batched_inputs[batch_idx * channels * input_size * input_size + channel_idx * input_size * input_size + (input_position_y + i) * input_size + input_position_x + j]/(double)(kernal_size * kernal_size);
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

__global__ static void Cuda_Pooling_Layer_Init_Cross_Catigorial_Loss_Back_Propigation(unsigned int* batched_targets, double* forward_output, double* backward_input, size_t batch_size, size_t neurons) {

	size_t batch_idx = (blockIdx.y * blockDim.y) + threadIdx.y;
	size_t neuron_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		backward_input[batch_idx * neurons + neuron_idx] = forward_output[batch_idx * neurons + neuron_idx];
		if (neuron_idx == batched_targets[batch_idx]) backward_input[batch_idx * neurons + neuron_idx] -= 1.0;
		backward_input[batch_idx * neurons + neuron_idx] /= (double)(batch_size);
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

__global__ void static Cuda_Matrix_Addition(double* residual_inputs, double* forward_output, size_t batch_size, size_t neurons) {

	size_t batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
	size_t neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (batch_idx < batch_size && neuron_idx < neurons) {
		forward_output[batch_idx * neurons + neuron_idx] += residual_inputs[batch_idx * neurons + neuron_idx];
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

pooling_layer::pooling_layer(size_t _input_size, size_t _channels, size_t _kernal_size, size_t _stride, pooling_type layer_type, activation_functions _layer_activation_function) {

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
	layer_activation_function = _layer_activation_function;
	pooling_layer_type = layer_type;
}

pooling_layer::~pooling_layer() {
	cudaFree(forward_output);
	cudaFree(backward_input);
}

void pooling_layer::forward(const std::vector<std::vector<double>>& batched_inputs) {

	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batched_inputs.size() * inputs * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in pooling_layer" << std::endl;
		exit(error_code);
	}

	for (int i = 0; i < batched_inputs.size(); i++) {

		if (batched_inputs[i].size() != inputs) {
			std::cerr << "Error: Batched_inputs of invalid input shape for pooling layer" << std::endl;
			exit(EXIT_FAILURE);
		}
		
		error_code = cudaMemcpy(input_arr + i * inputs, batched_inputs[i].data(), inputs * sizeof(double), cudaMemcpyHostToDevice);
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemcpyHostToDevice failed in pooling_layer" << std::endl;
			exit(error_code);
		}
	}

	forward(input_arr, inputs, batched_inputs.size());
	cudaFree(input_arr);
}
void pooling_layer::forward(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_inputs) {

	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batched_inputs.size() * channels * input_size * input_size * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in pooling_layer" << std::endl;
		exit(error_code);
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

				error_code = cudaMemcpy(input_arr + i * inputs + j * input_size * input_size + y * input_size, batched_inputs[i][j][y].data(), input_size * sizeof(double), cudaMemcpyHostToDevice);
				if (error_code != cudaError::cudaSuccess) {
					std::cerr << "Error: cudaMemcpyHostToDevice failed in pooling_layer" << std::endl;
					exit(error_code);
				}
			}
		}
	}
	forward(input_arr, inputs, batched_inputs.size());
	cudaFree(input_arr);
}

void pooling_layer::forward(double* batched_inputs, size_t _input_size, size_t _batch_size) {

	if (_input_size != inputs) {
		std::cerr << "Error: Batched_inputs of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (_batch_size != batch_size) {
		cudaFree(forward_output);
		cudaFree(backward_input);
		backward_input = nullptr;
		error_code = cudaMalloc((void**)&forward_output, _batch_size * inputs * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in pooling_layer" << std::endl;
			exit(error_code);
		}

		batch_size = _batch_size;
	}

	if (backward_input != nullptr) {
		error_code = cudaMemset(backward_input, 0, batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemset failed in pooling_layer" << std::endl;
			exit(error_code);
		}
	}

	dim3 blocks(batch_size / 6 + 1, channels / 6 + 1, (output_size * output_size) / 6 + 1);
	dim3 threads(6, 6, 6);

	if (pooling_layer_type == pooling_type::Max) {
		Cuda_Max_Pooling_Layer_Forward_Pass << <blocks, threads >> > (batched_inputs, forward_output, batch_size, channels, input_size, kernal_size, output_size, stride);
	}
	else if (pooling_layer_type == pooling_type::Average) {
		Cuda_Average_Pooling_Layer_Forward_Pass << <blocks, threads >> > (batched_inputs, forward_output, batch_size, channels, input_size, kernal_size, output_size, stride);
	}

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to lauch forward pass kernal in pooling_layer" << std::endl;
		exit(error_code);
	}

	dim3 blocks_2d(neurons / 16 + 1, batch_size / 16 + 1);
	dim3 threads_2d(16, 16);

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
		std::cerr << "Error: Failed to launch forward activation functions kernal in pooling_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
}

void pooling_layer::forward(double* batched_inputs, double* residual_inputs, size_t _input_size, size_t _batch_size) {

	if (_input_size != inputs) {
		std::cerr << "Error: batched_inputs of invalid shape for pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (_batch_size != batch_size || forward_output == nullptr) {
		cudaFree(forward_output);
		cudaFree(backward_input);
		backward_input = nullptr;
		error_code = cudaMalloc((void**)&forward_output, _batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in pooling_layer" << std::endl;
			exit(error_code);
		}
		batch_size = _batch_size;
	}

	if (backward_input != nullptr) {
		error_code = cudaMemset(backward_input, 0, batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemset failed in pooling_layer" << std::endl;
			exit(error_code);
		}
	}

	dim3 blocks(batch_size / 6 + 1, channels / 6 + 1, (output_size * output_size) / 6 + 1);
	dim3 threads(6, 6, 6);

	if (pooling_layer_type == pooling_type::Max) {
		Cuda_Max_Pooling_Layer_Forward_Pass<<<blocks, threads>>>(batched_inputs, forward_output, batch_size, channels, input_size, kernal_size, output_size, stride);
	}
	else if (pooling_layer_type == pooling_type::Average) {
		Cuda_Average_Pooling_Layer_Forward_Pass<<<blocks, threads>>>(batched_inputs, forward_output, batch_size, channels, input_size, kernal_size, output_size, stride);
	}

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
		std::cerr << "Error: Failed to launch forward activation function kernals in pooling_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
}

void pooling_layer::forward(const layer* prev_layer) {
	
	if (prev_layer->neurons != inputs) {
		std::cerr << "Error: Prev_layer of invalid input shape to connect to pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	forward(prev_layer->forward_output, prev_layer->neurons, prev_layer->batch_size);
}

void pooling_layer::forward(const layer* prev_layer, const layer* residual_layer) {
	
	if (prev_layer->neurons != inputs || residual_layer->neurons != neurons || prev_layer->batch_size != residual_layer->batch_size) {
		std::cerr << "Error: Prev_layer of invalid input shape to connect to pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	forward(prev_layer->forward_output, residual_layer->forward_output, prev_layer->neurons, prev_layer->batch_size);
}

double pooling_layer::loss(const std::vector<std::vector<double>>& batched_targets) const {
	
	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	double result = 0.0;
	double* host_forward_output = (double*)malloc(batch_size * neurons * sizeof(double));
	if (host_forward_output == nullptr) {
		std::cerr << "Error: Failed to allocate memory in pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code = cudaMemcpy(host_forward_output, forward_output, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpyDeviceToHost failed in pooling_layer" << std::endl;
		exit(error_code);
	}

	for (size_t i = 0; i < batch_size; i++) {
		
		if (batched_targets[i].size() != neurons) {
			std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
			exit(EXIT_FAILURE);
		}

		for (size_t j = 0; j < neurons; j++) {
			result += ((host_forward_output[i * neurons + j] - batched_targets[i][j]) * (host_forward_output[i * neurons + j] - batched_targets[i][j])) / (double)(batch_size * neurons);
		}
	}

	free(host_forward_output);
	return result;
}
double pooling_layer::loss(const std::vector<unsigned int>& batched_targets) const {

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (layer_activation_function != activation_functions::Softmax) {
		std::cerr << "Error: Not a classification model" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (forward_output == nullptr) {
		std::cerr << "Error: No forwad output in pooling_layer to calculate loss" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* host_forward_output = (double*)malloc(batch_size * neurons * sizeof(double));
	if (host_forward_output == nullptr) {
		std::cerr << "Error: Failed to allocate memory in pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code = cudaMemcpy(host_forward_output, forward_output, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpyDeviceToHost failed in pooling_layer" << std::endl;
		exit(error_code);
	}

	double result = 0.0;
	for (int i = 0; i < batch_size; i++) {

		if (batched_targets[i] >= neurons) {
			std::cerr << "Error: invalid batched_tagets input" << std::endl;
			exit(EXIT_FAILURE);
		}
		host_forward_output[i * neurons + batched_targets[i]] = (host_forward_output[i * neurons + batched_targets[i]] > 1e-7) ? host_forward_output[i * neurons + batched_targets[i]] : 1e-7;
		host_forward_output[i * neurons + batched_targets[i]] = (host_forward_output[i * neurons + batched_targets[i]] < 1 - 1e-7) ? host_forward_output[i * neurons + batched_targets[i]] : 1 - 1e-7;
		result += -std::log(host_forward_output[i * neurons + batched_targets[i]]) / (double)(batch_size);
	}
	free(host_forward_output);
	return result;
}
double pooling_layer::loss(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) const {

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	double result = 0.0;
	double* host_forward_output = (double*)malloc(batch_size * neurons * sizeof(double));
	if (host_forward_output == nullptr) {
		std::cerr << "Error: Failed to allocate memory in pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	cudaError error_code = cudaMemcpy(host_forward_output, forward_output, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToHost);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpyDeviceToHost failed in pooling_layer" << std::endl;
		exit(error_code);
	}

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
					result += ((host_forward_output[i * channels * output_size * output_size + j * output_size * output_size + y * output_size + x] - batched_targets[i][j][y][x]) * (host_forward_output[i * channels * output_size * output_size + j * output_size * output_size + y * output_size + x] - batched_targets[i][j][y][x])) / (double)(batch_size * neurons);
				}
			}
		}
	}

	free(host_forward_output);
	return result;
}

void pooling_layer::init_back_propigation(const std::vector<unsigned int>& batched_targets) {

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of invalid input shape for pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	for (int i = 0; i < batch_size; i++) {
		if (batched_targets[i] >= neurons) {
			std::cerr << "Error: Invalid inputs in batched_targets" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	unsigned int* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * sizeof(unsigned int));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in pooling_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaMemcpy(input_arr, batched_targets.data(), batch_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpyHostToDevice failed in pooling_layer" << std::endl;
		exit(error_code);
	}

	init_back_propigation(input_arr, batch_size);
	cudaFree(input_arr);
}

void pooling_layer::init_back_propigation(const std::vector<std::vector<double>>& batched_targets) {
	
	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in pooling_layer" << std::endl;
		exit(error_code);
	}

	for (size_t i = 0; i < batch_size; i++) {

		if (batched_targets[i].size() != neurons) {
			std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
			exit(EXIT_FAILURE);
		}

		error_code = cudaMemcpy(input_arr + i * neurons, batched_targets[i].data(), neurons * sizeof(double), cudaMemcpyHostToDevice);
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMemcpyHostToDevice failed in pooling_layer" << std::endl;
			exit(error_code);
		}
	}
	
	init_back_propigation(input_arr, neurons, batch_size);
	cudaFree(input_arr);
}
void pooling_layer::init_back_propigation(const std::vector<std::vector<std::vector<std::vector<double>>>>& batched_targets) {

	if (batched_targets.size() != batch_size) {
		std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	double* input_arr = nullptr;
	cudaError error_code = cudaMalloc((void**)&input_arr, batch_size * neurons * sizeof(double));
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMalloc failed in pooling_layer" << std::endl;
		exit(error_code);
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

				error_code = cudaMemcpy(input_arr + i * neurons + j * channels * output_size * output_size + y * output_size, batched_targets[i][j][y].data(), output_size * sizeof(double), cudaMemcpyHostToDevice);;
				if (error_code != cudaError::cudaSuccess) {
					std::cerr << "Error: cudaMemcpyHostToDevice failed in pooling_layer" << std::endl;
					exit(error_code);
				}
			}
		}
	}

	init_back_propigation(input_arr, neurons, batch_size);
	cudaFree(input_arr);
}

void pooling_layer::init_back_propigation(unsigned int* batched_targets, size_t _batch_size) {

	if (batch_size != _batch_size) {
		std::cerr << "Error: Invalid input size for pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	if (layer_activation_function != activation_functions::Softmax) {
		std::cerr << "Error: Invalid activation_function for Init_Cross_Catagorial_Loss in pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (backward_input == nullptr) {
		error_code = cudaMalloc((void**)&backward_input, batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in pooling_layer" << std::endl;
			exit(error_code);
		}
	}

	dim3 blocks(neurons / 16 + 1, batch_size / 16 + 1);
	dim3 threads(16, 16);

	Cuda_Pooling_Layer_Init_Cross_Catigorial_Loss_Back_Propigation<<<blocks, threads>>>(batched_targets, forward_output, backward_input, batch_size, neurons);

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch init_backpropigation kernal in pooling_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSunchronize failed" << std::endl;
		exit(error_code);
	}
}

void pooling_layer::init_back_propigation(double* batched_targets, size_t _input_size, size_t _batch_size) {

	if (batch_size != _batch_size || _input_size != neurons) {
		std::cerr << "Error: Batched_targets of invalid input shape for pooling layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (backward_input == nullptr) {
		error_code = cudaMalloc((void**)&backward_input, batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in pooling_layer" << std::endl;
			exit(error_code);
		}
	}

	dim3 blocks(batch_size / 16 + 1, neurons / 16 + 1);
	dim3 threads(16, 16);
	
	Cuda_Pooling_Layer_Init_Backpropigation<<<blocks, threads>>>(forward_output, batched_targets, backward_input, batch_size, neurons);
	
	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch init_backpropigation kernal in pooling layer" << std::endl;
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
		std::cerr << "Error: Failed to launch backward activation function kernals in pooling_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
}

void pooling_layer::backward(layer* prev_layer) {
	
	if (prev_layer->batch_size != batch_size || prev_layer->neurons != inputs) {
		std::cerr << "Error: Prev_layer of invalid input shape or batch size to connected to pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}
	
	if (backward_input == nullptr) {
		std::cerr << "Error: Pooling_layer not initialized for backward pass" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (prev_layer->backward_input == nullptr) {
		error_code = cudaMalloc((void**)&prev_layer->backward_input, batch_size * inputs * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in pooling_layer" << std::endl;
			exit(error_code);
		}
		
		error_code = cudaMemset(prev_layer->backward_input, 0, batch_size * inputs * sizeof(double));
		if(error_code != cudaError::cudaSuccess){
			std::cerr << "Error: cudaMemset failed in pooling_layer" << std::endl;
			exit(error_code);
		}
	}

	dim3 blocks(batch_size/6 + 1, channels/6 + 1, (input_size * input_size)/6 + 1);
	dim3 threads(6, 6, 6);
	
	if (pooling_layer_type == pooling_type::Max) {
		Cuda_Max_Pooling_Layer_Partial_Derivitive_of_Loss<<<blocks, threads>>>(prev_layer->forward_output, backward_input, prev_layer->backward_input, batch_size, channels, output_size, input_size, kernal_size, stride);
	}
	else if (pooling_layer_type == pooling_type::Average) {
		Cuda_Average_Pooling_Layer_Partial_Derivitive_of_Loss<<<blocks, threads>>>(prev_layer->forward_output, backward_input, prev_layer->backward_input, batch_size, channels, output_size, input_size, kernal_size, stride);
	}

	error_code = cudaGetLastError();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: Failed to launch Pooling layer pairtial derivite of loss kernal" << std::endl;
		exit(error_code);
	}

	dim3 blocks_2d(inputs / 16 + 1, batch_size / 16 + 1);
	dim3 threads_2d(16, 16);

	if (prev_layer->layer_activation_function == activation_functions::Sigmoid) {
		Cuda_Sigmoid_Activation_Backward_Pass<<<blocks_2d, threads_2d>>>(prev_layer->backward_input, prev_layer->forward_output, batch_size, inputs);
	}
	else if (prev_layer->layer_activation_function == activation_functions::Rectified_Linear) {
		Cuda_Rectified_Linear_Activation_Backward_Pass<<<blocks_2d, threads_2d>>>(prev_layer->backward_input, prev_layer->forward_output, batch_size, inputs);
	}
	else if (prev_layer->layer_activation_function == activation_functions::Softmax) {
		Cuda_Softmax_Activation_Bakcward_Pass<<<blocks_2d, threads_2d>>>(prev_layer->forward_output, prev_layer->backward_input, batch_size, inputs);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed" << std::endl;
		exit(error_code);
	}
}

void pooling_layer::backward(layer* prev_layer, layer* residual_layer) {

	if (prev_layer->batch_size != batch_size || prev_layer->neurons != inputs || residual_layer->batch_size != batch_size || residual_layer->neurons != neurons) {
		std::cerr << "Error: Prev_layer or residual_layer of invalid input shape or batch size to connected to pooling_layer" << std::endl;
		exit(EXIT_FAILURE);
	}

	cudaError error_code;

	if (residual_layer->backward_input == nullptr) {
		error_code = cudaMalloc((void**)&residual_layer->backward_input, batch_size * neurons * sizeof(double));
		if (error_code != cudaError::cudaSuccess) {
			std::cerr << "Error: cudaMalloc failed in pooling_layer" << std::endl;
			exit(error_code);
		}
	}

	error_code = cudaMemcpy(residual_layer->backward_input, backward_input, batch_size * neurons * sizeof(double), cudaMemcpyDeviceToDevice);
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaMemcpyDeviceToDevice failed in pooling_layer" << std::endl;
		exit(error_code);
	}

	error_code = cudaDeviceSynchronize();
	if (error_code != cudaError::cudaSuccess) {
		std::cerr << "Error: cudaDeviceSynchronize failed in pooling_layer" << std::endl;
		exit(error_code);
	}

	backward(prev_layer);
}