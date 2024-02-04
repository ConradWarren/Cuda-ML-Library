#include <iostream>
#include <vector>

#include "layer.hpp"

void Print_Forward_Output(double* arr, size_t batch_size, size_t neurons) {

	for (int i = 0; i < batch_size; i++) {

		std::cout << "{";
		for (int j = 0; j < neurons; j++) {
			std::cout <<i * neurons + j<<" = > "<< arr[i * neurons + j];
			if (j + 1 < neurons) std::cout << ", ";
		}
		std::cout << "}\n";
	}
	std::cout << "\n";
}

//TODO: Fix cuda Kernal configurations
//Implement Softmax + do math
//Implement auto flattening / de-flattening
//Rename Loss functions / Init_Loss functions to proper names. Cross_Entropy_Mean_Loss ect. 
//Optimize kernals (specifically the inputs).
//Need to test second backward pass in dense_layer. 99% sure error has been fixed but needs to be checked.

int main(void) {

	std::vector<std::vector<std::vector<std::vector<double>>>> batched_inputs = { {{ {1,1,1,1}, {1,1,1,1}, {1,1,1,1}, {1,1,1,1} }, { {1,1,1,1}, {1,1,1,1}, {1,1,1,1}, {1,1,1,1} }}, {{ {2,2,2,2}, {2,2,2,2}, {2,2,2,2}, {2,2,2,2} }, { {2,2,2,2}, {2,2,2,2}, {2,2,2,2}, {2,2,2,2} }} };
	std::vector<std::vector<double>> batched_targets_2d = { {5,5,5,5,5,5,5,5}, {10,10,10,10, 10,10,10,10} };
	std::vector<std::vector<std::vector<std::vector<double>>>> batched_targets_4d = { { {{5, 5}, {5, 5}}, {{5, 5}, {5, 5}} }, {{{10, 10}, {10, 10}}, {{10, 10}, {10, 10}}} };
	//std::vector<std::vector<std::vector<std::vector<double>>>> batched_targets_4d = { {{{1}}}, {{{1}}} };

	std::cout << batched_targets_4d.size() << '\n';
	std::cout << batched_targets_2d.size() << '\n';

	convolutional_layer layer_1(4, 2, 2, 2, 2, 0);
	convolutional_layer layer_2(2, 2, 1, 2, 1, 0);


	layer_1.forward(batched_inputs);
	std::cout << layer_1.loss(batched_targets_2d) << '\n';
	std::cout << layer_1.loss(batched_targets_4d) << '\n';
	layer_1.init_back_propigation(batched_targets_2d);
	layer_1.backward(batched_inputs);

	Print_Forward_Output(layer_1.d_weights, 2, 4);
	Print_Forward_Output(layer_1.d_weights + 8, 2, 4);
	Print_Forward_Output(layer_1.d_bias, 1, 2);


	/*
	{0 = > 3.13872e+66, 1 = > 3.13872e+66, 2 = > 3.13872e+66, 3 = > 3.13872e+66}
	{4 = > 3.13872e+66, 5 = > 3.13872e+66, 6 = > 3.13872e+66, 7 = > 3.13872e+66}

	{0 = > 6.27744e+66, 1 = > 6.27744e+66, 2 = > 6.27744e+66, 3 = > 6.27744e+66}
	{4 = > 6.27744e+66, 5 = > 6.27744e+66, 6 = > 6.27744e+66, 7 = > 6.27744e+66}
	
	*/

	return 0;
}