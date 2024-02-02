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

int main(void) {

	std::vector<std::vector<std::vector<std::vector<double>>>> batched_inputs = { {{ {1,1,1,1}, {1,1,1,1}, {1,1,1,1}, {1,1,1,1} }, { {1,1,1,1}, {1,1,1,1}, {1,1,1,1}, {1,1,1,1} }}, {{ {2,2,2,2}, {2,2,2,2}, {2,2,2,2}, {2,2,2,2} }, { {2,2,2,2}, {2,2,2,2}, {2,2,2,2}, {2,2,2,2} }} };
	std::vector<std::vector<double>> batched_targets_2d = { {5,5,5,5,5,5,5,5}, {10,10,10,10, 10,10,10,10} };
	std::vector<std::vector<std::vector<std::vector<double>>>> batched_targets_4d = { {{{5, 5}, {5,5}}, {{5, 5}, {5, 5}}}, {{{10, 10}, {10,10}}, {{10, 10}, {10, 10}}} };


	convolutional_layer layer_1(4, 2, 2, 2, 2, 0);
	
	layer_1.forward(batched_inputs);
	layer_1.init_back_propigation(batched_targets_4d);
	
	Print_Forward_Output(layer_1.forward_output, 2, 2);
	Print_Forward_Output(layer_1.backward_input, 2, 2);
	Print_Forward_Output(layer_1.forward_output + 4, 2, 2);
	Print_Forward_Output(layer_1.backward_input + 4, 2, 2);
	Print_Forward_Output(layer_1.forward_output + 8, 2, 2);
	Print_Forward_Output(layer_1.backward_input + 8, 2, 2);
	Print_Forward_Output(layer_1.forward_output + 12, 2, 2);
	Print_Forward_Output(layer_1.backward_input + 12, 2, 2);

	return 0;
}