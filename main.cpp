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
//Rename Loss functions / Init_Loss functions to proper names. Cross_Entropy_Mean_Loss ect. 
//Optimize kernals (specifically the inputs).
//Need to test second backward pass in dense_layer. 99% sure error has been fixed but needs to be checked.
//Need to add a check that backward_input is intialized in each layer in backward pass.

int main(void) {

	std::vector<std::vector<std::vector<std::vector<double>>>> batched_inputs = { {{{5, 5, 5,5}, {5,5,5,5}, {5, 5, 5, 5}, {1, 1, 1, 1}}, {{0.6, 0.6, 0.4,0.5}, {5,5,5,5}, {1, 1, 0.5, 0.5}, {1, 1, 1, 1}}}, {{{0.5, 0.5, 0.5,0.5}, {1.5,1.5,1.5,1.5}, {1.5, 1.5, 1.5, 5}, {0.1, 0.1, 10, 1}}, {{0.6, 0.6, 0.4,0.5}, {5,5,5,5}, {1, 1, 0.5, 0.5}, {1, 1, 1, 1}}} };
	
	convolutional_layer layer_1(4, 2, 4, 2, 1, 1, activation_functions::Sigmoid);
	convolutional_layer layer_2(5, 4, 4, 3, 1, 1, activation_functions::Sigmoid);
	convolutional_layer layer_3(5, 4, 4, 3, 2, 0, activation_functions::Sigmoid);
	dense_layer layer_4(16, 2, activation_functions::Sigmoid);

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);
	layer_3.forward(&layer_2);
	layer_4.forward(&layer_3);

	Print_Forward_Output(layer_3.forward_output, 2, 16);

	return 0;
}