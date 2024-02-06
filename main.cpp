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

int main(void) {

	std::vector<std::vector<std::vector<std::vector<double>>>> batched_inputs = { {{ {1,1,1,1}, {1,1,1,1}, {1,1,1,1}, {1,1,1,1} }, { {1,1,1,1}, {1,1,1,1}, {1,1,1,1}, {1,1,1,1} }}, {{ {2,2,2,2}, {2,2,2,2}, {2,2,2,2}, {2,2,2,2} }, { {2,2,2,2}, {2,2,2,2}, {2,2,2,2}, {2,2,2,2} }} };
	std::vector<std::vector<double>> batched_targets_2d = { {5,5,5,5,5,5,5,5}, {10,10,10,10, 10,10,10,10} };
	//std::vector<std::vector<std::vector<std::vector<double>>>> batched_targets_4d = { { {{5, 5}, {5, 5}}, {{5, 5}, {5, 5}} }, {{{10, 10}, {10, 10}}, {{10, 10}, {10, 10}}} };
	std::vector<std::vector<std::vector<std::vector<double>>>> batched_targets_4d = { {{{1}}}, {{{1}}} };

	

	convolutional_layer layer_1(4, 2, 2, 2, 2, 0);
	convolutional_layer layer_2(2, 2, 1, 2, 1, 0);




	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);
	layer_2.init_back_propigation(batched_targets_4d);
	layer_2.backward(&layer_1);
	layer_1.backward(batched_inputs);

	Print_Forward_Output(layer_1.d_bias, 1, 2);
	Print_Forward_Output(layer_2.d_bias, 1, 1);

	//LAYER_1
	//39708, 145596

	//Layer_2 
	//6618

	return 0;
}