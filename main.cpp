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

//TODO: 
//Fix cuda Kernal configurations (just need types changed to U32 now). 
//Implement Softmax + do math
//Rename Loss functions / Init_Loss functions to proper names. Cross_Entropy_Mean_Loss ect. 
//Optimize kernals (specifically the inputs).
//Add training optimizers.

//The major optimization of moving everying off host. Will do at the very end. 

//4d Convolutions?

//Not sure if I want to add a model class or not, will decide later. 

int main(void) {
	
	std::vector<std::vector<double>> batched_inputs = { {1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4}, {4,3,2,1,4,3,2,1,4,3,2,1,4,3,2,1} };
	std::vector<std::vector<double>> batched_targets = { {0.5,0.5,0.1,0.4}, {0.1, 0.7, 0.2, 0} };

	convolutional_layer layer_1(4, 1, 1, 2, 1, 1, activation_functions::Linear);
	convolutional_layer layer_2(5, 1, 1, 3, 1, 1, activation_functions::Sigmoid);
	convolutional_layer layer_3(5, 1, 3, 3, 1, 0, activation_functions::Sigmoid);
	convolutional_layer layer_4(3, 3, 1, 2, 1, 0, activation_functions::Linear);
	dense_layer layer_5(4, 4, activation_functions::Softmax);

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1, &layer_1);
	layer_3.forward(&layer_2);
	layer_4.forward(&layer_3);
	layer_5.forward(&layer_4);

	Print_Cuda_Forward_Output(layer_5.forward_output, 2, 4);

	return 0;
}