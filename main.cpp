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
	std::vector<std::vector<double>> batched_targets = { {0.5,0.5,0.1,0.4, 0.3}, {0.9, 0.4, 0.7, 0.2, 1} };

	dense_layer layer_1(16, 5, activation_functions::Linear);
	dense_layer layer_2(5, 5, activation_functions::Sigmoid);
	dense_layer layer_3(5, 5, activation_functions::Sigmoid);

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1, &layer_1);
	layer_3.forward(&layer_2);

	std::cout << layer_3.loss(batched_targets) << '\n';

	for (int i = 0; i < 10000; i++) {

		layer_3.init_back_propigation(batched_targets);
		layer_3.backward(&layer_2);
		layer_2.backward(&layer_1, &layer_1);
		layer_1.backward(batched_inputs);

		layer_1.update_paramters(1e-3);
		layer_2.update_paramters(1e-3);
		layer_3.update_paramters(1e-3);

		layer_1.forward(batched_inputs);
		layer_2.forward(&layer_1, &layer_1);
		layer_3.forward(&layer_2);
		std::cout << layer_3.loss(batched_targets) << '\n';
	}


	return 0;
}