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
//Rename Loss functions / Init_Loss functions to proper names. Cross_Entropy_Mean_Loss ect. 
//Optimize kernals (specifically the inputs).
//Add training optimizers.
//4d Convolutions?
//Add model class. 


int main(void) {
	
	std::vector<std::vector<double>> batched_inputs = { {1,4,3,2}, {0,0,0,0} };
	std::vector<unsigned int> batched_targets = { 1, 4 };

	dense_layer layer_1(4, 10, activation_functions::Sigmoid);
	dense_layer layer_2(10, 15, activation_functions::Rectified_Linear);
	dense_layer layer_3(15, 10, activation_functions::Linear);
	dense_layer layer_4(10, 5, activation_functions::Softmax);

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);
	layer_3.forward(&layer_2);
	layer_4.forward(&layer_3);

	for (int i = 0; i < 1000; i++) {

		std::cout << i << ": " << layer_4.loss(batched_targets)<<'\n';

		layer_4.init_back_propigation(batched_targets);
		layer_4.backward(&layer_3);
		layer_3.backward(&layer_2);
		layer_2.backward(&layer_1);
		layer_1.backward(batched_inputs);

		layer_4.update_paramters_adaptive_momentum(1e-3, 0.9, 0.999);
		layer_3.update_paramters_adaptive_momentum(1e-3, 0.9, 0.999);
		layer_2.update_paramters_adaptive_momentum(1e-3, 0.9, 0.999);
		layer_1.update_paramters_adaptive_momentum(1e-3, 0.9, 0.999);

		layer_1.forward(batched_inputs);
		layer_2.forward(&layer_1);
		layer_3.forward(&layer_2);
		layer_4.forward(&layer_3);
	}

	return 0;
}