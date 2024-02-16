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
//Need to implement residual backward pass for convolutional and pooling layers. 
//Need to test residual connections for convolutional layers and pooling layers. 

//The major optimization of moving everying off host. Will do at the very end. 

int main(void) {
	
	std::vector<std::vector<double>> batched_inputs = { {1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4}, {4,3,2,1,4,3,2,1,4,3,2,1,4,3,2,1} };
	std::vector<std::vector<double>> batched_targets = { {0.5,0.5,0.1,0.4, 0.3}, {0.9, 0.4, 0.7, 0.2, 1} };

	convolutional_layer layer_1(4, 1, 3, 2, 1, 1, activation_functions::Sigmoid);
	convolutional_layer layer_2(5, 3, 3, 3, 1, 1, activation_functions::Sigmoid);
	pooling_layer layer_3(5, 3, 3, 1, pooling_type::Average);
	dense_layer layer_4(27, 5, activation_functions::Linear);
	
	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1, &layer_1);
	layer_3.forward(&layer_2);
	layer_4.forward(&layer_3);

	/*
	layer_4.init_back_propigation(batched_targets);
	layer_4.backward(&layer_3);
	layer_3.backward(&layer_2);
	layer_2.backward(&layer_1, &layer_1);
	layer_1.backward(batched_inputs);
	Print_Forward_Output(layer_1.d_weights, 6, 2);
	
	return 0;
	*/

	/*
	{0 = > 0.591946, 1 = > 1.12686}
	{2 = > 0.626061, 3 = > 1.25453}
	{4 = > 0.11119, 5 = > 0.0530353}
	{6 = > 0.0285237, 7 = > 0.0421917}
	{8 = > -0.0292862, 9 = > 0.215525}
	{10 = > 0.181844, 11 = > 0.629477}
	*/

	double loss = layer_4.loss(batched_targets);
	double esp = 1e-3;
	layer_1.weights[11] += esp;
	
	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1, &layer_1);
	layer_3.forward(&layer_2);
	layer_4.forward(&layer_3);
	
	double loss_ph = layer_4.loss(batched_targets);
	double dl_dp = (loss_ph - loss) / esp;
	std::cout << dl_dp << '\n';

	return 0;
}