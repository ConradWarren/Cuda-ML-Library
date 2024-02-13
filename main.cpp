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
//Possibly worth it to check if batched_inputs is equal to nullptr or mayber the forward_output of prev_layer.

//The major optimization of moving everying off host. Will do at the very end. 

int main(void) {
	
	std::vector<std::vector<double>> batched_inputs = { {1,2,3,4}, {1,2,3,4} };
	std::vector<std::vector<double>> batched_targets = { {10,20,30,40}, {10,20,30,40} };
	dense_layer layer_1(4, 4, activation_functions::Sigmoid);
	dense_layer layer_2(4, 4, activation_functions::Linear);
	dense_layer layer_3(4, 4, activation_functions::Linear);

	
	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);
	layer_3.forward(&layer_1, &layer_2);
	
	/*
	layer_3.init_back_propigation(batched_targets);

	
	layer_3.backward(&layer_1, &layer_2);
	std::cout << "Layer 3 backward pass finished" << std::endl;
	layer_2.backward(&layer_1);
	std::cout << "Layer 2 backward pass finished" << std::endl;
	layer_1.backward(batched_inputs);
	std::cout << "Layer 1 backward pass finished" << std::endl;

	Print_Forward_Output(layer_1.d_weights, 4, 4);
	*/

	
	double loss = layer_3.loss(batched_targets);
	double esp = 1e-6;
	layer_1.weights[0] += esp;

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);
	layer_3.forward(&layer_1, &layer_2);

	double loss_ph = layer_3.loss(batched_targets);
	double dl_dp = (loss_ph - loss) / esp;

	std::cout << dl_dp << '\n';
	

	/*
	{0 = > -3.73744, 1 = > -7.47488, 2 = > -11.2123, 3 = > -14.9498}
	{4 = > 0.00534944, 5 = > 0.0106989, 6 = > 0.0160483, 7 = > 0.0213978}
	{8 = > -10.7433, 9 = > -21.4866, 10 = > -32.2299, 11 = > -42.9732}
	{12 = > 1.32522, 13 = > 2.65044, 14 = > 3.97566, 15 = > 5.30088}
	*/

	//0 => 

	return 0;
}