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
	
	std::vector<std::vector<double>> batched_inputs = { {1,4,3,2}, {0,0,0,0} };
	std::vector<unsigned int> batched_targets = { 1, 10 };

	convolutional_layer layer_1(2, 1, 3, 2, 1, 2, activation_functions::Linear);
	convolutional_layer layer_2(5, 3, 3, 2, 1, 1, activation_functions::Softmax);
	convolutional_layer layer_3(6, 3, 3, 3, 3, 0, activation_functions::Softmax);
	
	
	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);
	layer_3.forward(&layer_2);
	/*
	layer_3.init_back_propigation(batched_targets);
	layer_3.backward(&layer_2);
	layer_2.backward(&layer_1);
	layer_1.backward(batched_inputs);
	
	Print_Cuda_Forward_Output(layer_3.forward_output, 12, 2);
	Print_Cuda_Forward_Output(layer_3.backward_input, 12, 2);

	Print_Cuda_Forward_Output(layer_1.d_weights, 6, 2);
	return 0;
	*/
	
	double loss = layer_3.loss(batched_targets);
	double esp = 1e-3;
	Modify_Cuda_Weight(layer_1.weights, 11, esp);
	
	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);
	layer_3.forward(&layer_2);
	
	
	double loss_ph = layer_3.loss(batched_targets);
	double dl_dp = (loss_ph - loss) / esp;
	std::cout << dl_dp << '\n';

	/*
	{0 = > -0.000137681, 1 = > -0.00798057}
	{2 = > -0.0115917, 3 = > -0.0010312}
	{4 = > -0.00221095, 5 = > -0.00126609}
	{6 = > -0.00178631, 7 = > 0.00303579}
	{8 = > 0.00279488, 9 = > -0.00176391}
	{10 = > 0.0030253, 11 = > -0.00293021}
	*/

	return 0;
}