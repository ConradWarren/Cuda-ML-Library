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
	std::vector<std::vector<double>> batched_targets = { {1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1},{1, 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1} };


	convolutional_layer layer_1(4, 2, 4, 2, 1, 1, activation_functions::Linear);
	convolutional_layer layer_2(5, 4, 4, 3, 1, 1, activation_functions::Linear);
	convolutional_layer layer_3(5, 4, 4, 3, 2, 0, activation_functions::Linear);
	pooling_layer layer_4(2, 4, 2, 1, pooling_type::Max);
	//dense_layer layer_5(4, 2, activation_functions::Linear);

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);
	layer_3.forward(&layer_2);
	//layer_4.forward(&layer_3);

	std::cout << "finished forward pass" << std::endl;
	//layer_5.forward(&layer_4);
	
	
	double loss = layer_3.loss(batched_targets);
	std::cout << "loss = " << loss << '\n';
	double esp = 1e-3;
	layer_1.weights[5] += esp;

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);
	layer_3.forward(&layer_2);
	//layer_4.forward(&layer_3);
	//layer_5.forward(&layer_4);
	double loss_ph = layer_3.loss(batched_targets);
	std::cout << "loss_ph = " << loss_ph << '\n';
	double dl_dp = (loss_ph - loss) / esp;
	std::cout << dl_dp << "\n";
	
	/*
	layer_3.init_back_propigation(batched_targets);
	std::cout << "Finished init_back_prop" << std::endl;
	//layer_5.backward(&layer_4);
	layer_3.backward(&layer_2);
	std::cout << "Finished layer_3 backward pass" << std::endl;
	layer_2.backward(&layer_1);
	std::cout << "Finished layer_2 backward pass" << std::endl;
	layer_1.backward(batched_inputs);
	std::cout << "Finished layer_1 backward pass" << std::endl;
	
	Print_Forward_Output(layer_1.d_weights, 8, 2);
	*/
	/*
	{0 = > 3275.15, 1 = > 4059.87} 0 = > 3275.93, 1 => 4172.79, 4171.84, 4171.75, 4171.74 ??
	{2 = > 511.155, 3 = > 1810.47} 2 => 511.565, 3 => 1831.7
	{4 = > -1202.56, 5 = > -75.1781} 4 => -1202.21, 5 => -466.445
	{6 = > 3147.18, 7 = > 2932.62} 6 => 
	{8 = > 769.531, 9 = > 661.845}
	{10 = > -855.111, 11 = > 154.524}
	{12 = > -1365.54, 13 = > -453.202}
	{14 = > 1543.22, 15 = > 1094.88}
	*/


	return 0;
}