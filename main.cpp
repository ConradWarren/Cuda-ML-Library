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
	std::vector<std::vector<double>> batched_targets = { {1,1},{2,2} };

	convolutional_layer layer_1(4, 2, 4, 2, 1, 1, activation_functions::Linear);
	convolutional_layer layer_2(5, 4, 4, 3, 1, 1, activation_functions::Linear);
	convolutional_layer layer_3(5, 4, 4, 3, 2, 0, activation_functions::Linear);
	pooling_layer layer_4(2, 4, 2, 1, pooling_type::Max);
	dense_layer layer_5(4, 2, activation_functions::Linear);

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);
	layer_3.forward(&layer_2);
	layer_4.forward(&layer_3);
	layer_5.forward(&layer_4);
	std::cout << "finished forward pass" << std::endl;
	/*
	double loss = layer_5.loss(batched_targets);
	std::cout << "loss = " << loss << '\n';
	double esp = 1e-4;
	layer_1.weights[15] += esp;

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);
	layer_3.forward(&layer_2);
	layer_4.forward(&layer_3);
	layer_5.forward(&layer_4);
	double loss_ph = layer_5.loss(batched_targets);
	std::cout << "loss_ph = " << loss_ph << '\n';
	double dl_dp = (loss_ph - loss) / esp;
	std::cout << dl_dp << "\n";
	*/
	
	layer_5.init_back_propigation(batched_targets);
	std::cout << "Finished init_back_prop" << std::endl;
	layer_5.backward(&layer_4);
	std::cout << "Finished layer_5 backward pass" << std::endl;
	layer_4.backward(&layer_3);
	std::cout << "Finished layer_4 backward pass" << std::endl;
	layer_3.backward(&layer_2);
	std::cout << "Finished layer_3 backward pass" << std::endl;
	layer_2.backward(&layer_1);
	std::cout << "Finished layer_2 backward pass" << std::endl;
	layer_1.backward(batched_inputs);
	std::cout << "Finished layer_1 backward pass" << std::endl;
	
	Print_Forward_Output(layer_1.d_weights, 8, 2);
	

	/*
	{0 = > 2573.53, 1 = > 52.225} 0 => 2573.62, 1 => 52.2413
	{2 = > -1612.34, 3 = > -2226.04} 2 => -1612.31, 3 => -2226.01
	{4 = > -4329.49, 5 = > -4061.62} 4 => -4329.37, 5 => -4061.54
	{6 = > 5163.37, 7 = > 2836.03}  6 => 5163.49, 7 => 2836.06
	{8 = > 809.043, 9 = > 2472.08}  8 => 809.049, 9 => 2472.09
	{10 = > -1800.76, 11 = > 1889.7} 10 => -1800.75, 11 => 1889.71
	{12 = > -2504.28, 13 = > -127.897} 12 => -2504.25, 13 => -127.874
	{14 = > 2613.98, 15 = > 3012.55} 14 => 2614.01, 15 => 3012.59
	*/
	
	return 0;
}