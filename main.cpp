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
//Need to add a check that backward_input/forward_output is intialized in each layer in backward pass.
//Add residual_layers. (forward pass for dense_layer done)

int main(void) {

	std::vector<std::vector<std::vector<std::vector<double>>>> batched_inputs = { {{{5, 5, 5,5}, {5,5,5,5}, {5, 5, 5, 5}, {1, 1, 1, 1}}, {{0.6, 0.6, 0.4,0.5}, {5,5,5,5}, {1, 1, 0.5, 0.5}, {1, 1, 1, 1}}}, {{{0.5, 0.5, 0.5,0.5}, {1.5,1.5,1.5,1.5}, {1.5, 1.5, 1.5, 5}, {0.1, 0.1, 10, 1}}, {{0.6, 0.6, 0.4,0.5}, {5,5,5,5}, {1, 1, 0.5, 0.5}, {1, 1, 1, 1}}} };
	std::vector<std::vector<double>> batched_targets = { {1,1},{2,2} };

	convolutional_layer layer_1(4, 2, 4, 2, 1, 1, activation_functions::Linear);
	convolutional_layer layer_2(5, 4, 4, 3, 1, 1, activation_functions::Linear);
	convolutional_layer layer_3(5, 4, 4, 3, 2, 0, activation_functions::Linear);
	pooling_layer layer_4(2, 4, 2, 1, pooling_type::Average);
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
	
	//Print_Forward_Output(layer_1.d_weights, 8, 2);
	
	/*
	{0 = > 266.151, 1 = > -161.152} 0 => 266.168, 1 => -161.147
	{2 = > -1785.43, 3 = > -1669.29} 2 => -1785.41, 3 => -1669.27
	{4 = > -2731.55, 5 = > -2792.2} 4 => -2731.48, 5 => -2792.13
	{6 = > 1939.53, 7 = > 1796.93} 6 => 1939.56, 7 => 1796.95
	{8 = > 1654.74, 9 = > 1646.17} 8 => 1654.75, 9 => 146.401
	{10 = > 146.399, 11 = > 733.704} 10 => 146.401, 11 => 733.708
	{12 = > -1023.31, 13 = > -240.998} 12 => -1023.3, 13 => -240.996
	{14 = > 2030.46, 15 = > 2021.17}  14 => 2030.49,  15 => 2021.2
	
	*/

	return 0;
}