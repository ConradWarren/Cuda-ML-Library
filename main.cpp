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
//Implement auto flattening / de-flattening
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
	
	/*
	double loss = layer_2.loss(batched_targets_4d);
	double esp = 1e-3;
	layer_1.weights[15] += esp;

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);

	double loss_ph = layer_2.loss(batched_targets_4d);
	double dl_dp = (loss_ph - loss) / esp;
	std::cout << dl_dp << '\n';
	*/
	
	layer_2.init_back_propigation(batched_targets_4d);
	layer_2.backward(&layer_1);
	layer_1.backward(batched_inputs);

	Print_Forward_Output(layer_1.d_weights, 8, 2);
	
	/*
	{0 = > 0, 1 = > 2213, 2 = > 4426, 3 = > 6639, 4 = > 8852, 5 = > 11065, 6 = > 13278, 7 = > 15491}
	{8 = > 0, 9 = > 4405, 10 = > 8810, 11 = > 13215, 12 = > 17620, 13 = > 22025, 14 = > 26430, 15 = > 30835}
	*/
	/*
	{0 = > 0, 1 = > 2213, 2 = > 4426, 3 = > 6639, 4 = > 8852, 5 = > 11065, 6 = > 13278, 7 = > 15491}
	{8 = > 0, 9 = > 4405, 10 = > 8810, 11 = > 13215, 12 = > 17620, 13 = > 22025, 14 = > 26430, 15 = > 30835}
	*/
	//3, 5, 6, 7, 
	/*
		0 - 7 => 66138.1, 
		8 - 15 => 242507
	*/

	/* LAYER_1 Weights. 
	{0 = > 152256, 1 = > 152256}
	{2 = > 152256, 3 = > 152256}
	{4 = > 152256, 5 = > 152256}
	{6 = > 152256, 7 = > 152256}
	{8 = > 558104, 9 = > 558104}
	{10 = > 558104, 11 = > 558104}
	{12 = > 558104, 13 = > 558104}
	{14 = > 558104, 15 = > 558104}
	*/

	/* LAYER_2 Wegihts
	{0 = > 308644, 1 = > 308644}
	{2 = > 308644, 3 = > 308644}
	{4 = > 1.02073e+06, 5 = > 1.02073e+06}
	{6 = > 1.02073e+06, 7 = > 1.02073e+06}
	*/
	return 0;
}