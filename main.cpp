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
//All checks for residual connections need to double checked.
//Pooling layer and convolutional layer residual forward pass needs to be re-done.

//Dense_layer residual forward pass is done, but not tested. Backward pass is almost done, but structure is now understood.

//The major optimization of moving everying off host. Will do at the very end. 

int main(void) {
	
	std::vector<std::vector<double>> batched_inputs = { {1,2,3,4}, {1,2,3,4} };
	std::vector<std::vector<double>> batched_targets = { {10,20,30,40}, {10,20,30,40} };
	dense_layer layer_1(4, 4, activation_functions::Linear);
	dense_layer layer_2(4, 4, activation_functions::Sigmoid);
	dense_layer layer_3(4, 4, activation_functions::Rectified_Linear);

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1, &layer_1);
	layer_3.forward(&layer_2);
	
	double loss = layer_3.loss(batched_targets);
	double esp = 1e-3;
	layer_1.weights[15] += esp;

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1, &layer_1);
	layer_3.forward(&layer_2);
	
	double loss_ph = layer_3.loss(batched_targets);
	double dl_dp = (loss_ph - loss) / esp;

	std::cout << dl_dp << "\n";

	/*
	{0 = > -0.28058, 1 = > -0.56116, 2 = > -0.841741, 3 = > -1.12232}
	{4 = > -0.165454, 5 = > -0.330908, 6 = > -0.496361, 7 = > -0.661815}
	{8 = > 0.0678055, 9 = > 0.135611, 10 = > 0.203417, 11 = > 0.271222}
	{12 = > 0.0455907, 13 = > 0.0911813, 14 = > 0.136772, 15 = > 0.182363}
	*/

	return 0;
}