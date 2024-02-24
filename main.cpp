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
	
	std::vector<std::vector<double>> batched_inputs = { {1,4,3,2}, {3,2,1,0} };
	std::vector<std::vector<double>> batched_targets = { {0.1, 0.3, 0.5, 0.1}, {0.1, 0.5, 0.2, 0.2} };

	dense_layer layer_1(4, 4, activation_functions::Softmax);
	dense_layer layer_2(4, 4, activation_functions::Softmax);
	dense_layer layer_3(4, 4, activation_functions::Softmax);
	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);
	layer_3.forward(&layer_2, &layer_1);
	
	/*
	layer_3.init_back_propigation(batched_targets);
	layer_3.backward(&layer_2, &layer_1);
	layer_2.backward(&layer_1);
	layer_1.backward(batched_inputs);
	Print_Cuda_Forward_Output(layer_1.d_weights, 4, 4);
	
	return 0;
	*/
	double loss = layer_3.loss(batched_targets);
	double esp = 1e-3;
	Modify_Cuda_Weight(layer_1.weights, 3, esp);

	layer_1.forward(batched_inputs);
	layer_3.forward(&layer_2, &layer_1);

	double loss_ph = layer_3.loss(batched_targets);

	double dl_dp = (loss_ph - loss) / esp;
	std::cout << dl_dp << '\n';
	/*
	{0 = > 0.00281724, 1 = > 0.00171696, 2 = > 0.000810122, 3 = > -9.67191e-05}
	{4 = > -0.00815965, 5 = > -0.00519317, 6 = > -0.00252261, 7 = > 0.000147958}
	{8 = > 0.00560841, 9 = > 0.00372606, 10 = > 0.00185917, 11 = > -7.72882e-06}
	{12 = > -0.000266004, 13 = > -0.000249853, 14 = > -0.000146682, 15 = > -4.35101e-05}
	*/
	return 0;
}