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

//TODO: Test backward pass, still need to implement backward(prev_layer) method. 

int main(void) {

	dense_layer layer_1(3, 3);
	dense_layer layer_2(3, 1);

	std::vector<std::vector<double>> batched_inputs = { {1,2,3},{22,2,2},{3,2,1} };
	std::vector<std::vector<double>> batched_targets = { {1}, {2}, {3} };

	for (int i = 0; i < 10000; i++) {
		layer_1.forward(batched_inputs);
		layer_2.forward(&layer_1);

		std::cout <<i<<" : "<< layer_2.loss(batched_targets) << '\n';

		layer_2.init_back_propigation(batched_targets);
		layer_2.backward(&layer_1);
		layer_1.backward(batched_inputs);

		layer_2.update_paramters(1e-5);
		layer_1.update_paramters(1e-5);
	}

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);



	/*
	* LAYER_1 d_WEIGHTS
	{0 = > 0, 1 = > 0, 2 = > 0}
	{3 = > 6350.67, 4 = > 852, 5 = > 860}
	{6 = > 12701.3, 7 = > 1704, 8 = > 1720}
	//0 => 0, 1 => 0, 2 => 0
	//3 => 6314.83, 4 => 844.004, 5 => 853.338
	//6 => 12630, 7 => 1688.02, 8 =>  1706


	LAYER_2 d_WEIGHTS
	{0 = > 2572, 1 = > 27186, 2 = > 51800}
	//0 => 2550.71, 1 => 270011.5, 2 => 51476.9
	
	*/
	return 0;
}