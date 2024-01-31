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

	layer_1.forward(batched_inputs);
	layer_2.forward(&layer_1);

	Print_Forward_Output(layer_1.forward_output, 3, 3);
	Print_Forward_Output(layer_2.forward_output, 3, 1);

	layer_2.init_back_propigation(batched_targets);
	layer_2.backward(&layer_1);
	
	

	return 0;
}