import json
import numpy as np
from layer import Layer
from neural_network import NeuralNet
from training import Training


neural_net = NeuralNet()
neural_net.load("model_params.json")

# Test the neural net 
input_data = np.array([0, 0, 0])

output_data = neural_net.forward_propagation(input_data)

percent = (output_data[1] - output_data[0])

# If the distance is negative, then the color is red, otherwise it is not red
# This is specified in the definition of red python file
is_red = "is not red"
if percent < 0:
    is_red = "is red"
    percent *= -1

print("Verdict: the (r, g, b) color triple", is_red, "with {:.16f}% confidence.".format(percent * 100))

