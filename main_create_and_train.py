import json
import numpy as np
from layer import Layer
from neural_network import NeuralNet
from training import Training


# Make the neural net
neural_net = NeuralNet()

# Input layer "previousLayer size" parameter should always be its own size
input_layer = Layer(previousLayer_size=3, layer_size=3, layer_type='input')
hidden_layer1 = Layer(previousLayer_size=3, layer_size=10, layer_type='hidden')
hidden_layer2 = Layer(previousLayer_size=10, layer_size=15, layer_type='hidden')
hidden_layer3 = Layer(previousLayer_size=15, layer_size=10, layer_type='hidden')
output_layer = Layer(previousLayer_size=10, layer_size=2, layer_type='output')
neural_net.add_layer(input_layer)
neural_net.add_layer(hidden_layer1)
neural_net.add_layer(hidden_layer2)
neural_net.add_layer(hidden_layer3)
neural_net.add_layer(output_layer)

# Train the neural net
with open("color_data.json", "r") as file:
    data = json.load(file)

input_data = np.array(data["RBG_Values"])
target_data = np.array(data["Is_Red"])


training = Training(neural_net, learning_rate=0.005, clip_value=5)

num_epochs = 200
training.train(input_data, target_data, epochs=num_epochs)



# Save the neural net #

# Everything is stored in a json file
neural_net.save("model_params.json")