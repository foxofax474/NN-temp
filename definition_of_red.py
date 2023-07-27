import json
import math
import random

# Ok I'm gonna do something really dumb lmao since I have no ideas other than the handwritten digits thing
# 
# The color red basically is defined as follows
# Create an RGB color triple (r, g, b) and graph it on a 3D coordinate system with each axis being one of r, g, or b
# The color red is defined as any point that is within 127 (ie 255/2 rounded down) units (inclusive) of the point (255, 0, 0) in this 3D coordinate system
# Basically its like part of a sphere with radius 127 centered at (255, 0, 0), and anything in this sphere is considered red
def is_color_red(r, g, b):

    # Define the coordinates for the "definition" of the red point (255, 0, 0)
    red_point = (255, 0, 0)

    # Calculate the distance between the given color and the red point via 3D distance formula
    distance = math.sqrt((r - red_point[0])**2 + (g - red_point[1])**2 + (b - red_point[2])**2)


    if distance <= 127:
        return True
    else:
        return False

def generate_output(isRed):
    # (x, y) where x is the output for red and y is the output for not red
    # If the color is red, we want the output to be (1, 0)
    # If the color is not red, we want the output to be (0, 1)

    if isRed:
        return (1, 0)
    else:
        return (0, 1)
    


# Generate training or testing data


def generate_random_rgb():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

dataLength = 500

# Create a set to store unique RGB triples (we do not want duplicates)
unique_rgb_set = set()

while len(unique_rgb_set) < dataLength:
    unique_rgb_set.add(generate_random_rgb())

data_entry_1 = list(unique_rgb_set)

data_entry_2 = []
for r, g, b in data_entry_1:
    is_red = is_color_red(r, g, b)
    data_entry_2.append(generate_output(is_red))


data = {
    "RBG_Values": data_entry_1,
    "Is_Red": data_entry_2
}

with open("color_data.json", "w") as file:
    json.dump(data, file)