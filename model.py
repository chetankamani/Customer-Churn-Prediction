import torch
import torch.nn as nn

# solve runtime error mat1 and mat2 must have the same shape, but they have shapes [32, 61] and [20, 64]
# network should have four hidden layers 64, 32 , 16 and 8 neurons respectively, and an output layer with 2 neurons for binary classification
# add batch normalization layers after each hidden layer to improve training stability and performance
# add skip connections from the input layer to the second and third hidden layers to help with gradient flow and mitigate vanishing gradient issues
# the size of the tensor a 32 must match the ize of tensor b at non-singleton dimension, so we need to make sure that the output of the first hidden layer has 32 neurons to match the input of the second hidden layer, and the output of the second hidden layer has 16 neurons to match the input of the third hidden layer

class CustomerChurnPredictor(nn.Module):
    def __init__(self):
        super(CustomerChurnPredictor, self).__init__()
        self.fc1 = nn.Linear(61, 64)
        # self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        # self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        # self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        # self.bn4 = nn.BatchNorm1d(8)
        self.output = nn.Linear(8, 2)  # Output layer for binary classification

        # self.skip1 = nn.Linear(61, 32)  # Skip connection from input to second hidden layer
        # self.skip2 = nn.Linear(61, 16)  # Skip connection from input to third hidden layer

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1)) # + self.skip1(x)  # Skip connection from input to second hidden layer
        x3 = torch.relu(self.fc3(x2)) # + self.skip2(x)  # Skip connection from input to third hidden layer
        x4 = torch.relu(self.fc4(x3))
        out = self.output(x4)
        return out