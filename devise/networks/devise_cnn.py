import torch.nn as nn


def convert_devise_cnn(model, checkpoint, output_dimension):
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module
    """
    for param in model.parameters():
        param.requires_grad = False
    """

    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    input_dimension = model.linear.in_features
    model.linear = nn.Linear(input_dimension, output_dimension)
    return nn.DataParallel(model).cuda()

def convert_enhanced_devise_cnn(base_model, checkpoint, output_dimension):
    base_model.load_state_dict(checkpoint['state_dict'])
    base_model = base_model.module

    model = EnhancedDeviseCnn(base_model, output_dimension)
    return nn.DataParallel(model).cuda()

"""
class EnhancedDeviseCnn(nn.Module):
    # model_1

    def __init__(self, base_model, output_dimension):
        super(EnhancedDeviseCnn, self).__init__()

        # Replace the last fully-connected layer of the base model
        # Parameters of newly constructed modules have requires_grad=True by default
        base_linear_in_features = base_model.linear.in_features
        base_model.linear = nn.Linear(base_linear_in_features, output_dimension)

        self.base_model = base_model
        self.relu_1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.base_model(x)
        x = self.relu_1(x)
        return x
"""


class EnhancedDeviseCnn(nn.Module):
    # model_0

    def __init__(self, base_model, output_dimension):
        super(EnhancedDeviseCnn, self).__init__()

        # Replace the last fully-connected layer of the base model
        # Parameters of newly constructed modules have requires_grad=True by default
        base_linear_in_features = base_model.linear.in_features
        base_linear_out_features = int(base_linear_in_features / 4)
        base_model.linear = nn.Linear(base_linear_in_features, base_linear_out_features)

        self.base_model = base_model
        self.relu_1 = nn.ReLU(inplace=True)
        self.linear = nn.Linear(base_linear_out_features, output_dimension)

    def forward(self, x):
        x = self.base_model(x)
        x = self.relu_1(x)
        x = self.linear(x)
        return x
