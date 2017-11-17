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
