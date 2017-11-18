import os
import argparse

import torch
import torch.nn as nn

from networks import resnet
from networks import toy_cnn
from networks import devise_cnn
from utils.training_flow import TrainingFlow
from utils import devise_loss
from utils.accuracy import Accuracy
from utils import pickle_tools
from utils.embedding_tools import EmbeddingTools


parser = argparse.ArgumentParser(description='PyTorch Devise Finetuning')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to the latest checkpoint (default: none)')


if __name__ == '__main__':
    # Parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    cifar10_embeddings_dict_name = '../word2vec/cifar-10_embeddeds.pkl'

    csv_log_name = 'records/finetune_log.csv'
    checkpoint_name = 'records/finetune_checkpoint.pth.tar'
    best_model_name = 'records/finetune_best_model.pth.tar'
    pretrained_checkpoint_name = 'records/best_model.pth.tar'

    word2vec_embedding_dim = 300

    arch = 'ResNet18'
    optimizer_type = 'SGD'

    args = parser.parse_args()

    # Hyper-parameters
    epochs = 350
    lr = args.lr
    batch_size = 128
    saturate_patience = 12
    reduce_patience = 4
    margin = 0.1

    # Initialize
    cifar10_embeddings_dict = pickle_tools.load_pickle(cifar10_embeddings_dict_name)
    embedding_tools = EmbeddingTools(cifar10_embeddings_dict=cifar10_embeddings_dict, num_classes=len(classes), embedding_dim=word2vec_embedding_dim)

    pretrained_checkpoint = torch.load(pretrained_checkpoint_name)
    base_model = resnet.ResNet18()
    base_model = nn.DataParallel(base_model).cuda()
    model = devise_cnn.convert_devise_cnn(base_model, pretrained_checkpoint, word2vec_embedding_dim)

    #params_to_optimize = model.module.linear.parameters()
    params_to_optimize = model.parameters()
    loss_function = devise_loss.MaxMarginLoss(embedding_tools=embedding_tools, margin=margin)
    
    compute_batch_accuracy = Accuracy(embedding_tools=embedding_tools).compute_embedding_batch_accuracy

    training_flow = TrainingFlow(model=model,
                                 params_to_optimize=params_to_optimize,
                                 loss_function=loss_function,
                                 compute_batch_accuracy=compute_batch_accuracy,
                                 epochs=epochs,
                                 lr=lr,
                                 batch_size=batch_size,
                                 classes=classes,
                                 saturate_patience=saturate_patience,
                                 reduce_patience=reduce_patience,
                                 csv_log_name=csv_log_name,
                                 checkpoint_name=checkpoint_name,
                                 best_model_name=best_model_name,
                                 arch=arch,
                                 optimizer_type=optimizer_type,
                                 args=args)

    # Start training
    training_flow.train()
