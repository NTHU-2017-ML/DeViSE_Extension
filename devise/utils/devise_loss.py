import torch
import torch.nn as nn
from torch.autograd import Variable

class MaxMarginLoss(nn.Module):
    def __init__(self, embedding_tools=None, margin=0.1):
        super(MaxMarginLoss, self).__init__()
        self.margin = margin
        self.embedding_tools = embedding_tools
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, inputs, targets):
        """
        input: should be the embeddings output by the core-visual-model in the DeViSE framework
        target: should be the ground truth labels for the input
        """
        self.batch_size = targets.size(0)
        self.class_embeddings_expanded = self.embedding_tools.prepare_all_class_embeddings(self.batch_size)
        num_classes = len(self.class_embeddings_expanded)

        # use word2vec look-up-table to retrieve corresponding embedding vectors
        target_embeddings = self.embedding_tools.lookup_embedding_vectors(inputs, targets)
        
        #loss = torch.sum(inputs * target_embeddings, 1) * (-num_classes)
        loss = self.cos(inputs, target_embeddings) * (-num_classes)
        for class_embedding_expanded in self.class_embeddings_expanded:
            #loss = loss + torch.sum(inputs * class_embedding_expanded, 1)
            loss = loss + self.cos(inputs, class_embedding_expanded)

        zeros = Variable(torch.zeros(loss.size())).cuda()
        loss = torch.max(zeros, self.margin * (num_classes - 1) + loss).sum() / self.batch_size
        return loss
