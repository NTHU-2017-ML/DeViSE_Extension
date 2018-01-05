import torch
import torch.nn as nn
from torch.autograd import Variable

class MaxMarginLoss(nn.Module):
    def __init__(self, embedding_tools=None, margin=0.1):
        super(MaxMarginLoss, self).__init__()
        self.margin = margin
        self.embedding_tools = embedding_tools
        #self.cos = nn.CosineSimilarity(dim=0)
        self.cos = nn.CosineSimilarity(dim=1)
        
        self.class_embeddings = self.embedding_tools.class_embeddings
        self.num_classes = self.class_embeddings.size(0)
        self.embedding_dim = self.class_embeddings.size(1)

    def forward(self, inputs, targets):
        """
        input: should be the embeddings output by the core-visual-model in the DeViSE framework
        target: should be the ground truth labels for the input
        """
        batch_size = targets.size(0)
        self.class_embeddings_expanded = self.embedding_tools.prepare_all_class_embeddings(batch_size)
        num_classes = len(self.class_embeddings_expanded)

        # use word2vec look-up-table to retrieve corresponding embedding vectors
        target_embeddings = self.embedding_tools.lookup_embedding_vectors(inputs, targets)

        """
        loss = self.cos(inputs, target_embeddings) * (-num_classes)
        for class_embedding_expanded in self.class_embeddings_expanded:
            loss = loss + self.cos(inputs, class_embedding_expanded)

        zeros = Variable(torch.zeros(loss.size())).cuda()
        loss = torch.max(zeros, self.margin * (num_classes - 1) + loss).sum() / self.batch_size
        import pdb; pdb.set_trace()
        return loss
        """

        """
        batch_size = targets.size(0)
        loss = Variable(torch.zeros(1), requires_grad=True).cuda()
        
        for i in range(batch_size):
            input = inputs[i, :]
            target = int(targets[i])
            current_sample_loss = Variable(torch.zeros(1), requires_grad=True).cuda()
            for j in range(self.num_classes):
                if j == target:
                    continue

                max_second_term = self.margin - self.cos(self.class_embeddings[target, :], input) + self.cos(self.class_embeddings[j, :], input)
                zero = Variable(torch.zeros(max_second_term.size()), requires_grad=True).cuda()

                this_label_loss = torch.max(zero, max_second_term)
                current_sample_loss += this_label_loss

            loss += current_sample_loss

        loss = loss / batch_size
        return loss
        """

        # cosine of (inputs, all corresponding target embedded vectors), final size would be batch_size * num_classes
        target_cosine_term = self.cos(inputs, target_embeddings)
        target_cosine_term_added_dim = target_cosine_term.unsqueeze(1)
        target_cosine_term_stacked_along_dim1 = target_cosine_term_added_dim.repeat(1, self.num_classes)

        # cosine of (inputs, all classes embedded vectors), final size should be also batch_size * num_classes
        class_embeddings_added_dim = self.class_embeddings.unsqueeze(2)
        class_embeddings_transpose = class_embeddings_added_dim.transpose(0, 2)
        class_embeddings_stacked_along_dim0 = class_embeddings_transpose.repeat(batch_size, 1, 1)
        inputs_added_dim = inputs.unsqueeze(2)
        inputs_stacked_along_dim2 = inputs_added_dim.repeat(1, 1, self.num_classes)
        classes_cosine_term = self.cos(inputs_stacked_along_dim2, class_embeddings_stacked_along_dim0)

        # now calculate all the loss terms
        max_second_terms = self.margin - target_cosine_term_stacked_along_dim1 + classes_cosine_term
        zeros = Variable(torch.zeros([batch_size, self.num_classes])).cuda()
        max_results = torch.max(zeros, max_second_terms)
        sum_results = max_results.sum() - self.margin  # since we didn't jump over the terms when j == label, we need to subtract one margin
        loss = sum_results.mean()
        return loss
