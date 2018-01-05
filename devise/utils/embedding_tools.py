import torch
import torch.nn as nn
from torch.autograd import Variable


class EmbeddingTools():
    def __init__(self, cifar10_embeddings_dict=None, num_classes=None, embedding_dim=None):
        self.cifar10_embeddings_dict = cifar10_embeddings_dict
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.cos = nn.CosineSimilarity(dim=1)
        self.prepare_class_embeddings()

    def prepare_class_embeddings(self):
        class_embeddings = torch.Tensor(self.num_classes, self.embedding_dim)
        for c in sorted(self.cifar10_embeddings_dict.keys()):
            class_embedding = torch.Tensor(self.cifar10_embeddings_dict[c])
            class_embeddings[c, :] = class_embedding / torch.norm(class_embedding)  # normalize all the t vectors

        self.class_embeddings = Variable(class_embeddings).cuda()

    def lookup_embedding_vectors(self, inputs, targets):
        target_embeddings = torch.Tensor(inputs.size())
        for i, target in enumerate(targets):
            target_embedding = self.cifar10_embeddings_dict[target.data[0]]
            target_embeddings[i, :] = torch.Tensor(target_embedding)

        target_embeddings = Variable(target_embeddings).cuda()
        return target_embeddings

    def find_batch_predictions(self, outputs):
        batch_size = outputs.size(0)
        predictions = Variable(torch.Tensor(batch_size, self.num_classes)).cuda()
        for i, output in enumerate(outputs):
            output_expanded = output.expand(self.num_classes, self.embedding_dim)
            similarities = self.cos(output_expanded, self.class_embeddings)
            predictions[i, :] = similarities
            
        predictions = predictions.max(dim=1)[1]
        return predictions
        
    def prepare_all_class_embeddings(self, batch_size):
        # generate embedding vector matrices for all class
        class_embeddings_expanded = []
        for class_embedding in self.class_embeddings:
            class_embedding_expanded = class_embedding.expand(batch_size, class_embedding.size(0))
            class_embeddings_expanded.append(class_embedding_expanded)

        return class_embeddings_expanded

    def prepare_all_class_embeddings_new(self):
        return self.class_embeddings
