class Accuracy():
    
    def __init__(self, embedding_tools=None):
        self.embedding_tools = embedding_tools

    def compute_batch_accuracy(self, outputs, targets):
        predictions = outputs.max(dim=1)[1]
        correct = ((predictions == targets).sum()).data[0]
        total = targets.size(0)
        accuracy = float(correct) / float(total)
        return correct, accuracy

    def compute_embedding_batch_accuracy(self, outputs, targets):
        predictions = self.embedding_tools.find_batch_predictions(outputs)
        correct = ((predictions == targets).sum()).data[0]
        total = targets.size(0)
        accuracy = float(correct) / float(total)
        return correct, accuracy
