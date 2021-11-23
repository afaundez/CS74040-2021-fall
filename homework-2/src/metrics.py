from src.structures.matrix import Matrix

class Metrics:
    def score(true_labels, pred_labels, labeler):
        labels = labeler.labels()
        confusion = Matrix(labels, labels, default=0, name='true\predicted')
        for true_label, pred_label in zip(true_labels, pred_labels):
            confusion[labeler.encode(true_label), labeler.encode(pred_label)] += 1

        correct = sum(confusion[i, i] for i in range(len(labels)))
        total = sum(sum(confusion[i]) for i in range(len(labels)))
        accuracy = 1. * correct / total
        results = {
            'accuracy': accuracy,
            'confusion': confusion
        }
        return results
