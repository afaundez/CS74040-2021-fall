class Metrics:
    def score(true_labels, pred_labels, verbose=True):
        tp = 0
        fp = 0
        for true_label, pred_label in zip(true_labels, pred_labels):
            if true_label == pred_label:
                tp += 1
            else:
                fp += 1
        accuracy = tp / (tp + fp)
        results = { 'accuracy': accuracy }
        if verbose:
            print({ 'Metrics.score': results })
        return results
