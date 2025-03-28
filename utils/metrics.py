import torch
from collections import Counter
from sklearn.metrics import classification_report

EPSILON = 1e-10

class Accuracy(object):
    def __init__(self,topK=1):
        super(Accuracy,self).__init__()
        self.topK = topK
    def __call__(self, output, target):
        batch_size = target.size(0)
        _, pred = output.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topK].view(-1).float().sum(0)
        result = correct_k / batch_size
        return result

def precision(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    true_positives = Counter()
    predicted_positives = Counter()
    y_pred = y_pred.tolist() if torch.is_tensor else y_pred
    y_true = y_true.tolist() if torch.is_tensor else y_true
    for true, pred in zip(y_true, y_pred):
        if pred == true:
            true_positives[true] += 1
        predicted_positives[pred] += 1
    classes = sorted(true_positives.keys())
    prec = dict()
    for i in classes:
        prec[i] = true_positives[i] / (predicted_positives[i] + EPSILON)

    return prec

def recall(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    true_positives = Counter()
    total = Counter()
    y_pred = y_pred.tolist() if torch.is_tensor else y_pred
    y_true = y_true.tolist() if torch.is_tensor else y_true
    for true, pred in zip(y_true, y_pred):
        if pred == true:
            true_positives[true] += 1
        total[true] += 1
    classes = sorted(true_positives.keys())
    recall = dict()
    for i in classes:
        recall[i] = true_positives[i] / (total[i] + EPSILON)
    return recall


def f1Score(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    prec = precision(y_pred, y_true)
    rc= recall(y_pred, y_true)
    f1 = dict()
    for i in rc.keys():
        f1[i] = 2 * prec[i] * rc[i] / (prec[i] + rc[i])
    return f1

def mse(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    return torch.mean((y_true - y_pred)**2).numpy()

def mae(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    return torch.mean(torch.abs(y_true - y_pred))
