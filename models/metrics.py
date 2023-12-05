from sklearn import metrics

def accuracy_score(pred_label, true_label):
    return metrics.accuracy_score(true_label, pred_label) * 100

def f1_score(pred_label, true_label):
    return metrics.f1_score(true_label, pred_label, average="macro")