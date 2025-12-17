from sklearn.metrics import confusion_matrix

def precision_recall_f1(true_labels, pred_labels, debug=False):
    """计算精确率、召回率和F1分数"""
    cm = confusion_matrix(true_labels, pred_labels, labels=[True, False])
    tp, fn, fp, tn = cm.ravel()
    if debug:
        print(f"混淆矩阵 - tp: {tp}, fn: {fn}, fp: {fp}, tn: {tn}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1