def calculate_accuracy(y_true, y_pred):
    correct_predictions = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    return correct_predictions / len(y_true)

def calculate_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def calculate_f1_score(y_true, y_pred):
    tp = sum((y_t == 1 and y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    fp = sum((y_t == 0 and y_p == 1) for y_t, y_p in zip(y_true, y_pred))
    fn = sum((y_t == 1 and y_p == 0) for y_t, y_p in zip(y_true, y_pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)