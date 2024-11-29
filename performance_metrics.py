from sklearn.metrics import confusion_matrix, classification_report


def get_Accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)


def get_Precision(TP, FP):
    return TP / (TP + FP)


def get_Recall(TP, FN):
    return TP / (TP + FN)


def get_F1_Score(TP, FP, FN):
    precision = get_Precision(TP, FP)
    recall = get_Recall(TP, FN)

    return (2*precision*recall) / (precision + recall)


def print_all_metrics(target_classifications, predicted_classifications):
    TN, FP, FN, TP = confusion_matrix(target_classifications, predicted_classifications).ravel()

    # Print metrics
    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"False Negatives (FN): {FN}")

    # Calculate accuracy, precision, recall, and F1-score
    report = classification_report(target_classifications, predicted_classifications, digits=4)
    adv_metrics = [get_Accuracy(TP, TN, FP, FN), get_Precision(TP, FP),
                   get_Recall(TP, FN), get_F1_Score(TP, FP, FN)]

    print("Classification Report:\n", report)

    return [TP, FP, TN, FN], adv_metrics
