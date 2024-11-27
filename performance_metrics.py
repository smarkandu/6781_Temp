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


def get_all_metrics(TP, TN, FP, FN):
    return get_Accuracy(TP, TN, FP, FN), get_Precision(TP, FP), get_Recall(TP, FN), get_F1_Score(TP, FP, FN)


def print_all_metrics(TP, TN, FP, FN):
    accuracy, precision, recall, f1_score = get_all_metrics(TP, TN, FP, FN)
    print('TP: ', TP)
    print('TN: ', TN)
    print('FP: ', FP)
    print('FN: ', FN)
    print('\n')
    print('accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1_score: ', f1_score)


def get_basic_performance_metrics(predicted_classifications, target_classifications, positive_classification):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(0, len(predicted_classifications)):
        predicted_classification = predicted_classifications[i]
        target_classification = target_classifications[i]

        if predicted_classification == target_classification:
            if predicted_classification == positive_classification:
                TP += 1
            else:
                TN += 1
        else:
            if predicted_classification == positive_classification:
                FP += 1
            else:
                FN += 1

    return TP, FP, TN, FN
