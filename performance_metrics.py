from sklearn.metrics import confusion_matrix, classification_report


def print_all_metrics(target_classifications, predicted_classifications):
    tn, fp, fn, tp = confusion_matrix(target_classifications, predicted_classifications).ravel()

    # Print metrics
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")

    # Calculate accuracy, precision, recall, and F1-score
    report = classification_report(target_classifications, predicted_classifications, digits=4)
    dict_metrics = classification_report(target_classifications, predicted_classifications, digits=4, output_dict=True)
    adv_metrics = [dict_metrics["accuracy"], dict_metrics["1"]["precision"],
                   dict_metrics["1"]["recall"], dict_metrics["1"]["f1-score"]]

    print("Classification Report:\n", report)

    return [tp, fp, tn, fn], adv_metrics
