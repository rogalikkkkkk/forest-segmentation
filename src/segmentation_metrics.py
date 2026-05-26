import numpy as np


def update_confusion_matrix(confusion_matrix, predictions, targets, num_classes):
    valid_pixels = (targets >= 0) & (targets < num_classes)

    encoded = num_classes * targets[valid_pixels] + predictions[valid_pixels]
    batch_confusion = np.bincount(
        encoded,
        minlength=num_classes * num_classes,
    )
    batch_confusion = batch_confusion.reshape(num_classes, num_classes)

    confusion_matrix += batch_confusion


def calculate_metrics(confusion_matrix):
    true_positive = np.diag(confusion_matrix)
    ground_truth_pixels = confusion_matrix.sum(axis=1)
    predicted_pixels = confusion_matrix.sum(axis=0)

    total_correct = true_positive.sum()
    total_pixels = confusion_matrix.sum()
    pixel_accuracy = total_correct / total_pixels

    union = ground_truth_pixels + predicted_pixels - true_positive
    valid_classes = union > 0
    iou_per_class = np.full(confusion_matrix.shape[0], np.nan, dtype=np.float64)
    iou_per_class[valid_classes] = true_positive[valid_classes] / union[valid_classes]
    mean_iou = np.nanmean(iou_per_class)

    return pixel_accuracy, mean_iou, iou_per_class


def get_class_status(gt_pixels, pred_pixels, true_positive):
    if gt_pixels == 0 and pred_pixels == 0:
        return "absent_in_gt_and_prediction"

    if gt_pixels == 0 and pred_pixels > 0:
        return "absent_in_gt_false_positive"

    if gt_pixels > 0 and pred_pixels == 0:
        return "present_in_gt_not_predicted"

    if true_positive == 0:
        return "present_but_no_true_positive"

    return "present"


def get_class_statistics(confusion_matrix):
    true_positive = np.diag(confusion_matrix)
    ground_truth_pixels = confusion_matrix.sum(axis=1)
    predicted_pixels = confusion_matrix.sum(axis=0)
    union_pixels = ground_truth_pixels + predicted_pixels - true_positive

    return ground_truth_pixels, predicted_pixels, true_positive, union_pixels
