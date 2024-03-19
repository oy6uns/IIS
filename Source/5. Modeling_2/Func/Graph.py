def make_confusion_matrix(predicted, labels):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import numpy as np

    if type(predicted) != np.ndarray:
        # 텐서를 CPU로 이동시키고 NumPy 배열로 변환
        all_predicted_cpu = predicted.cpu().numpy()
        all_labels_cpu = labels.cpu().numpy()
    else:
        all_predicted_cpu = predicted
        all_labels_cpu = labels
  
    # 혼동 행렬 계산
    cm = confusion_matrix(all_labels_cpu, all_predicted_cpu)

    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='g', cmap='Blues')
    plt.rc('font', size=16) 
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.rc('axes', labelsize=20)   # x,y축 label 폰트 크기
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16) 
    title_font = {
        'fontsize': 16
    }
    plt.title('Confusion Matrix', fontdict=title_font)
    plt.show()

    # 라벨별 정확도 계산
    def calculate_label_accuracy(y_true, y_pred):
        # Unique classes in the true labels
        labels = np.unique(y_true)
        label_accuracies = {}

        # Calculate accuracy for each label
        for label in labels:
            # Indices where the true label is equal to the current label
            true_indices = np.where(np.array(y_true) == label)[0]
            # Subset of predicted labels for these indices
            pred_labels_for_true = np.array(y_pred)[true_indices]
            # Calculate accuracy as the fraction of correct predictions for this label
            accuracy = np.mean(pred_labels_for_true == label)
            label_accuracies[label] = accuracy

        return label_accuracies

    # Calculate label accuracies
    label_accuracies = calculate_label_accuracy(all_labels_cpu, all_predicted_cpu)

    # Calculate the average accuracy across labels
    average_accuracy = np.mean(list(label_accuracies.values()))

    print(label_accuracies, average_accuracy)