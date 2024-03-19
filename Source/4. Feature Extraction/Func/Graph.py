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