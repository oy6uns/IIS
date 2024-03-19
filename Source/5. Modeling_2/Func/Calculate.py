def division(a, b):
    if b == 0:
        return 0
    else:
        return a/b
    
def calculate_accuracy(predicted, labels):
    predicted = predicted.tolist()
    labels = labels.tolist()
    correct = [0, 0, 0]
    count = []
    for i in range(3):
        count.append(labels.count(i))
    print(count)
    for i, j in zip(predicted, labels):
        if i == j:
            correct[j] += 1
    print(f'''Label 1: 전체 {count[0]}개 중 {correct[0]}개 맞춤, 예측 성공률 {division(correct[0] * 100, count[0]):.2f}%
Label 2: 전체 {count[1]}개 중 {correct[1]}개 맞춤, 예측 성공률 {division(correct[1] * 100, count[1]):.2f}%
Label 3: 전체 {count[2]}개 중 {correct[2]}개 맞춤, 예측 성공률 {division(correct[2] * 100, count[2]):.2f}%''')