import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=[1, 1, 1], gamma=2, reduction='mean', device='cpu'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha = self.alpha[targets] if len(self.alpha) > 1 else self.alpha
        F_loss = alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class MultiClassF1Loss(nn.Module):
    def __init__(self, num_classes=3, epsilon=1e-7):
        super(MultiClassF1Loss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # 원-핫 인코딩 변환
        y_true_one_hot = F.one_hot(y_true, num_classes=self.num_classes)

        # 소프트맥스 적용하여 확률을 계산
        y_pred_softmax = F.softmax(y_pred, dim=1)
        
        # y_pred를 이진 형태로 변환
        y_pred_one_hot = y_pred_softmax.round()

        # 클래스별 true positive, predicted positive, actual positive 계산
        true_positive = torch.sum(y_true_one_hot * y_pred_one_hot, dim=0)
        predicted_positive = torch.sum(y_pred_one_hot, dim=0)
        actual_positive = torch.sum(y_true_one_hot, dim=0)

        # 클래스별 F1 점수 계산
        precision = true_positive / (predicted_positive + self.epsilon)
        recall = true_positive / (actual_positive + self.epsilon)
        f1_score = 2 * precision * recall / (precision + recall + self.epsilon)
        f1_score = f1_score[torch.isfinite(f1_score)]  # NaN 제거

        # 평균 F1 점수 계산
        avg_f1_score = torch.mean(f1_score)

        # F1 손실 (1 - F1 점수)
        f1_loss = 1 - avg_f1_score

        # 멀티클래스 교차 엔트로피 손실 추가
        cross_entropy_loss = F.cross_entropy(y_pred, y_true)

        # 최종 손실: 교차 엔트로피와 F1 손실의 평균
        combined_loss = (cross_entropy_loss + f1_loss) / 2
        return combined_loss
    
class SigmoidF1Loss(nn.Module):
    def __init__(self, num_classes=3, beta=1.0, eta=0.0):
        super(SigmoidF1Loss, self).__init__()
        self.beta = beta
        self.eta = eta
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        y_true_one_hot = F.one_hot(y_true, num_classes=self.num_classes).float()
        sig = 1 / (1 + torch.exp(-self.beta * (y_pred + self.eta)))
        tp = torch.sum(sig * y_true_one_hot, dim=0)
        fp = torch.sum(sig * (1 - y_true_one_hot), dim=0)
        fn = torch.sum((1 - sig) * y_true_one_hot, dim=0)
        sigmoid_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
        return 1 - sigmoid_f1.mean()