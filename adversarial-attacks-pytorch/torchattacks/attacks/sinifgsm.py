import torch
import torch.nn as nn

from ..attack import Attack


class SINIFGSM(Attack):
    r"""
    SI-NI-FGSM adapted for tabular data.
    Reference: 'Nesterov Accelerated Gradient and Scale-Invariance for Adversarial Attacks'
    [https://arxiv.org/abs/1908.06281], ICLR 2020
    Modified from "https://github.com/JHL-HUST/SI-NI-FGSM"

    Distance Measure: Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation for each feature. (Default: 0.1)
        alpha (float): step size for each feature. (Default: 0.02)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        m (int): number of scale copies. (Default: 5)

    Shape:
        - data: (N, F) where N = number of samples, F = number of features.
        - labels: (N) where each value y_i is 0 <= y_i <= number of labels.
        - output: (N, F).
    """
    def __init__(self, model, eps=0.1, alpha=0.02, steps=10, decay=1.0, m=5):
        super().__init__("SINIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m = m
        self.supported_mode = ["default", "targeted"]

    def forward(self, data, labels):
        r"""
        Overridden for tabular data.
        Arguments:
            data: (N, F) tensor, tabular input data.
            labels: (N) tensor, ground truth labels.
        """
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(data, labels)

        momentum = torch.zeros_like(data).detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_data = data.clone().detach()

        for _ in range(self.steps):
            adv_data.requires_grad = True
            nes_data = adv_data + self.decay * self.alpha * momentum

            # 计算多尺度拷贝的梯度和
            adv_grad = torch.zeros_like(data).detach().to(self.device)
            for i in torch.arange(self.m):
                nes_data_scaled = nes_data / torch.pow(2, i)
                outputs = self.get_logits(nes_data_scaled)
                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
                adv_grad += torch.autograd.grad(
                    cost, adv_data, retain_graph=False, create_graph=False
                )[0]
            adv_grad = adv_grad / self.m

            # 更新对抗数据
            grad = self.decay * momentum + adv_grad / torch.mean(
                torch.abs(adv_grad), dim=1, keepdim=True
            )
            momentum = grad
            adv_data = adv_data.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_data - data, min=-self.eps, max=self.eps)
            adv_data = (data + delta).detach()  # 不强制限制 [0, 1]，根据实际数据范围调整

        return adv_data