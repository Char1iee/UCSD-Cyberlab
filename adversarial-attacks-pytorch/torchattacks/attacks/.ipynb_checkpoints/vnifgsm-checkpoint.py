import torch
import torch.nn as nn

from ..attack import Attack

class VNIFGSM(Attack):
    r"""
    VNI-FGSM adapted for tabular data.
    Distance Measure: Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation for each feature. (Default: 0.1)
        alpha (float): step size for each feature. (Default: 0.02)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)
        N (int): number of sampled examples in the neighborhood. (Default: 5)
        beta (float): upper bound of neighborhood scaling. (Default: 1.5)
    Shape:
        - data: (N, F) where N = number of samples, F = number of features.
        - labels: (N) where each value y_i is 0 <= y_i <= number of labels.
        - output: (N, F).
    """
    def __init__(
        self, model, eps=0.1, alpha=0.02, steps=10, decay=1.0, N=5, beta=1.5
    ):
        super().__init__("VNIFGSM", model)
        self.eps = eps  # 调整为适合表格数据的扰动范围
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
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

        # 初始化动量和梯度方差张量，适应表格数据的形状 (N, F)
        momentum = torch.zeros_like(data).detach().to(self.device)
        v = torch.zeros_like(data).detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_data = data.clone().detach()

        for _ in range(self.steps):
            adv_data.requires_grad = True
            nes_data = adv_data + self.decay * self.alpha * momentum
            outputs = self.get_logits(nes_data)

            # 计算损失
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # 计算对抗梯度
            adv_grad = torch.autograd.grad(
                cost, adv_data, retain_graph=False, create_graph=False
            )[0]

            # 标准化梯度，适应表格数据的二维形状
            grad = (adv_grad + v) / torch.mean(
                torch.abs(adv_grad + v), dim=1, keepdim=True
            )
            grad = grad + momentum * self.decay
            momentum = grad

            # 计算梯度方差
            GV_grad = torch.zeros_like(data).detach().to(self.device)
            for _ in range(self.N):
                # 生成邻域样本，针对表格数据调整噪声范围
                neighbor_data = adv_data.detach() + torch.randn_like(
                    data
                ).uniform_(-self.eps * self.beta, self.eps * self.beta)
                neighbor_data.requires_grad = True
                outputs = self.get_logits(neighbor_data)

                if self.targeted:
                    cost = -loss(outputs, target_labels)
                else:
                    cost = loss(outputs, labels)
                GV_grad += torch.autograd.grad(
                    cost, neighbor_data, retain_graph=False, create_graph=False
                )[0]

            # 计算梯度方差
            v = GV_grad / self.N - adv_grad

            # 更新对抗数据
            adv_data = adv_data.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_data - data, min=-self.eps, max=self.eps)
            adv_data = (data + delta).detach()  # 不强制限制 [0, 1]，根据实际数据范围调整

        return adv_data
