import torch
import torch.nn as nn
from ..attack import Attack

class VMIFGSM(Attack):
    def __init__(
        self, model, eps=0.1, alpha=0.02, steps=10, decay=1.0, N=5, beta=1.5
    ):
        super().__init__("VMIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.N = N
        self.beta = beta
        self.supported_mode = ["default", "targeted"]

    def forward(self, data, labels):
        data = data.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(data, labels)

        momentum = torch.zeros_like(data).detach().to(self.device)
        v = torch.zeros_like(data).detach().to(self.device)
        loss = nn.CrossEntropyLoss()
        adv_data = data.clone().detach()

        for _ in range(self.steps):
            adv_data.requires_grad = True
            outputs = self.get_logits(adv_data)

            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            adv_grad = torch.autograd.grad(
                cost, adv_data, retain_graph=False, create_graph=False
            )[0]

            grad = (adv_grad + v) / torch.mean(
                torch.abs(adv_grad + v), dim=1, keepdim=True
            )
            grad = grad + momentum * self.decay
            momentum = grad

            GV_grad = torch.zeros_like(data).detach().to(self.device)
            for _ in range(self.N):
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

            v = GV_grad / self.N - adv_grad

            adv_data = adv_data.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_data - data, min=-self.eps, max=self.eps)
            adv_data = (data + delta).detach()

        return adv_data