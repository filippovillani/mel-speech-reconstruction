import torch
import torch.nn as nn
import torch.nn.functional as F      

def l2_regularization(model):
    
    l2_reg = 0.
    for param in model.parameters():
        l2_reg += torch.linalg.norm(param)
        
    return l2_reg

class ComplexMSELoss(nn.Module):
    def __init__(self):
        super(ComplexMSELoss, self).__init__() 
    
    def forward(self, s_target, s_hat):
        return torch.abs(torch.mean((s_target - s_hat) ** 2))

class FrobeniusLoss(nn.Module):
    def __init__(self):
        super(FrobeniusLoss, self).__init__()
    
    def forward(self, s_target, s_hat):
        return self._frobenius_loss(s_target, s_hat)

    def _frobenius_loss(self,
                        s_target: torch.Tensor,
                        s_hat: torch.Tensor)->torch.Tensor:
        batch_loss = torch.tensor([torch.linalg.norm(s_target[b] - s_hat[b], ord="fro") \
            for b in range(s_target.shape[0])], 
                                requires_grad=True,
                                device = s_target.device)
        return torch.mean(batch_loss)
