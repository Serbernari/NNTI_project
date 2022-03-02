import torch


def similarity_score(input1, input2):
    # Get similarity predictions:
    dif = input1 - input2

    norm = torch.norm(dif, p=1, dim=dif.dim() - 1)
    y_hat = torch.exp(-norm)
    y_hat = torch.clamp(y_hat, min=1e-7, max=1.0 - 1e-7)
    #print(y_hat)
    return y_hat
