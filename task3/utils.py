import torch


def similarity_score(input1, input2):
    # Get similarity predictions:
    dif = input1 - input2

    norm = torch.norm(dif, p=1, dim=dif.dim() - 1)
    y_hat = torch.exp(-norm)
    y_hat = torch.clamp(y_hat, min=1e-7, max=1.0 - 1e-7)
    return y_hat


def cosine_similarity(input1, input2):
    ## calculate cosine similarity like it is used in sentence-BERT

    sim = torch.nn.CosineSimilarity()
    y_hat = sim(input1, input2)
    return y_hat
