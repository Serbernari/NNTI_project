import torch
import matplotlib.pyplot as plt
import numpy as np

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



def plot_attention_maps(input_data, attn_maps, idx=0, save =False, use_words=False, sick_data=None):
    '''
    takes the tokenized sentence tensors as input and draws the color maps from the corresponding attention matrices.
    Function taken from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html and modified
    '''
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
        ## index where padding starts for sentence - we dont need to plot the padding attention
        padding_start = np.where(input_data == 1)[0][0]
        input_data = input_data[:padding_start]
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])

    ## attn_maps shape: (n_layer, batch_size, n_heads, seq_len, seq_len)
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    
    fig_size = 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ## remove padding attentions (they are set to 0 because of the mask anyway)
            temp = np.zeros((padding_start, padding_start))
            temp = attn_maps[row][column][:padding_start]
            temp = temp[:, :padding_start]

            ## plot matrices using imshow
            ax[row][column].imshow(temp, origin='lower', cmap='hot')
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist(), rotation=45)
            ax[row][column].set_yticklabels(input_data.tolist())
            if use_words: ## convert tokens to words and plot those along axis
                ax[row][column].set_xticklabels([sick_data.vocab.itos[ind] for ind in input_data.tolist()], rotation=45)
                ax[row][column].set_yticklabels([sick_data.vocab.itos[ind] for ind in input_data.tolist()])

            ax[row][column].set_title(f"Layer {row+1}, Head {column+1}, Sen {idx}")
    fig.subplots_adjust(hspace=0.5)
    if save:
        plt.savefig('attention_heat_maps')
    plt.show()