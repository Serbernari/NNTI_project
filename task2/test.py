from numpy import argmax, argmin
from scipy import stats
import logging
import sklearn
import torch
from sklearn.metrics import r2_score,mean_squared_error



def evaluate_test_set(model, data_loader, config_dict):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on test set")
    device = config_dict['device']
    y_true = list()
    y_pred = list()
    sentences1 = list()
    sentences2 = list()
    #total_loss = 0
    for (
        sent1,
        sent2,
        sents1_len,
        sents2_len,
        targets,
        _,
        _,
    ) in data_loader["test"]:
        ## perform forward pass
        pred = None
        loss = None

        ## perform forward pass

        (pred,sent1_annotation_weight_matrix,sent2_annotation_weight_matrix,) = model(sent1.to(device),sent2.to(device))
        y_true += list(targets.float())
        y_pred += list(pred.data.float().detach().cpu().numpy())
        sentences1 += list(sent1)
        sentences2 += list(sent2)
        #total_loss += loss
    ## computing accuracy using sklearn's function
    acc = r2_score(y_true, y_pred)
    r = stats.pearsonr(y_true, y_pred)
    rho = stats.spearmanr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print('Worst score is {} and should be {}'.format(y_pred[argmin(y_pred)]*5.0, y_true[argmin(y_pred)]*5.0))
    print('Best score is {} and should be {}'.format(y_pred[argmax(y_pred)]*5.0, y_true[argmax(y_pred)]*5.0))
    print(sentences1[argmax(y_true)])
    print(sentences2[argmax(y_true)])


    return acc, r, rho, mse, sentences1[argmin(y_pred)], sentences2[argmin(y_pred)], sentences1[argmax(y_pred)], sentences2[argmax(y_pred)]

