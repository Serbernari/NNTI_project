from numpy import argmax, argmin
from scipy import stats
import logging
from sklearn.metrics import r2_score,mean_squared_error



def evaluate_test_set(model, data_loader, config_dict):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on test set")
    device = config_dict['device']
    true_scores = list()
    predicted_scores = list()

    ## keep track of tokenized sentences so we can do some analysis
    sentences1 = list()
    sentences2 = list()

    for (
        sent1,
        sent2,
        sents1_len,
        sents2_len,
        targets,
        _,
        _,
    ) in data_loader["test"]:
        ## forward pass
        (pred,sent1_annotation_weight_matrix,sent2_annotation_weight_matrix,) = model(sent1.to(device),sent2.to(device))

        ## keep track of gold labels and predictions
        true_scores += list(targets.float())
        predicted_scores += list(pred.data.float().detach().cpu().numpy())

        ## keep track of tokenized sentences so we can do some analysis
        sentences1 += list(sent1)
        sentences2 += list(sent2)

    ## computing different accuracy measures
    acc = r2_score(true_scores, predicted_scores)
    r = stats.pearsonr(true_scores, predicted_scores)
    rho = stats.spearmanr(true_scores, predicted_scores)
    mse = mean_squared_error(true_scores, predicted_scores)

    print('Worst score is {} and should be {}'.format(predicted_scores[argmin(predicted_scores)]*5.0, true_scores[argmin(predicted_scores)]*5.0))
    print('Best score is {} and should be {}'.format(predicted_scores[argmax(predicted_scores)]*5.0, true_scores[argmax(predicted_scores)]*5.0))
    print(sentences1[argmax(predicted_scores)])
    print(sentences2[argmax(predicted_scores)])


    return acc, r, rho, mse, sentences1[argmin(predicted_scores)], sentences2[argmin(predicted_scores)], sentences1[argmax(predicted_scores)], sentences2[argmax(predicted_scores)]

