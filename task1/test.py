from scipy import stats
import logging
from sklearn.metrics import r2_score, mean_squared_error



def evaluate_test_set(model, data_loader, config_dict):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on test set")
    device = config_dict['device']
    true_scores = list()
    predicted_scores = list()

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
        pred,sent1_annotation_weight_matrix,sent2_annotation_weight_matrix = model(sent1.to(device),sent2.to(device))

        ## keep track of gold labels and predictions
        true_scores += list(targets.float())
        predicted_scores += list(pred.data.float().detach().cpu().numpy())

    ## computing different accuracy measurements
    acc = r2_score(true_scores, predicted_scores)
    r = stats.pearsonr(true_scores, predicted_scores)
    rho = stats.spearmanr(true_scores, predicted_scores)
    mse = mean_squared_error(true_scores, predicted_scores)

    return acc, r, rho, mse

