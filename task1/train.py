import random
import torch
from torch import nn
from sklearn.metrics import r2_score, f1_score
import logging
from tqdm import tqdm
from siamese_lstm_attention import SiameseBiLSTMAttention
from scipy import stats


logging.basicConfig(level=logging.INFO)

"""
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""


def train_model(model, optimizer, dataloader, data, max_epochs, config_dict):
    device = config_dict["device"]
    criterion = nn.MSELoss()  
    max_accuracy = 1e-1

    ## to keep track of training parameters
    best_model = None
    train_losses = list()
    val_losses = list()
    train_accs = list()
    val_accs = list()

    for epoch in tqdm(range(max_epochs)):

        logging.info("Epoch: {}".format(epoch))
        ## keep track of predictions and gold labels
        true_scores = list()
        predicted_scores = list()

        total_loss = 0.0
        for (
            sent1,
            sent2,
            sents1_len,
            sents2_len,
            targets,
            _,
            _,
        ) in dataloader["train"]:
            model.train()
            model.zero_grad()

            ## forward pass
            pred, sent1_annotation_weight_matrix, sent2_annotation_weight_matrix = model(sent1.to(device), sent2.to(device))

            ## get loss for attention penalty as shown in AAAI paper
            sent1_attention_loss = attention_penalty_loss(
                sent1_annotation_weight_matrix,
                config_dict["self_attention_config"]["penalty"],
                device,
            )
            sent2_attention_loss = attention_penalty_loss(
                sent2_annotation_weight_matrix,
                config_dict["self_attention_config"]["penalty"],
                device,
            )

            ##  loss with penalty
            loss = (
                criterion( pred.to(device),targets.float().to(device))
                + sent1_attention_loss
                + sent2_attention_loss
            )

            ## backward pass
            loss.backward()

            # clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            ## update gradients
            optimizer.step()

            ## save gold labels
            true_scores += list(targets.float().numpy())

            ## save output predictions
            predicted_scores += list(pred.data.float().detach().cpu().numpy())

            ## keep track of loss over batches
            total_loss += loss

        ## computing accuracy using pearson correlation
        acc, p = stats.pearsonr(true_scores, predicted_scores)

        ## compute model metrics on dev set
        val_acc, val_loss = evaluate_dev_set(
            model, data, criterion, dataloader, config_dict, device
        )

        if val_acc > max_accuracy:
            max_accuracy = val_acc
            best_model = model
            logging.info(
                "new model saved"
            )  ## save the model if it is better than the prior best
            torch.save(best_model.state_dict(), "{}.pth".format(config_dict["model_name"]))


        logging.info(
            "Train loss: {} - acc: {} -- Validation loss: {} - acc: {}".format(
                total_loss.data.float()/len(dataloader["train"]), acc, val_loss, val_acc
            )
        )

        ## save losses and accuracy for visualization
        train_losses.append((total_loss.data.float().item()/len(dataloader["train"])))
        train_accs.append(acc.item())
        val_losses.append(val_loss.item())
        val_accs.append(val_acc.item())

    return best_model, train_losses, train_accs, val_losses, val_accs

def train_hyperparameters(model, optimizer, dataloader, data, max_epochs, config_dict):
    '''
    training loop used when trying to find best hyperparameter combination.
    Does not log as much info or keep track of as many lists
    '''

    device = config_dict["device"]
    criterion = nn.MSELoss() 
    max_accuracy = 0.0

    for epoch in tqdm(range(max_epochs)):
        ## keep track of predictions and gold labels
        true_scores = list()
        predicted_scores = list()
        total_loss = 0.0
        model.train()
        for (
            sent1,
            sent2,
            sents1_len,
            sents2_len,
            targets,
            _,
            _,
        ) in dataloader["train"]:
            
            model.zero_grad()

            ## forward pass
            pred,sent1_annotation_weight_matrix,sent2_annotation_weight_matrix = model(sent1.to(device), sent2.to(device))
            
            ## get loss for attention penalty as shown in AAAI paper
            sent1_attention_loss = attention_penalty_loss(
                sent1_annotation_weight_matrix,
                config_dict["self_attention_config"]["penalty"],
                device,
            )
            sent2_attention_loss = attention_penalty_loss(
                sent2_annotation_weight_matrix,
                config_dict["self_attention_config"]["penalty"],
                device,
            )

            ## compute loss with penalty
            loss = (
                criterion(pred.to(device), targets.float().to(device))
                + sent1_attention_loss
                + sent2_attention_loss
            )

            ## backward pass
            loss.backward()

            # clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            ## update gradients
            optimizer.step()

            ## save labels
            true_scores += list(targets.float().numpy())

            ## save output predictions
            predicted_scores += list(pred.data.float().detach().cpu().numpy())

            ## accumulate train loss
            total_loss += loss

        model.zero_grad()

        ## compute model metrics on dev set
        val_acc, val_loss = evaluate_dev_set(
            model, data, criterion, dataloader, config_dict, device
        )

        if val_acc > max_accuracy:
            max_accuracy = val_acc

    logging.info(max_accuracy)
    return max_accuracy

def init_candidate():
    '''
    randomly generate hyperparameter combination for creating a model
    '''
    hidden_sizes = [64,128,192,256,320,400,512,600]
    learning_rates = [1e-3,2e-3,5e-4,8e-5]
    fc_hidden_sizes = [64, 128, 256, 400]
    attention_hidden_sizes = [128,256,400]
    attention_output_sizes = [10,20,30,40,45]
    

    hidden_size = random.choice(hidden_sizes)
    lstm_layer = random.randint(1,7)+1
    learning_rate = random.choice(learning_rates)
    fc_hidden_size = random.choice(fc_hidden_sizes)
    dropout = random.random()*0.5
    attention_hidden_size = random.choice(attention_hidden_sizes)
    attention_output_size = random.choice(attention_output_sizes)
    attention_penalty = random.random()
    output_size = random.randint(1,300)

    return {
        'hidden_size':hidden_size,
        'lstm_layer':lstm_layer, 
        'lr':learning_rate, 
        'fc':fc_hidden_size, 
        'dropout':dropout, 
        'output_size': output_size,
        'a_hs':attention_hidden_size, 
        'a_os':attention_output_size, 
        'a_p':attention_penalty
        }

def get_hyperparameter_set():
    '''
    Python set of all hyperparameters that are relevant during training
    '''

    hp_set = set()
    hp_set.add('hidden_size')
    hp_set.add('lstm_layer')
    hp_set.add('lr')
    hp_set.add('fc')
    hp_set.add('dropout')
    hp_set.add('a_hs')
    hp_set.add('a_os')
    hp_set.add('a_p')
    hp_set.add('output_size')

    return hp_set

def get_mutation(hp, value):
    '''
    Changes (mutates) the value of a hyperparamter (hp). Special rules apply to each type pf hyperparameter
    '''

    if hp == 'hidden_size':
        change = (random.randint(1, 50) - 25)
        value = value + change
        value = max(1, value) #no negative or 0 hidden size
        return value
    if hp == 'lstm_layer':
        change = random.choice([-1,1])
        value = value + change
        value = max(2, value) #at least 2 layers
        return value
    if hp == 'lr':
        change =(random.random()*1e-5 - 2e-5)
        value = value + change
        value = max(1e-6, value)
        return value
    if hp == 'fc':
        change = (random.randint(1, 50) - 25)
        value = value + change
        value = max(1, value) 
        return value
    if hp == 'dropout':
        change = (random.random()*0.2 - 0.1)
        value = value + change
        value = min(0.99, value) #not over 0.99
        value = max(1e-7, value) #keep at least some dropout
        return value
    if hp == 'a_hs':
        change = (random.randint(1, 50) - 25)
        value = value + change
        value = max(1, value) 
        return value
    if hp == 'a_os':
        change = (random.randint(1, 10) - 5)
        value = value + change
        value = max(1, value) 
        return value
    if hp == 'a_p':
        change = (random.random()*0.2 - 0.1)
        value = value + change
        value = min(1.0, value)
        value = max(0.0, value) 
        return value
    if hp == 'output_size':
        change = (random.randint(1, 50) - 25)
        value = value + change
        value = max(2, value) 
        return value


def genetic_hyperparam_search(data_loader, device,vocab_size, embedding_weights,config_dict, mutation_chance=0.25, num_candidates=8,batch_size=128, output_size=128,embedding_size=300,bidirectional=True, max_epochs=25, num_gens=10):
    """
    Finding best hyperparameters using genetic algorithm
    """

    logging.info("Finding best hyperparameters using genetic algorithm")
    models = list()

    for _ in range(num_candidates):
        c = init_candidate()
        self_attention_config = {
            "hidden_size": c['a_hs'],  ## refers to variable 'da' in the ICLR paper
            "output_size": c['a_os'],  ## refers to variable 'r' in the ICLR paper
            "penalty": c['a_p'],  ## refers to penalty coefficient term in the ICLR paper
        }
        model = SiameseBiLSTMAttention(
            batch_size=batch_size,
            output_size=c['output_size'],
            hidden_size=c['hidden_size'],
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            embedding_weights=embedding_weights,
            lstm_layers=c['lstm_layer'],
            self_attention_config=self_attention_config,
            fc_hidden_size=c['fc'],
            device=device,
            bidirectional=bidirectional,
            dropout=c['dropout']
        )
        model.to(device)

        models.append(
        {
            'm':model, 
            'lr':c['lr'], 
            'config':{
                "device": device,
                "model_name": "siamese_lstm_attention",
                "self_attention_config": self_attention_config,
            },
            'c':c
        }
        )

    for generation in range(num_gens):
        logging.info('starting generation {}'.format(generation))
        for m in models:
            try:
                logging.info("Starting train for model with parameters hidden_size={},lstm_layers={},fc={},dropout={},a_hs={},a_os={},p={}, lr={}, o_size={}".format(m['c']['hidden_size'],m['c']['lstm_layer'],m['c']['fc'],m['c']['dropout'],m['c']['a_hs'],m['c']['a_os'],m['c']['a_p'],m['lr'], m['c']['output_size']))
                optimizer = torch.optim.Adam(params=m['m'].parameters(), lr=m['lr'])
            
            except:
                print('ERERRRRRRRRRRRRROOOOOOOOOOOOOOOOORRRRR')
                print(m)
            
            best_val_acc = train_hyperparameters(m['m'], optimizer, data_loader, None, max_epochs, m['config'])
            m['val_acc'] = best_val_acc
        
        sorted_model_list = sorted(models, key=lambda acc: float(acc['val_acc']), reverse=True)
        models = sorted_model_list[:4]
        #print(sorted_model_list)

        children = crossing_over(sorted_model_list, num_candidates, mutation_chance, device, vocab_size, embedding_weights, batch_size, output_size,embedding_size,bidirectional)
        #print('here model len ins {} ad childrren len is {}'.format(len(models), len(children)))
        models = models + children
        #print('then model len is {}'.format(len(models)))

    best_model = sorted(models, key=lambda acc: float(acc['val_acc']), reverse=True)[0]
    return best_model

def crossing_over(parents,num_candidates, mutation_rate, device, vocab_size, embedding_weights, batch_size=64, output_size=1,embedding_size=300,bidirectional=True):
    '''
    Handles the step of producing child model from the best parent models, passing down their hyperparameters and randomly mutating them.
    '''

    children = list()
    ## Parent combinations that produce a child
    marriages = [(parents[0], parents[1]),(parents[0], parents[3]),(parents[1], parents[2]),(parents[2], parents[3])]
    for (p1,p2) in marriages:
        hyperparams = get_hyperparameter_set()
        child_params = dict.fromkeys(hyperparams, 0)

        ## iterate over every relevant hyperparameter
        for hp in list(hyperparams):

            ## choose from which parent to take the hyperparameter value
            chance = random.random()
            param = p2['c'][hp]
            if(chance > 0.5):
                param = p1['c'][hp]
            
            ## randomly mutate some hyperparameter values, depending on mutation rate
            mutation = random.random()
            if(mutation > (1.0-mutation_rate)):
                param = get_mutation(hp, param)
                ## negative values are not supported
                param = max(0, param)

            ## save new parameter value
            child_params[hp] = param
        

        ## Create dictionary object of new child
        self_attention_config = {
            "hidden_size": child_params['a_hs'],  ## refers to variable 'da' in the ICLR paper
            "output_size": child_params['a_os'],  ## refers to variable 'r' in the ICLR paper
            "penalty": child_params['a_p'],  ## refers to penalty coefficient term in the ICLR paper
        }
        child = SiameseBiLSTMAttention(
            batch_size=batch_size,
            output_size=child_params['output_size'],
            hidden_size=child_params['hidden_size'],
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            embedding_weights=embedding_weights,
            lstm_layers=child_params['lstm_layer'],
            self_attention_config=self_attention_config,
            fc_hidden_size=child_params['fc'],
            device=device,
            bidirectional=bidirectional,
            dropout=child_params['dropout']
        )
        child.to(device)
        children.append(
            {
                'm':child, 
                'lr':child_params['lr'], 
                'config':{
                    "device": device,
                    "model_name": "siamese_lstm_attention",
                    "self_attention_config": self_attention_config,
                },
                'c':child_params
            }
        )

    return children
        

def evaluate_dev_set(model, data, criterion, data_loader, config_dict, device):
    """
    Evaluates the model performance on dev data
    """
    ## eval mode
    model.eval()

    ## keep track of predictions and gold labels
    true_scores = list()
    predicted_scores = list()
    total_loss = 0
    with torch.no_grad():
        for (
            sent1,
            sent2,
            sents1_len,
            sents2_len,
            targets,
            _,
            _,
        ) in data_loader["validation"]:
            
            ## perform forward pass
            pred,sent1_annotation_weight_matrix,sent2_annotation_weight_matrix = model(sent1.to(device), sent2.to(device))

            ## calculate loss penalty
            sent1_attention_loss = attention_penalty_loss(
                sent1_annotation_weight_matrix,
                config_dict["self_attention_config"]["penalty"],
                device,
            )
            sent2_attention_loss = attention_penalty_loss(
                sent2_annotation_weight_matrix,
                config_dict["self_attention_config"]["penalty"],
                device,
            )

            ## compute loss
            loss = (
                criterion(pred.to(device), targets.float().to(device))
                + sent1_attention_loss
                + sent2_attention_loss
            )

            ## save predicted scores and true scores
            true_scores += list(targets.float())
            predicted_scores += list(pred.data.float().detach().cpu().numpy())
            total_loss += loss

    ## compute Pearson correlation
    acc, p = stats.pearsonr(true_scores, predicted_scores)
    return acc, total_loss.data.float()/len(data_loader['validation'])


def attention_penalty_loss(annotation_weight_matrix, penalty_coef, device):
    """
    This function computes the loss from annotation/attention matrix
    to reduce redundancy in annotation matrix and for attention
    to focus on different parts of the sequence corresponding to the
    penalty term 'P' in the ICLR paper
    ----------------------------------
    'annotation_weight_matrix' refers to matrix 'A' in the ICLR paper
    annotation_weight_matrix shape: (batch_size, attention_out, seq_len)
    """
    batch_size, attention_out_size = annotation_weight_matrix.size(0), annotation_weight_matrix.size(1)

    ## Transpose seq_len and attention_out
    annotation_weight_matrix_trans = annotation_weight_matrix.transpose(1, 2)

    ## calculate AA.T
    annotation_mul = torch.bmm(annotation_weight_matrix, annotation_weight_matrix_trans)

    ## make identity matrix of correct shape like attention matrix
    identity = torch.eye(attention_out_size, out=torch.empty_like(annotation_mul))

    ## claculate difference
    annotation_mul_difference = annotation_mul - identity

    ## use pytorch to calculate frobenius norm and square it like shown in paper
    penalty = torch.norm(annotation_mul_difference, p='fro')**2

    ## return penalty loss
    return (penalty_coef * penalty / batch_size).type(torch.FloatTensor)


def frobenius_norm(annotation_mul_difference):
    """
    Computes the frobenius norm of the annotation_mul_difference input as matrix
    """
    pass
