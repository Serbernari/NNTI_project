import random
from sklearn.ensemble import GradientBoostingClassifier
import torch
from torch import embedding, nn
from sklearn.metrics import r2_score, mean_squared_error
import logging
from tqdm import tqdm
from torch.autograd import Variable
#import transformers
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
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
        model.train()
        total_loss = 0.0
        for (
            sent1,
            sent2,
            _,
            _,
            targets,
            _,
            _,
        ) in dataloader["train"]:
            
            model.zero_grad()

            ## forward pass
            pred = model(sent1.to(device),sent2.to(device)
            )

            ## loss, no more self attention penalty
            loss = criterion(pred.to(device),targets.float().to(device),)

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

            ## keep track of loss over batches
            total_loss += loss

        ## computing accuracy using spearman correlation
        acc = stats.spearmanr(true_scores, predicted_scores).correlation
        mse = mean_squared_error(true_scores, predicted_scores)

        ## compute model metrics on dev set
        val_acc, val_loss, val_mse = evaluate_dev_set(
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
            "Train loss: {} - acc: {} - mse: {} -- Validation loss: {} - acc: {} - mse: {}".format(
                torch.mean(total_loss.data.float()/len(dataloader["train"])), acc, mse, val_loss, val_acc, val_mse
            )
        )

        ## save losses and accuracy for visualization
        train_losses.append( torch.mean(total_loss.data.float()/len(dataloader["train"])).item())
        train_accs.append(acc.item())
        val_losses.append(val_loss.item())
        val_accs.append(val_acc)
    return best_model, train_losses, train_accs, val_losses, val_accs

def train_hyperparameters(model, optimizer, dataloader, data, max_epochs, config_dict):
    '''
    training loop used when trying to find best hyperparameter combination.
    Does not log as much info or keep track of as many lists
    '''

    device = config_dict["device"]
    criterion = nn.MSELoss()  
    max_accuracy = 0.0
    best_val_loss =9999999

    for epoch in tqdm(range(max_epochs)):
        total_loss = 0
        for (
            sent1,
            sent2,
            _,
            _,
            targets,
            _,
            _,
        ) in dataloader["train"]:
            model.train()
            model.zero_grad()

            ## forward pass
            pred = model(sent1.to(device),sent2.to(device))

            ## compute loss, no more self attention penalty
            loss = ( criterion(pred.to(device), targets.float().to(device)))

            ## backward pass
            loss.backward()

            # clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            ## update gradients
            optimizer.step()

            ## accumulate train loss
            total_loss += loss

        ## compute model metrics on dev set
        val_acc, val_loss, val_mse = evaluate_dev_set(
            model, data, criterion, dataloader, config_dict, device
        )

        if val_acc > max_accuracy:
            max_accuracy = val_acc

    logging.info(val_acc)
    return val_acc

def init_candidate():
    '''
    randomly generate hyperparameter combination for creating a model
    '''

    learning_rates = [1e-3,2e-3,5e-4,8e-5]
    n_heads = [6, 12, 15, 20, 30]

    learning_rate = random.choice(learning_rates)
    dropout = random.random()*0.5
    encoder_layers = random.randint(1,8)
    embedding_size = 300
    n_heads = random.choice(n_heads)
    return {

        'lr':learning_rate, 
        'dropout':dropout, 
        'e_layers': encoder_layers,
        "e_heads": n_heads,
        'embedding_size': embedding_size
        }

def get_hyperparameter_set():
    '''
    Returns python set of all hyperparameters that are relevant during training
    '''
    hp_set = set()
    hp_set.add('lr')
    hp_set.add('dropout')
    hp_set.add('e_layers')
    hp_set.add('e_heads')
    hp_set.add('embedding_size')
    return hp_set

def get_mutation(hp, value):
    '''
    Changes (mutates) the value of a hyperparamter (hp). Special rules apply to each type pf hyperparameter
    '''
    if hp == 'e_layers':
        change = random.choice([-1,1])
        value = value + change
        value = max(1, value) #at least 1 layer
        return value

    if hp == 'e_heads':
        ## changing heads is difficult, because it must be evenly divisble into embedding_size
        return value
    if hp == 'lr':
        change =(random.random()*3e-5 - 2e-5)
        value = value + change
        value = max(1e-6, value)
        return value
    if hp == 'dropout':
        change = (random.random()*0.2 - 0.1)
        value = value + change
        value = min(0.99, value) #not over 0.99
        value = max(1e-7, value) #keep at least some dropout
        return value
    if hp == 'embedding_size':
        ## embedding size change no longer considered - always set at 300 due to fastext embedding
        return value


def genetic_hyperparam_search(data_loader, device,vocab_size, mutation_chance=0.33, num_candidates=8,batch_size=64, max_epochs=25, num_gens=10, pad_index=0, embedding_weights=None):
    """
    inding best hyperparameters using genetic algorithm
    """
    logging.info("Finding best hyperparameters using genetic algorithm")
    models = list()
    best_parents = list()

    ## make initial random hyperparam configuration for 8 candidates    
    for _ in range(num_candidates):
        
        candidate = init_candidate()

        models.append(
            {
                'c':candidate,
                'config_dict':{
                    "device": device,
                    "model_name": "siamese_lstm_attention"
                },
            }
        )

    for generation in range(num_gens):
        logging.info('starting generation {}'.format(generation))

        ## For every hyperparameter configuration, make a model and train it. While training a model, we keep the one with the best validation accuracy.
        for hyperparameters in models:
            logging.info("Starting train for model with parameters dropout={}, lr={}, n_layers={}, n_heads={}, embedding_size={}".format(hyperparameters['c']['dropout'],hyperparameters['c']['lr'], hyperparameters['c']['e_layers'],hyperparameters['c']['e_heads'],hyperparameters['c']['embedding_size']))
            ## Make a new model from the hyperparameters and train it
            model = make_model(hyperparameters, batch_size=batch_size, vocab_size=vocab_size, device=device, pad_index=pad_index, embedding_weights=embedding_weights)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=hyperparameters['c']['lr'])
            best_val_acc = train_hyperparameters(model, optimizer, data_loader, None, max_epochs, hyperparameters['config_dict'])
            hyperparameters['val_acc'] = best_val_acc # Save best performance during training

        ## add the parents back to get best overall models
        models = models + best_parents
        ## Sort the model list by validation accuracy. The 4 best models will be used as parents to create 4 child models
        sorted_model_list = sorted(models, key=lambda acc: float(acc['val_acc']), reverse=True)
        ## and log list to console
        for model in sorted_model_list:
            logging.info("model with score {} and parameters dropout={}, lr={}, e_layers={}, embedding_size={}".format(model['val_acc'],model['c']['dropout'],model['c']['lr'], model['c']['e_layers'],model['c']['embedding_size']))        
        ## Only keep 4 best performing models
        best_parents = sorted_model_list[:4]
        ## Make children from best models by randomly combining hyperparameters
        children = crossing_over(best_parents, mutation_chance, device)
        ## Save children and start training next generation
        models = children

    models = models + best_parents
    for model in models:
        logging.info("Final model list: dropout={}, lr={}, e_layers={}, embedding_size={}".format(model['c']['dropout'],model['c']['lr'], model['c']['e_layers'],model['c']['embedding_size']))

    return models[0]

def make_model(hyperparameters, batch_size,vocab_size, device, pad_index=0, embedding_weights=None):
    '''
    returns a SiameseBiLSTMAttention object created with passed hyperparameters
    '''
    hyperparam_dict = hyperparameters['c']

    attention_encoder_config = {
        "n_layers": hyperparam_dict['e_layers'],  ## number of encoder layers
        "n_heads": hyperparam_dict['e_heads'],  ## heads in multi-head attention
        "expansion": 4, ## encoder feed forward embedding size expansion factor
        "vocab_max": 40 ## max sequence length
    }

    model = SiameseBiLSTMAttention(
        batch_size=batch_size,
        vocab_size=vocab_size,
        embedding_size=hyperparam_dict['embedding_size'],
        embedding_weights=embedding_weights,
        attention_encoder_config=attention_encoder_config,
        device=device,
        pad_index=pad_index,
        dropout=hyperparam_dict['dropout']
    )
    
    model.to(device)
    return model

def crossing_over(parents,mutation_rate, device):
    '''
    Creates the next generation of models by randomly combining the hyperparameters of the 4 best parent models
    '''
    children = list()
    marriages = [(parents[0], parents[1]),(parents[0], parents[3]),(parents[1], parents[2]),(parents[2], parents[3])] ## only supports exactly 8 models per generation
    for (p1,p2) in marriages:
        hyperparams = get_hyperparameter_set()
        child_params = dict.fromkeys(hyperparams, 0)
        ## Create child models by combining the parent hyperparameters randomly
        for hp in list(hyperparams):
            chance = random.random()
            param = p2['c'][hp]
            if(chance > 0.5):
                param = p1['c'][hp]

            ## With a certain chance, a hyperparameter value will 'muatate', randomly changing its value
            mutation = random.random()
            if(mutation > (1.0-mutation_rate)):
                param = get_mutation(hp, param)

            ## Save hyperparameter value to child config
            child_params[hp] = param
        
        children.append(
            {
                'config_dict':{
                    "device": device,
                    "model_name": "siamese_lstm_attention"
                },
                'c':child_params
            }
        )

    return children
        

def evaluate_dev_set(model, data, criterion, data_loader, config_dict, device):
    """
    Evaluates the model performance on dev data
    """
    model.eval()
    true_scores = list()
    predicted_scores = list()
    total_loss = 0.0
    with torch.no_grad():
        for (
            sent1,
            sent2,
            _,
            _,
            targets,
            _,
            _,
        ) in data_loader["validation"]:

            ## forward pass
            pred = model(
                sent1.to(device),
                sent2.to(device)
            )

            ## loss
            loss = criterion(pred.to(device),targets.float().to(device))           
            total_loss += loss

            ## keep track of predictions and gold labels
            true_scores += list(targets.float())
            predicted_scores += list(pred.data.float().detach().cpu().numpy())
            
    ## computing accuracy using Spearman rho
    acc = stats.spearmanr(true_scores, predicted_scores).correlation
    mse = mean_squared_error(true_scores, predicted_scores)

    return acc, torch.mean(total_loss.data.float()/len(data_loader["validation"])), mse 
