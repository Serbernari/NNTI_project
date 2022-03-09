from transformers import RobertaTokenizer, RobertaModel
import torch

def get_embeddings(sentences: list, batch_size = 4) -> torch.Tensor:
  """
  Gets list of sentences as input
  Example: sentences = ["I'm a first sentence", "I'm a second one"]

  returns list of embeddings with length of 769 
  
  Embeddings of tokens averaged w.r.t. their attention scores
  """
  tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
  model = RobertaModel.from_pretrained("roberta-base")
  model.eval()

  for idx in range(0, len(sentences), batch_size):
    batch = sentences[idx : min(len(sentences), idx+batch_size)]
    
    # encoded = tokenizer(batch)
    encoded = tokenizer.batch_encode_plus(batch)
  
    encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
    with torch.no_grad():        
        outputs = model(**encoded)

    lhs = outputs.last_hidden_state
    attention = encoded['attention_mask'].reshape((lhs.size()[0], lhs.size()[1], -1)).expand(-1, -1, 768)
    embeddings = torch.mul(lhs, attention)
    denominator = torch.count_nonzero(embeddings, dim=1)
    summation = torch.sum(embeddings, dim=1)
    mean_embeddings = torch.div(summation, denominator)

    return mean_embeddings
  
