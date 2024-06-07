from typing import Union, Iterable, Callable
import random

import torch.nn as nn
import torch


def load_datasets(data_directory: str) -> Union[dict, dict]:
    """
    Reads the training and validation splits from disk and load
    them into memory.

    Parameters
    ----------
    data_directory: str
        The directory where the data is stored.

    Returns
    -------
    train: dict
        The train dictionary with keys 'premise', 'hypothesis', 'label'.
    validation: dict
        The validation dictionary with keys 'premise', 'hypothesis', 'label'.
    """
    import json
    import os

    with open(os.path.join(data_directory, "train.json"), "r") as f:
        train = json.load(f)

    with open(os.path.join(data_directory, "validation.json"), "r") as f:
        valid = json.load(f)

    return train, valid


def tokenize(
    text: "list[str]", max_length: int = None, normalize: bool = True
) -> "list[list[str]]":
    """
    Tokenize the text into individual words (nested list of string),
    where the inner list represent a single example.

    Parameters
    ----------
    text: list of strings
        Your cleaned text data (either premise or hypothesis).
    max_length: int, optional
        The maximum length of the sequence. If None, it will be
        the maximum length of the dataset.
    normalize: bool, default True
        Whether to normalize the text before tokenizing (i.e. lower
        case, remove punctuations)
    Returns
    -------
    list of list of strings
        The same text data, but tokenized by space.

    Examples
    --------
    >>> tokenize(['Hello, world!', 'This is a test.'], normalize=True)
    [['hello', 'world'], ['this', 'is', 'a', 'test']]
    """
    import re

    if normalize:
        regexp = re.compile("[^a-zA-Z ]+")
        # Lowercase, Remove non-alphanum
        text = [regexp.sub("", t.lower()) for t in text]

    return [t.split()[:max_length] for t in text]


def build_word_counts(token_list: "list[list[str]]") -> "dict[str, int]":
    """
    This builds a dictionary that keeps track of how often each word appears
    in the dataset.

    Parameters
    ----------
    token_list: list of list of strings
        The list of tokens obtained from tokenize().

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.

    Notes
    -----
    If you have  multiple lists, you should concatenate them before using
    this function, e.g. generate_mapping(list1 + list2 + list3)
    """
    word_counts = {}

    for words in token_list:
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    return word_counts


def build_index_map(
    word_counts: "dict[str, int]", max_words: int = None
) -> "dict[str, int]":
    """
    Builds an index map that converts a word into an integer that can be
    accepted by our model.

    Parameters
    ----------
    word_counts: dict of {str: int}
        A dictionary mapping every word to an integer representing the
        appearance frequency.
    max_words: int, optional
        The maximum number of words to be included in the index map. By
        default, it is None, which means all words are taken into account.

    Returns
    -------
    dict of {str: int}
        A dictionary mapping every word to an integer representing the
        index in the embedding.
    """

    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    if max_words:
        sorted_counts = sorted_counts[:max_words-1]
    
    sorted_words = ["[PAD]"] + [item[0] for item in sorted_counts]

    return {word: ix for ix, word in enumerate(sorted_words)}


def tokens_to_ix(
    tokens: "list[list[str]]", index_map: "dict[str, int]"
) -> "list[list[int]]":
    """
    Converts a nested list of tokens to a nested list of indices using
    the index map.

    Parameters
    ----------
    tokens: list of list of strings
        The list of tokens obtained from tokenize().
    index_map: dict of {str: int}
        The index map from build_index_map().

    Returns
    -------
    list of list of int
        The same tokens, but converted into indices.

    Notes
    -----
    Words that have not been seen are ignored.
    """
    return [
        [index_map[word] for word in words if word in index_map] for words in tokens
    ]


### Batching, shuffling, iteration
def build_loader(
    data_dict: dict, batch_size: int = 64, shuffle: bool = False
) -> Callable[[], Iterable[dict]]:
    keys = list(data_dict.keys())
#     print(keys)
    num_samples = len(data_dict[keys[0]])
#     print(num_samples)
    
    def loader():
        if shuffle:
            indices = random.sample(range(num_samples), num_samples)
        else:
            indices = list(range(num_samples))

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            batch_data = {key: [data_dict[key][i] for i in batch_indices] for key in keys}
            
            yield batch_data

    return loader


### Converting a batch into inputs
def convert_to_tensors(text_indices: "list[list[int]]") -> torch.Tensor:
    # taking the max length of the inner lists => L
    L = max(len(inner_list) for inner_list in text_indices)
    # then we pad the indices per the slides: for each inner list we add as many zeros as we need to reach length of L
    padded_indices = [inner_list + [0] * (L - len(inner_list)) for inner_list in text_indices]
    return torch.tensor(padded_indices, dtype=torch.int32)


### Design a logistic model with embedding and pooling
def max_pool(x: torch.Tensor) -> torch.Tensor:
    return torch.max(x, dim=1)[0]


class PooledLogisticRegression(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding
        self.layer_pred = nn.Linear(embedding.embedding_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        
        # get embeddings for each of premise and hypothesis, dim = (N,L,E) L=Lp or Lh
        premise_embedded = emb(premise)
        hypothesis_embedded = emb(hypothesis)
        
        # max_pool each of them, get (N,E)
        premise_pooled = max_pool(premise_embedded)
        hypothesis_pooled = max_pool(hypothesis_embedded)
        
        # concatenate the premise and hypothesis, dim=(N,2E) => have to pick dim=1 which appends to right of premise i.e. extra columns
        concat = torch.cat([premise_pooled, hypothesis_pooled], dim=1)
        
        # Apply logistic regression
        logits = layer_pred(concat)

        # Apply sigmoid and also change dim from (n,1) to n
        predictions = sigmoid(logits).squeeze(dim=1)
        
        return predictions
        


### Choose an optimizer and a loss function
def assign_optimizer(model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), **kwargs)
    return optimizer


def bce_loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # Avoids division by zero
    epsilon = 1e-10  
    # FORMULA: Lce = -(y*log(y_hat) + (1-y)*log(1-y_hat))
    loss = - (y * torch.log(y_pred + epsilon) + (1 - y) * torch.log(1 - y_pred + epsilon))

    return torch.mean(loss)


### Forward and backward pass
def forward_pass(model: nn.Module, batch: dict, device="cpu"):
    # Use ".to(device)" to make sure they're on the same device we run the model in case we do cuda
    premise_tensor = convert_to_tensors(batch["premise"]).to(device)
    hypothesis_tensor = convert_to_tensors(batch["hypothesis"]).to(device)
    
    y_pred = model.forward(premise_tensor, hypothesis_tensor)
    return y_pred


def backward_pass(
    optimizer: torch.optim.Optimizer, y: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    
    optimizer.zero_grad()
    loss = bce_loss(y, y_pred)
    
    loss.backward()
    # Update the weights after the backprop, this makes use of the optimizer chosen after grads computed
    optimizer.step()
    return loss



### Evaluation
def f1_score(y: torch.Tensor, y_pred: torch.Tensor, threshold=0.5) -> torch.Tensor:
    
    y_pred_binary = (y_pred >= threshold).float()
    
    TP = torch.sum((y == 1) & (y_pred_binary == 1)).float()
    FP = torch.sum((y == 0) & (y_pred_binary == 1)).float()
    FN = torch.sum((y == 1) & (y_pred_binary == 0)).float()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return f1


### Train loop
def eval_run(
    model: nn.Module, loader: Callable[[], Iterable[dict]], device: str = "cpu"
):  
    #PyTorch: this is how you turn model into evaluation mode and disable gradient computation...
    model.eval()
    with torch.no_grad():
        # saving the y and y_pred for each sample input (to be evaluated)
        y_true = []
        y_pred = []
        for batch in loader():
            y = torch.tensor(batch["label"], dtype=torch.float32).to(device)
            batch_pred = forward_pass(model, batch).to(device)
            # .cpu() is just saving them to disk
            y_true.extend(y.cpu().tolist())
            y_pred.extend(batch_pred.cpu().tolist())
    return torch.tensor(y_true), torch.tensor(y_pred)


def train_loop(
    model: nn.Module,
    train_loader,
    valid_loader,
    optimizer,
    n_epochs: int = 3,
    device: str = "cpu",
):
    
    # saving the f1 scores to look at after
    scores = []
    for epoch in range(n_epochs):
        #PyTorch practice: set model on train mode when training just to be sure you're actually doing grads
        model.train()
        for train_batch in train_loader():
            y = torch.tensor(train_batch["label"], dtype=torch.float32).to(device)
            y_pred = forward_pass(model, train_batch, device)
            loss = backward_pass(optimizer, y, y_pred)
            
        # get the f1 score and append to the list scores
        y_true, y_pred = eval_run(model, valid_loader, device)
        f1 = f1_score(y_true, y_pred)
        scores.append(f1)
        print(f"Epoch {epoch + 1}/{n_epochs}, F1 Score: {f1}")
        
    return scores


### Shallow Neural Network Implementation
class ShallowNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int):
        super().__init__()
        # same as before we have a Linear model with in_features = 2E but now out_features=H since we have a hidden layer (N,H)
        self.ff_layer = nn.Linear(embedding.embedding_dim * 2, hidden_size)
        # hidden layer -> layer_pred so in_features=H and out_features=1 bc (N,1)
        self.layer_pred = nn.Linear(hidden_size, 1)
        self.embedding = embedding
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()

    def get_ff_layer(self):
        return self.ff_layer

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation


    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layer = self.get_ff_layer()
        act = self.get_activation()
        
        # once again we embed each premise and hypothesis (N,L,E)
        premise_embedded = emb(premise)
        hypothesis_embedded = emb(hypothesis)
        
        # then we call max pool => (N,E)
        premise_pooled = max_pool(premise_embedded)
        hypothesis_pooled = max_pool(hypothesis_embedded)
        
        # and concatenated the max pooled => (N,2E)
        concatenated = torch.cat([premise_pooled, hypothesis_pooled], dim=1)
        
        # then we feed it through Linear and apply relu
        hidden_layer = act(ff_layer(concatenated))
        
        # Then we do Linear + Sigmoid
        logits = layer_pred(hidden_layer)
        preds = sigmoid(logits).squeeze(dim=1)
        
        return preds


### Deep Neural Nework Implementation
class DeepNeuralNetwork(nn.Module):
    def __init__(self, embedding: nn.Embedding, hidden_size: int, num_layers: int = 2):
        super().__init__()

        self.ff_layers = nn.ModuleList([
            nn.Linear(embedding.embedding_dim * 2 if i == 0 else hidden_size, hidden_size) for i in range(num_layers)
        ])
        # hidden layer -> layer_pred so in_features=H and out_features=1 bc (N,1)
        self.layer_pred = nn.Linear(hidden_size, 1)
        self.embedding = embedding
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.ReLU()

    def get_ff_layers(self):
        return self.ff_layers

    def get_layer_pred(self):
        return self.layer_pred

    def get_embedding(self):
        return self.embedding

    def get_sigmoid(self):
        return self.sigmoid

    def get_activation(self):
        return self.activation

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor) -> torch.Tensor:
        emb = self.get_embedding()
        layer_pred = self.get_layer_pred()
        sigmoid = self.get_sigmoid()
        ff_layers = self.get_ff_layers()
        act = self.get_activation()

        premise_embedded = emb(premise)
        hypothesis_embedded = emb(hypothesis)
        
        # then we call max pool => (N,E)
        premise_pooled = max_pool(premise_embedded)
        hypothesis_pooled = max_pool(hypothesis_embedded)
        
        # and concatenated the max pooled => (N,2E)
        concatenated = torch.cat([premise_pooled, hypothesis_pooled], dim=1)
        
        for layer in ff_layers:
            # then we feed it through Linear and apply relu
            concatenated = act(layer(concatenated))
        
        # Then we do Linear + Sigmoid
        logits = layer_pred(concatenated)
            
        preds = sigmoid(logits).squeeze(dim=1)
        
        return preds
    


if __name__ == "__main__":
    # If you have any code to test or train the model, do it below!

    # Seeds to ensure reproducibility
    random.seed(2022)
    torch.manual_seed(2022)

    # If you use GPUs, use the code below:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prefilled code showing you how to use the helper functions
    train_raw, valid_raw = load_datasets("data")
    #For kaggle: ../input/a1-data/data

    train_tokens = {
        "premise": tokenize(train_raw["premise"], max_length=64),
        "hypothesis": tokenize(train_raw["hypothesis"], max_length=64),
    }

    valid_tokens = {
        "premise": tokenize(valid_raw["premise"], max_length=64),
        "hypothesis": tokenize(valid_raw["hypothesis"], max_length=64),
    }

    word_counts = build_word_counts(
        train_tokens["premise"]
        + train_tokens["hypothesis"]
        + valid_tokens["premise"]
        + valid_tokens["hypothesis"]
    )
    index_map = build_index_map(word_counts, max_words=10000)
#     print("index map")
#     print(index_map)
    
    train_indices = {
        "label": train_raw["label"],
        "premise": tokens_to_ix(train_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(train_tokens["hypothesis"], index_map)
    }

    valid_indices = {
        "label": valid_raw["label"],
        "premise": tokens_to_ix(valid_tokens["premise"], index_map),
        "hypothesis": tokens_to_ix(valid_tokens["hypothesis"], index_map)
    }
    
#     print(train_indices)

    # 1.1
    # Train set to be shuffled and Valid set to be non-shuffled
    train_loader = build_loader(train_indices, batch_size=64, shuffle=True)
    valid_loader = build_loader(valid_indices, batch_size=64, shuffle=False)

    # 1.2
    batch = next(train_loader())
    #print("batch: ", batch)
    y = torch.tensor(batch["label"], dtype=torch.float32) 

    # 2.1
    embedding = nn.Embedding(len(index_map), embedding_dim=50) 
    print("embedding: ", embedding)
    model = PooledLogisticRegression(embedding) 

    # 2.2
    optimizer = assign_optimizer(model, lr=0.01) 

    # 2.3
    y_pred = forward_pass(model, batch) 
    loss = backward_pass(optimizer, y, y_pred)

    # 2.4
    score =  f1_score(y, y_pred).item() 

    # 2.5
    n_epochs = 2

    embedding =  nn.Embedding(len(index_map), embedding_dim=50) 
    model = PooledLogisticRegression(embedding)  
    optimizer = assign_optimizer(model, lr=0.01) 

    scores = train_loop(model,
                        train_loader,
                        valid_loader,
                        optimizer,
                        n_epochs = n_epochs,
                        device=device) 

    # 3.1 Train ShallowNeuralNetwork
    embedding = nn.Embedding(len(index_map), embedding_dim=50) 
    model = ShallowNeuralNetwork(embedding, hidden_size=50)
    optimizer = assign_optimizer(model, lr=0.01)

    scores =  train_loop(model,
                    train_loader,
                    valid_loader,
                    optimizer,
                    n_epochs=n_epochs,
                    device=device)

    # 3.2 Train DeepNeuralNetwork
    embedding = nn.Embedding(len(index_map), embedding_dim=50)
    model = DeepNeuralNetwork(embedding, hidden_size=50, num_layers=3) 
    optimizer = assign_optimizer(model, lr=0.01) 
    
    scores = train_loop(model,
                    train_loader,
                    valid_loader,
                    optimizer,
                    n_epochs=n_epochs,
                    device=device) 
