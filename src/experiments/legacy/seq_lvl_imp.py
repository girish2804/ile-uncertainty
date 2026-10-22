# from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers.models import Pooling
from sentence_transformers import models
from sentence_transformers.similarity_functions import SimilarityFunction
import pdb
# from info_nce import InfoNCE, info_nce

sentence_grads = [None]
sentence_embeddings = [None]

# sim_model = SentenceTransformer()

def sent_forward_hook(module, input, output):
    sentence_embedding = output['sentence_embedding'] if (isinstance(output, dict) or hasattr(output, '__getitem__')) and not isinstance(output, torch.Tensor) else output
    sentence_embeddings[0] = sentence_embedding.detach()
    sentence_embedding.register_hook(lambda grad: sentence_grads.__setitem__(0, grad))
 

class CustomSentenceEmbedderWithGrad(torch.nn.Module):
    
    def __init__(self, model_name = 'sentence-transformers/sentence-t5-large'):
        super().__init__()
        try:
            self.model = models.Transformer(model_name)
        except Exception:
            from sentence_transformers import SentenceTransformer
            st = SentenceTransformer(model_name)
            self.model = st[0]

        try:
            self.sim_fn = SimilarityFunction.to_similarity_fn("cosine")
        except Exception:
            from sentence_transformers.util import cos_sim
            self.sim_fn = cos_sim

        if hasattr(self.model, 'get_embedding_dimension'):
            hidden_dim = self.model.get_embedding_dimension()
        elif hasattr(self.model, 'get_word_embedding_dimension'):
            hidden_dim = self.model.get_word_embedding_dimension()
        elif hasattr(self.model, 'auto_model') and hasattr(self.model.auto_model, 'config'):
            hidden_dim = getattr(self.model.auto_model.config, 'hidden_size', 768)
        else:
            hidden_dim = 1024

        try:
            self.pooling = Pooling(embedding_dimension=hidden_dim)
        except TypeError:
            self.pooling = Pooling(word_embedding_dimension=hidden_dim)

    def forward(self, sentences):
        # Determine device dynamically from model parameters
        device = next(self.parameters()).device
        inputs = self.model.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = self.model.forward(inputs)
        features = self.pooling(outputs)
        if isinstance(features, torch.Tensor):
            return features
        if hasattr(features, '__getitem__') and 'sentence_embedding' in features:
            return features['sentence_embedding']
        return features
        


_embedder_cache = {}

# input_seq is a single string, additional_seq is list of string
def sentence_importance(question, input_seq, additional_seq, model, semantic_pair_matrix):
    global _embedder_cache
    if model not in _embedder_cache:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _embedder_cache[model] = CustomSentenceEmbedderWithGrad(model_name=model).to(device)
    embedder = _embedder_cache[model]
    device = next(embedder.parameters()).device
    
    input_ = [input_seq] + additional_seq
    if len(input_) <= 1:
        return torch.ones(len(input_), device=device)
        
    # Zero gradients to prevent accumulation
    embedder.zero_grad()
    
    sentence_grads[0] = None
    sentence_embeddings[0] = None
    handle = embedder.pooling.register_forward_hook(sent_forward_hook)
    
    try:
        embeddings = embedder.forward(input_)
        if not isinstance(embeddings, torch.Tensor):
            if hasattr(embeddings, '__getitem__') and 'sentence_embedding' in embeddings:
                embeddings = embeddings['sentence_embedding']
            elif hasattr(embeddings, 'sentence_embedding'):
                embeddings = embeddings.sentence_embedding
        similarities = embedder.sim_fn(embeddings, embeddings)
        target_similarities = semantic_pair_matrix.to(device)
        
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(similarities, target_similarities)
        loss.backward()
    finally:
        handle.remove()
    
    with torch.no_grad():
        if sentence_grads[0] is not None and sentence_embeddings[0] is not None and sentence_grads[0].shape == sentence_embeddings[0].shape:
            sentence_attr = sentence_grads[0] * sentence_embeddings[0]
            sentence_attr = torch.norm(sentence_attr, dim=1)
            sentence_attr = F.normalize(sentence_attr, p=1, dim=0)
        else:
            sentence_attr = torch.ones(len(input_), device=device) / len(input_)
            
    # Clean up global state references
    sentence_grads[0] = None
    sentence_embeddings[0] = None
    
    return sentence_attr


# class MatrixContrastiveLoss(nn.Module):
#     def __init__(self, margin=0.3):
#         """
#         Args:
#             margin (float): The threshold for negative pairs. 
#                             If similarity < margin, loss is 0.
#                             Recommended: 0.3 to 0.5 for Cosine Similarity.
#         """
#         super().__init__()
#         self.margin = margin

#     def forward(self, similarities, targets):
#         """
#         Args:
#             similarities: [N, N] tensor of cosine similarities (-1 to 1)
#             targets: [N, N] binary tensor (1 for positive, 0 for negative)
#         """
#         # 1. Positive Pairs (Targets = 1)
#         # We want similarity -> 1.0
#         # Loss = (1 - sim)^2
#         # This gives strong gradients when sim is low, weak when sim is close to 1
#         pos_loss = targets * (1 - similarities)**2
        
#         # 2. Negative Pairs (Targets = 0)
#         # We want similarity < margin
#         # Loss = max(0, sim - margin)^2
#         # This creates the "sparsity": if sim is already low, gradient is 0.
#         neg_mask = (1 - targets)
#         hinge = torch.clamp(similarities - self.margin, min=0)
#         neg_loss = neg_mask * (hinge ** 2)
        
#         # 3. Aggregation
#         # Summing or averaging. Averaging separately prevents class imbalance 
#         # (since you have many more 0s than 1s) from dominating the gradient.
#         pos_term = pos_loss.sum() / (targets.sum() + 1e-9)
#         neg_term = neg_loss.sum() / (neg_mask.sum() + 1e-9)
        
#         return pos_term + neg_term