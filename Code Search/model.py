import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
class CodeSearchModel(nn.Module):
    """
    GraphCodeBERT wrapper for code search fine-tuning.
    Uses mean pooling instead of CLS for better retrieval performance.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, attention_mask=None, nl_inputs=None):
        if code_inputs is not None:
            outputs = self.encoder(
                input_ids=code_inputs,
                attention_mask=attention_mask,
                return_dict=True
            )
            emb = mean_pooling(outputs.last_hidden_state, attention_mask)
            return F.normalize(emb, p=2, dim=1)  # add this
        else:
            outputs = self.encoder(
                input_ids=nl_inputs,
                attention_mask=attention_mask,
                return_dict=True
            )
            emb = mean_pooling(outputs.last_hidden_state, attention_mask)
            return F.normalize(emb, p=2, dim=1)  # add this