import torch
from torch.nn.functional import softmax, softplus

def calculate(actor_out):
    if actor_out.dim() == 1:
        is_1d = True
        actor_out = actor_out.unsqueeze(0)
    else:
        is_1d = False

    crypto_logits = actor_out[:, :-1]  # Shape: (B, num_cryptos)
    cash_logit = actor_out[:, -1]  # Shape: (B,)

    processed_cash_logit = softplus(cash_logit).unsqueeze(1)  # Shape: (B, 1)

    fiduciae = torch.zeros_like(actor_out)

    # --- 1. Long Book Calculation ---
    long_book_logits = torch.cat([crypto_logits, processed_cash_logit], dim=1)
    negative_crypto_mask = crypto_logits < 0
    long_book_logits[:, :-1][negative_crypto_mask] = -torch.inf
    long_book_probs = softmax(long_book_logits, dim=1)

    positive_crypto_mask = crypto_logits >= 0
    fiduciae[:, :-1][positive_crypto_mask] = long_book_probs[:, :-1][positive_crypto_mask]
    fiduciae[:, -1] = long_book_probs[:, -1]

    # --- 2. Short Book Calculation ---
    short_book_logits = torch.abs(crypto_logits)
    short_book_logits[positive_crypto_mask] = -torch.inf
    short_book_probs = softmax(short_book_logits, dim=1)
    short_book_probs = torch.nan_to_num(short_book_probs, nan=0.0)
    fiduciae[:, :-1][negative_crypto_mask] = -short_book_probs[negative_crypto_mask]

    if is_1d:
        fiduciae = fiduciae.squeeze(0)

    return fiduciae