import torch
import yaml
from typing import Optional
import pickle
from torch.nn.utils import skip_init

from modules import Transformer, Softmax
from train import load_checkpoint
from cs336_basics.tokenizer import Tokenizer

def Sample(
    logits: torch.Tensor,
    temperature: float,
    p: Optional[float] = 1.0
) -> int:
    if temperature == 0.0:
        return logits.argmax().item()
    
    probs = Softmax(logits.squeeze(0) / temperature, -1)
    if p == 1.0:
        return torch.multinomial(probs, num_samples=1).item()
    sorted_probs, indices = torch.sort(probs, descending=True)
    x = 0
    top = 0
    for prob in sorted_probs:
        x += prob
        top += 1
        if x >= p:
            break
    return indices[torch.multinomial(sorted_probs[:top], num_samples=1)].item()


def decode(
    start: str,
    tokenizer: Tokenizer,
    config_path: str,
    checkpoint_path: str,
    stop_token: bytes
) -> str:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    stop_id = tokenizer.tok_to_id[stop_token]

    model = Transformer(device=config["device"], **config["model"])
    load_checkpoint(checkpoint_path, model)

    sequence = tokenizer.encode(start)
    while len(sequence) < config["decoding"]["max_tokens"]:
        output = model.forward(torch.tensor(sequence, device=config["device"]).unsqueeze(0))
        sampled_id = Sample(output[..., -1, :], config["decoding"]["temperature"], config["decoding"]["p"])
        sequence.append(sampled_id)
        if sampled_id == stop_id:
            break

    return tokenizer.decode(sequence)

if __name__ == "__main__":
    with open("tokenizer/tokenizer.pkl", "rb") as f:
        obj = pickle.load(f)
        vocab = obj["vocab"]
        merges = obj["merges"]
    t = Tokenizer(vocab, merges, ["<|endoftext|>"])
    output = decode("In theory, ", t, "cs336_basics/config.yaml", "checkpoints/checkpoint_2000", b"<|endoftext|>")
    print(output)