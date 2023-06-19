import os
import dill
import argparse
import torch
from transformer import Transformer
from utils import *
from constants import *

from base.predictor import Predictor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_path", type=str, help="Path to training result folder (e.g. runs/...)")
    parser.add_argument("--input", type=str, default="I am a student")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    config_path = os.path.join(args.runs_path, 'config.yaml')
    ckpt_path = os.path.join(args.runs_path, 'best.pt')
    src_field_path = os.path.join(args.runs_path, 'src_field.pt')
    trg_field_path = os.path.join(args.runs_path, 'trg_field.pt')

    config_dict = load_config(config_path)
    src_field = torch.load(src_field_path, pickle_module=dill)
    trg_field = torch.load(trg_field_path, pickle_module=dill)
    src_vocab_size = len(src_field.vocab)
    trg_vocab_size = len(trg_field.vocab)

    model = Transformer(config_path=config_path,
                        src_vocab_size=src_vocab_size,
                        trg_vocab_size=trg_vocab_size)
    
    model.load_state_dict(torch.load(ckpt_path))

    predictor = Predictor(model, src_field, trg_field, device=args.device)

    input = args.input
    output = predictor(input, max_len=config_dict['DATA']['MAX_LEN'], beam_size=1)
    print(output)