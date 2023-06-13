import dill
import argparse
import torch
from transformer import Transformer
from utils import *
from constants import *

from base.predictor import Predictor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/_base_.yaml")
    parser.add_argument("--src-field-path", type=str, default="./fields/src_field.pt")
    parser.add_argument("--trg-field-path", type=str, default="./fields/trg_field.pt")
    parser.add_argument("--ckpt_path", type=str, default="./weights/best.pt")
    parser.add_argument("--input", type=str, default="I am a student")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    config_dict = load_config(args.config_path)

    src_field = torch.load(args.src_field_path, pickle_module=dill)
    trg_field = torch.load(args.trg_field_path, pickle_module=dill)
    src_vocab_size = len(src_field.vocab)
    trg_vocab_size = len(trg_field.vocab)

    model = Transformer(config_path=args.config_path,
                        src_vocab_size=src_vocab_size,
                        trg_vocab_size=trg_vocab_size)
    
    model.load_state_dict(torch.load(args.ckpt_path))

    predictor = Predictor(model, src_field, trg_field, device=args.device)

    input = args.input
    output = predictor(input, max_len=config_dict['DATA']['MAX_LEN'], beam_size=1)
    print(output)