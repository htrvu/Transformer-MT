import os, datetime, torch, dill, json
import argparse
from utils import load_config
from datasets.wrapper import TextDataWrapper
from transformer.model import Transformer
from base.metrics.bleu import calc_bleu
from base.predictor import Predictor
from constants import *

if __name__ == "__main__":
    """
    This program is used to evaluate the model.

    Usage
        python eval.py \
            --runs_path ./runs \
            --run-train-set \   # (or not)
            --train_src_path ./data/train.en \
            --train_trg_path ./data/train.vi \
            --valid_src_path ./data/val.en \
            --valid_trg_path ./data/val.vi \
            --test_src_path ./data/test.en \
            --test_trg_path ./data/test.vi \
            --device cuda:0
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs_path",
        type=str,
        help="Path to training result folder (e.g. ./runs/...)",
        required=True,
    )
    parser.add_argument("--run_train_set", action="store_true", default=False)
    parser.add_argument("--train_src_path", type=str, default="./data/train.en")
    parser.add_argument("--train_trg_path", type=str, default="./data/train.vi")
    parser.add_argument("--valid_src_path", type=str, default="./data/val.en")
    parser.add_argument("--valid_trg_path", type=str, default="./data/val.vi")
    parser.add_argument("--test_src_path", type=str, default="./data/test.en")
    parser.add_argument("--test_trg_path", type=str, default="./data/test.vi")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./eval_runs",
        help="directory to save checkpoints",
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    try:
        # Prepare evaluation result folder
        if args.out_dir == "./eval_runs":
            now = datetime.datetime.now()
            c = now.strftime("%Y-%m-%d_%H-%M-%S")
            args.out_dir = os.path.join(args.out_dir, c)

        # Load config, model, vocab field
        config_path = os.path.join(args.runs_path, "config.yaml")
        ckpt_path = os.path.join(args.runs_path, "best.pt")
        src_field_path = os.path.join(args.runs_path, "src_field.pt")
        trg_field_path = os.path.join(args.runs_path, "trg_field.pt")

        config_dict = load_config(config_path)
        src_field = torch.load(src_field_path, pickle_module=dill)
        trg_field = torch.load(trg_field_path, pickle_module=dill)
        src_vocab_size = len(src_field.vocab)
        trg_vocab_size = len(trg_field.vocab)

        # Load model
        model = Transformer(
            config_path=config_path,
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
        )
        model.load_state_dict(torch.load(ckpt_path))

        # Load dataset
        data_wrapper = TextDataWrapper(
            src_lang="en_core_web_sm",
            trg_lang="vi_core_news_lg",
            max_len=config_dict["DATA"]["MAX_LEN"],
            batch_size=config_dict["DATA"]["BATCH_SIZE"],
            device=args.device,
        )

        src_paths = []
        trg_paths = []
        if args.run_train_set:
            src_paths.append(args.train_src_path)
            trg_paths.append(args.train_trg_path)
        src_paths.extend([args.valid_src_path, args.test_src_path])
        trg_paths.extend([args.valid_trg_path, args.test_trg_path])

        # Create predictor
        predictor = Predictor(model, src_field, trg_field, device=args.device)

        # Predictor variables
        max_len = config_dict["DATA"]["MAX_LEN"]
        beam_size = 5

        results = []

        # Run on each set
        for src_path, trg_path in zip(src_paths, trg_paths):
            if not os.path.exists(src_path) or not os.path.exists(trg_path):
                print(f"File not found: {src_path} or {trg_path}")
                continue

            cur_dataloader = data_wrapper.create_dataloader(src_path, trg_path, is_train=False)

            # Load pairs of input and output sentence
            src_sentences = [
                " ".join(x.src) for x in cur_dataloader.dataset.examples
            ][1:]
            trg_sentences = [
                " ".join(x.trg) for x in cur_dataloader.dataset.examples
            ][1:]
            print("Start evaluating on set:", src_path, trg_path)
            preds = []
            for sentence in src_sentences:
                preds.append(
                    predictor(sentence, max_len=max_len, beam_size=beam_size)
                )
            preds = [trg_field.preprocess(x) for x in preds]

            trg_sentences = [[sentence.split()] for sentence in trg_sentences]

            bleu = calc_bleu(preds, trg_sentences)
            print(f"BLEU score :", bleu)
            print("-" * 50)

            results.append({ "src_path": src_path, "trg_path": trg_path, "bleu": bleu })

        # Save log
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "log.json"), "w") as f:
            json.dump(results, f, indent=4)
        print(f"Log is saved to {args.out_dir}/log.json")

    except Exception as e:
        raise e
