import torch
from base.losses.translation_loss import TranslationLoss
from base.metrics.bleu import bleu
from base.schedulers.trans_lr_scheduler import TransLRScheduler
from transformer.model import Transformer
from datasets.wrapper import TextDataWrapper
from utils import load_config
from base.trainer import Trainer
import argparse

if __name__ == "__main__":
    """
    Usage
        python train.py --config_path ./configs/_base_.yaml --train_src_path ./data/train.en --train_trg_path ./data/train.vi --valid_src_path ./data/tst2013.en --valid_trg_path ./data/tst2013.vi --device cuda --init_lr 0.2 --n_warmup_steps 1000 --num_epochs 10
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/_base_.yaml")
    parser.add_argument("--train_src_path", type=str, default="./data/train.en")
    parser.add_argument("--train_trg_path", type=str, default="./data/train.vi")
    parser.add_argument("--valid_src_path", type=str, default="./data/tst2013.en")
    parser.add_argument("--valid_trg_path", type=str, default="./data/tst2013.vi")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--init_lr", type=float, default=0.2)
    parser.add_argument("--n_warmup_steps", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=10)

    args = parser.parse_args()

    try:
        # Load config
        config_dict = load_config(args.config_path)

        # Load dataset
        data_wrapper = TextDataWrapper(src_lang="en_core_web_sm", 
                                    trg_lang="vi_core_news_lg", 
                                    max_len=config_dict['DATA']['MAX_LEN'], 
                                    batch_size=config_dict['DATA']['BATCH_SIZE'],
                                    device=args.device)
        train_dataloader = data_wrapper.create_dataloader(args.train_src_path, args.train_trg_path, is_train=True)
        valid_dataloader = data_wrapper.create_dataloader(args.valid_src_path, args.valid_trg_path, is_train=False)
        src_vocab_size = len(data_wrapper.src_field.vocab)
        trg_vocab_size = len(data_wrapper.trg_field.vocab)

        # Create model
        model = Transformer(config_path=args.config_path,
                            src_vocab_size=src_vocab_size,
                            trg_vocab_size=trg_vocab_size)

        # Optimizer and Loss function
        optimizer = TransLRScheduler(
            torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
            init_lr=args.init_lr,
            d_model=config_dict['MODEL']['ENCODER']['D_MODEL'],
            n_warmup_steps=args.n_warmup_steps,
        )
        criterion = TranslationLoss(
            classes=trg_vocab_size,
            padding_idx=data_wrapper.trg_field.vocab.stoi["<pad>"],
            smoothing=0.1,
        )

        # Run
        
        num_epochs = args.num_epochs
        scorer = bleu
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=num_epochs,
            scorer=scorer,
            src_field=data_wrapper.src_field,
            trg_field=data_wrapper.trg_field,
            max_len=config_dict['DATA']['MAX_LEN'],
            device=args.device,
        )
        
        trainer.fit(train_dataloader, valid_dataloader, k=5)

    except Exception as e:
        raise e