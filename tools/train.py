import os
import datetime
import torch
from base.losses.translation_loss import TranslationLoss
from base.metrics.bleu import calc_bleu
from base.schedulers.trans_lr_scheduler import TransLRScheduler
from transformer.model import Transformer
from datasets.wrapper import TextDataWrapper
from utils import load_config
from base.trainer import Trainer
from constants import *
import argparse

if __name__ == "__main__":
    """
    Usage
        python train.py \
            --config_path ./configs/_base_.yaml \
            --train_src_path ./data/train.en \
            --train_trg_path ./data/train.vi \
            --valid_src_path ./data/tst2013.en \
            --valid_trg_path ./data/tst2013.vi \
            --device cuda:0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/_base_.yaml")
    parser.add_argument("--train_src_path", type=str, default="./data/train.en")
    parser.add_argument("--train_trg_path", type=str, default="./data/train.vi")
    parser.add_argument("--valid_src_path", type=str, default="./data/tst2013.en")
    parser.add_argument("--valid_trg_path", type=str, default="./data/tst2013.vi")
    parser.add_argument("--load_from", type=str, default=None, help="path to checkpoint to be loaded")
    parser.add_argument("--out_dir", type=str, default="./runs", help="directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    try:
        # Load config
        config_dict = load_config(args.config_path)
        print('Config:', config_dict)

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
        if args.load_from is not None:
            model.load_state_dict(torch.load(args.load_from))

        # Optimizer and Loss function
        optimizer = TransLRScheduler(
            torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
            init_lr=config_dict['OPTIM']['INIT_LR'],
            d_model=config_dict['MODEL']['ENCODER']['D_MODEL'],
            n_warmup_steps=config_dict['OPTIM']['N_WARMUP_STEPS'],
        )
        criterion = TranslationLoss(
            classes=trg_vocab_size,
            padding_idx=data_wrapper.trg_field.vocab.stoi[PAD],
            smoothing=0.1,
        )

        # Run
        trainer = Trainer(
            model=model,
            optimizer=optimizer,    
            criterion=criterion,
            num_epochs=config_dict['TRAINER']['N_EPOCHS'],
            metric=calc_bleu,
            src_field=data_wrapper.src_field,
            trg_field=data_wrapper.trg_field,
            max_len=config_dict['DATA']['MAX_LEN'],
            device=args.device,
        )
        
        if args.out_dir == './runs':
            now = datetime.datetime.now()
            c = now.strftime("%Y-%m-%d_%H-%M-%S")
            args.out_dir = os.path.join(args.out_dir, c)

        start = datetime.datetime.now()
        trainer.fit(train_dataloader, valid_dataloader, out_dir=args.out_dir)
        end = datetime.datetime.now()
        print('Training time:', end - start)

    except Exception as e:
        raise e