import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext.legacy.data import Field
from typing import Tuple

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

from transformer import Transformer
from transformer.helpers import gen_mask
from utils import *
from constants import *


# def get_synonym(token: str, vocab: Field) -> int:
#     '''
#     Get synonym of a word in vocabulary

#     Args:
#         - token (str): word
#         - vocab (Field): data field (or vocabulary)
    
#     Returns: index of synonym in vocabulary
#     '''
#     syns = wordnet.synsets(token)
#     for syn in syns:
#         for l in syn.lemmas():
#             if l.name() in vocab.stoi:
#                 return vocab.stoi[l.name()]
#     return 0


def get_synonym(word, field):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if field.vocab.stoi[l.name()] != 0:
                return field.vocab.stoi[l.name()]
            
    return 0


def _first_generate(
    src: Variable,
    model: nn.Module,
    src_field: Field,
    trg_field: Field,
    device: str,
    beam_size: int,
    max_len: int,
) -> Tuple[torch.Tensor]:
    '''
    Translate the first word for a given source sentence

    Args:
        - src (Variable): source sentence
        - model (nn.Module): model
        - src_field (Field): source field (or vocabulary)
        - trg_field (Field): target field (or vocabulary)
        - device (str): device to run model
        - beam_size (int): beam size
        - max_len (int): maximum length of sentence

    Returns: (tuple[torch.Tensor]) generated output, encoder output, log scores
        - Generated output in shape (beam_size, max_len)
        - Encoder output in shape (beam_size, max_len, d_model)
        - Encoder padding mask in shape (1, 1, 1, max_len)
        - Log scores in shape (1, beam_size)
    '''

    # Decoder start with init token SOS
    init_token = trg_field.vocab.stoi[SOS]
    dec_input = torch.LongTensor([[init_token]]).to(device)
    
    # [DEBUG] hmm
    # Gen masks
    enc_padding_mask, dec_look_ahead_mask = gen_mask(src, 
                                                     src_field.vocab.stoi[PAD], 
                                                     dec_input,
                                                     trg_field.vocab.stoi[PAD])

    # Pass through encoder
    # enc_padding_mask = (src != src_field.vocab.stoi[PAD]).unsqueeze(-2)
    enc_output, _ = model.encoder(src, enc_padding_mask)

    # Pass through decoder
    dec_out, _ = model.decoder(dec_input, enc_output, dec_look_ahead_mask, enc_padding_mask)

    # Final feed forward to predict word
    cls_out = model.final_ffn(dec_out)
    cls_out = F.softmax(cls_out, dim=-1)

    # Get k best outputs for the first word
    probs, ix = cls_out[:, -1].data.topk(beam_size)
    log_scores = torch.Tensor([torch.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    # Create matrix result for the whole beam search process
    outputs_batch = torch.zeros(beam_size, max_len).long().to(device)
    outputs_batch[:, 0] = init_token
    outputs_batch[:, 1] = ix[0]   # First predicted word

    # Store encoder output for later uses
    enc_outputs = torch.zeros(beam_size, enc_output.size(-2), enc_output.size(-1)).to(device)
    enc_outputs[:, :] = enc_output[0]

    return outputs_batch, enc_outputs, enc_padding_mask, log_scores


def beam_search(
    src: Variable,
    model: Transformer,
    src_field: Field,
    trg_field: Field,
    max_len: int,
    beam_size: int = 1,
    device: str = 'cuda',
):
    '''
    Perform beam search to generate a translated sentence for a given source sentence

    Args:
        - src (Variable): source sentence
        - model (nn.Module): model
        - src_field (Field): source field (or vocabulary)
        - trg_field (Field): target field (or vocabulary)
        - max_len (int): maximum length of sentence
        - k (int): beam size (default to 1)
        - device (str): device to run model (default to 'cuda)
    '''
    # Translate first word. We only need to calculate encoder output once
    outputs_batch, enc_outputs, enc_padding_mask, log_scores = _first_generate(src, model, src_field, trg_field, device, beam_size, max_len)

    # Next words, we predict with decoder input is a batch of sentences
    eos_tok = trg_field.vocab.stoi[EOS]
    # [DEBUG] Hmm
    # enc_padding_mask = (src != src_field.vocab.stoi[PAD]).unsqueeze(-2)
    selected_result = None
    for i in range(2, max_len):
        # Gen new look ahead mask for decoder
        _, dec_look_ahead_mask = gen_mask(src, 
                                             src_field.vocab.stoi[PAD],
                                             outputs_batch[:, :i],
                                             trg_field.vocab.stoi[PAD])

        # Pass through decoder
        dec_out, _ = model.decoder(outputs_batch[:, :i], enc_outputs, dec_look_ahead_mask, enc_padding_mask)
        cls_out = model.final_ffn(dec_out)
        cls_out = F.softmax(cls_out, dim=-1)

        # Pick last word (predicted word at this step)
        probs, ix = cls_out[:, -1].data.topk(beam_size)
        
        # Update log scores (mul of prob --> sum of log prob)
        log_scores = torch.Tensor([torch.log(p) for p in probs.data.view(-1)]).view(beam_size, -1) \
                        + log_scores.transpose(0, 1)
        
        # Find top-k sentences (k sentences with highest log scores)
        k_probs, k_ix = log_scores.view(-1).topk(beam_size)
        row = torch.div(k_ix, beam_size, rounding_mode="floor")
        col = torch.fmod(k_ix, beam_size)
        outputs_batch[:, :i] = outputs_batch[row, :i]
        outputs_batch[:, i] = ix[row, col]
        log_scores = k_probs.unsqueeze(0)

        # Check if any sentence has ended (i.e. has EOS token)
        ones = (outputs_batch == eos_tok).nonzero()
        sentences_length = torch.zeros(len(outputs_batch), dtype=torch.long).to(device)
        for vec in ones:
            i = vec[0]
            if sentences_length[i] == 0:  # First end symbol has not been found yet
                sentences_length[i] = vec[1]  # Position of end symbol
        num_finished_sentences = len([s for s in sentences_length if s > 0])

        # Stop beam search if all sentences have ended
        if num_finished_sentences == beam_size:
            alpha = 0.7
            div = 1 / (sentences_length.type_as(log_scores) ** alpha)
            _, ind = torch.max(log_scores * div, 1)
            selected_result = ind.data[0]
            break

    if selected_result is None:
        length = -1
        if len((outputs_batch[0] == eos_tok).nonzero()) > 0:
            length = (outputs_batch[0] == eos_tok).nonzero()[0]
        return " ".join([trg_field.vocab.itos[tok] for tok in outputs_batch[0][1:length]])
    else:
        length = (outputs_batch[selected_result] == eos_tok).nonzero()[0]
        return " ".join([trg_field.vocab.itos[tok] for tok in outputs_batch[selected_result][1:length]])
