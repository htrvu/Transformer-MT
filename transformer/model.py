import torch
from torch import nn
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.helpers import gen_mask
from utils import *
from typing import Tuple, Dict

class Transformer(nn.Module):
    '''
    Transformer model.
    '''

    def __init__(self, config_path: str = '../configs/_base_.yaml'):
        '''
        Args:
            - config_path (str): The path to the config file (default to '../configs/_base_.yaml')
        '''
        super(Transformer, self).__init__()

        config_dict = load_config(config_path)
        self.__parse_config(config_dict)

        self.encoder = Encoder(**self.enc_config)
        self.decoder = Decoder(**self.dec_config)

        self.final_ffn = nn.Linear(self.dec_config['d_model'], self.dec_config['vocab_size'])


    def __parse_config(self, config_dict: dict):
        '''
        Parse the configuration of model from the config file
        '''

        self.enc_config = {
            'vocab_size': config_dict['VOCAB_SIZE']['INPUT'],
            'n_enc_layers': config_dict['ENCODER']['N_LAYERS'],
            'n_heads': config_dict['ENCODER']['N_HEADS'],
            'd_model': config_dict['ENCODER']['D_MODEL'],
            'd_ffn_hidden': config_dict['ENCODER']['D_FFN'],
            'dropout_prob': config_dict['ENCODER']['DROPOUT'],
            'eps': config_dict['ENCODER']['EPS']
        }

        self.dec_config = {
            'vocab_size': config_dict['VOCAB_SIZE']['TARGET'],
            'n_dec_layers': config_dict['DECODER']['N_LAYERS'],
            'n_heads': config_dict['DECODER']['N_HEADS'],
            'd_model': config_dict['DECODER']['D_MODEL'],
            'd_ffn_hidden': config_dict['DECODER']['D_FFN'],
            'dropout_prob': config_dict['DECODER']['DROPOUT'],
            'eps': config_dict['DECODER']['EPS']
        }
    

    def forward(self, inp: torch.Tensor, targ: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        '''
        Args:
            - inp (torch.Tensor): The input tensor in shape (batch_size, inp_length).
            - targ (torch.Tensor): The target tensor in shape (batch_size, targ_length).

        Returns: Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]] The output tensor and attention weights of encoder and decoder
            - The output tensor in shape (batch_size, targ_length)
        '''
        # Generate the masks for encoder and decoder
        enc_padding_mask, dec_look_ahead_mask, cross_padding_mask = gen_mask(inp, targ)

        # Pass throgh encoder
        enc_output, enc_attn_weights_dict = self.encoder(inp, enc_padding_mask)

        # Pass through decoder
        dec_output, dec_attn_weights_dict = self.decoder(targ, enc_output, dec_look_ahead_mask, cross_padding_mask)

        # Final feed forward
        output = self.final_ffn(dec_output)

        return output, enc_attn_weights_dict, dec_attn_weights_dict


# Test
if __name__ == '__main__':
    model = Transformer()
    inp = torch.randint(0, 10000, (64, 50))
    targ = torch.randint(0, 10000, (64, 70))
    output = model(inp, targ)
    print(output[0].shape)