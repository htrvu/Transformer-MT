import torch
from torch import nn
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from utils import *
from typing import Tuple, Dict

class Transformer(nn.Module):
    '''
    Transformer model.
    '''

    def __init__(self, config_path: str = '../configs/_base_.yaml', src_vocab_size: int = 10000, trg_vocab_size: int = 10000):
        '''
        Args:
            - config_path (str): The path to the config file (default to '../configs/_base_.yaml')
            - src_vocab_size (int): The size of source vocabulary (default to 10000)
            - trg_vocab_size (int): The size of target vocabulary (default to 10000)
        '''
        super(Transformer, self).__init__()

        config_dict = load_config(config_path)
        self.__parse_config(config_dict, src_vocab_size, trg_vocab_size)

        self.encoder = Encoder(**self.enc_config)
        self.decoder = Decoder(**self.dec_config)
        self.final_ffn = nn.Linear(self.dec_config['d_model'], trg_vocab_size)

        self.__init_weights()


    def __parse_config(self, config_dict: dict, src_vocab_size: int, trg_vocab_size: int):
        '''
        Parse the configuration of model from the config file
        '''
        config_dict = config_dict['MODEL']

        self.enc_config = {
            'vocab_size': src_vocab_size,
            'n_enc_layers': config_dict['ENCODER']['N_LAYERS'],
            'n_heads': config_dict['ENCODER']['N_HEADS'],
            'd_model': config_dict['ENCODER']['D_MODEL'],
            'd_ffn_hidden': config_dict['ENCODER']['D_FFN'],
            'dropout_prob': config_dict['ENCODER']['DROPOUT'],
            'eps': config_dict['ENCODER']['EPS']
        }

        self.dec_config = {
            'vocab_size': trg_vocab_size,
            'n_dec_layers': config_dict['DECODER']['N_LAYERS'],
            'n_heads': config_dict['DECODER']['N_HEADS'],
            'd_model': config_dict['DECODER']['D_MODEL'],
            'd_ffn_hidden': config_dict['DECODER']['D_FFN'],
            'dropout_prob': config_dict['DECODER']['DROPOUT'],
            'eps': config_dict['DECODER']['EPS']
        }
    
    
    def __init_weights(self):
        '''
        Initialize the weights of the model
        '''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, 
                enc_input: torch.Tensor, 
                dec_input: torch.Tensor, 
                enc_padding_mask: torch.Tensor = None, 
                dec_look_ahead_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        '''
        Args:
            - enc_input (torch.Tensor): The encoder input tensor in shape (batch_size, enc_inp_length).
            - dec_input (torch.Tensor): The decoder input tensor in shape (batch_size, dec_inp_length).
            - enc_padding_mask (torch.Tensor): The encoder padding mask in shape (batch_size, 1, 1, enc_inp_length).
            - dec_look_ahead_mask (torch.Tensor): The decoder look ahead mask in shape (batch_size, 1, dec_inp_length, dec_inp_length).

        Returns: Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]] The output tensor and attention weights of encoder and decoder
            - The output tensor in shape (batch_size, dec_inp_length, trg_vocab_size)
        '''
        # Pass throgh encoder
        enc_output, enc_attn_weights_dict = self.encoder(enc_input, enc_padding_mask)

        # Pass through decoder
        dec_output, dec_attn_weights_dict = self.decoder(dec_input, enc_output, dec_look_ahead_mask, enc_padding_mask)

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