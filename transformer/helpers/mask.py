import numpy as np
import torch

def _gen_padding_mask(inp: torch.Tensor) -> torch.Tensor:
    """
    Generate padding mask for given input.
    Example:
        input = [[1, 2, 3, 4, 0, 0]]
        output = [[0, 0, 0, 0, 1, 1]]
    
    Args:
        - inp (torch.Tensor): Input tensor in shape (batch_size, inp_len)

    Returns:
        - result (torch.Tensor): Padding mask in shape (batch_size, 1, 1, inp_len)
    """
    result = torch.eq(inp, 0).float()[:, np.newaxis, np.newaxis, :]
    return result

def _gen_look_ahead_mask(inp_len):
    """
    Generate look ahead mask for given input.
    Example:
        input = [[1, 2, 3, 0, 0]]
        output = [[[[0., 1., 1., 1., 1.],
                    [0., 0., 1., 1., 1.],
                    [0., 0., 0., 1., 1.],
                    [0., 0., 0., 1., 1.],
                    [0., 0., 0., 1., 1.]]]]

    Args:
        - inp (torch.Tensor): Input tensor in shape (batch_size, inp_len)

    Returns:
        - result (torch.Tensor): Padding mask in shape (batch_size, 1, inp_len, inp_len)
    """
    mask = 1 - torch.tril(torch.ones((inp_len, inp_len)))
    return mask


def gen_mask(inp: torch.Tensor, targ: torch.Tensor):
    """
    Generate masks for the giving input tensor and target tensor. There are 3 masks in the result:
        - Encoder padding mask (base on input)
        - Look ahead mask (base on target)
        - Decoder padding mask (base on target)

    Args:
        - inp (torch.Tensor): Input tensor in shape (batch_size, inp_len)
        - targ (torch.Tensor): Target tensor in shape (batch_size, targ_len)
    
    Returns: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) Three masks.
        - Encoder padding mask in shape (batch_size, 1, 1, inp_len)
        - Look ahead mask in shape (batch_size, 1, targ_len, targ_len)
        - Cross padding mask in shape (batch_size, 1, 1, targ_len)
    """
    # Encoder padding mask
    encoder_padding_mask = _gen_padding_mask(inp).to(inp.device)

    # Cross padding mask: Use for cross-attention for masking encoder output
    cross_padding_mask = _gen_padding_mask(inp).to(inp.device)

    # Look ahead padding mask
    decoder_look_ahead_mask = _gen_look_ahead_mask(targ.shape[1]).to(targ.device)
    # Decoder padding mask
    decoder_inp_padding_mask = _gen_padding_mask(targ).to(targ.device)
    # Update look ahead mask
    decoder_look_ahead_mask = torch.maximum(
        decoder_look_ahead_mask, decoder_inp_padding_mask).to(targ.device)

    return encoder_padding_mask, decoder_look_ahead_mask, cross_padding_mask


# Test
if __name__ == '__main__':
    inp = torch.tensor([[1, 2, 0, 0, 0], [1, 3, 3, 4, 0]])
    targ = torch.tensor([[1, 2, 3, 0, 0], [1, 3, 3, 4, 5]])
    x, y, z = gen_mask(inp, targ)
    print(x)
    print(x.shape)
    print(y)
    print(y.shape)
    print(z)
    print(z.shape)