import numpy as np
import torch

def _gen_padding_mask(inp: torch.Tensor, src_pad: int) -> torch.Tensor:
    """
    Generate padding mask for given input.
    Example:
        input = [[1, 2, 3, 4, 0, 0]]
        pad = 0
        output = [[0, 0, 0, 0, 1, 1]]
    
    Args:
        - inp (torch.Tensor): Input tensor in shape (batch_size, inp_len)
        - src_pad (int): The index of padding token in source vocabulary

    Returns:
        - result (torch.Tensor): Padding mask in shape (batch_size, 1, 1, inp_len)
    """
    # result = torch.eq(inp, 0).float()[:, np.newaxis, np.newaxis, :]

    result = torch.eq(inp, src_pad)[:, np.newaxis, np.newaxis, :]
    result = result.to(torch.uint8)

    # result = (inp == src_pad).unsqueeze(-2).to(torch.uint8)
    return result

def _gen_look_ahead_mask(inp_len):
    """
    Generate look ahead mask for given input.
    Example:
        input = [[1, 2, 3, 0, 0]]
        output = [[[[0., 1., 1., 1., 1.],
                    [0., 0., 1., 1., 1.],
                    [0., 0., 0., 1., 1.],
                    [0., 0., 0., 0., 1.],
                    [0., 0., 0., 0., 0.]]]]

    Args:
        - inp (torch.Tensor): Input tensor in shape (batch_size, inp_len)

    Returns:
        - result (torch.Tensor): Padding mask in shape (batch_size, 1, inp_len, inp_len)
    """
    mask = (1 - torch.tril(torch.ones((inp_len, inp_len))))
    mask = mask.to(torch.uint8)
    # mask = np.triu(np.ones((1, inp_len, inp_len)), k=1)
    # mask = torch.from_numpy(mask).to(torch.uint8)
    return mask

def gen_mask(enc_input: torch.Tensor, src_pad: int, dec_input: torch.Tensor, trg_pad: int):
    """
    Generate masks for the giving input tensor and target tensor. There are 3 masks in the result:
        - Encoder padding mask (base on input)
        - Look ahead mask (base on target)

    Args:
        - enc_input (torch.Tensor): Input tensor in shape (batch_size, enc_input_len)
        - src_pad (int): The index of padding token in source vocabulary
        - dec_input (torch.Tensor): Target tensor in shape (batch_size, dec_input_len)
        - trg_pad (int): The index of padding token in target vocabulary

    Returns: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) Three masks.
        - Encoder padding mask in shape (batch_size, 1, 1, enc_input_len)
        - Look ahead mask in shape (batch_size, 1, targ_len, dec_input_len)
    """
    # Encoder padding mask
    enc_padding_mask = _gen_padding_mask(enc_input, src_pad).to(enc_input.device)

    # Look ahead padding mask
    dec_look_ahead_mask = _gen_look_ahead_mask(dec_input.shape[1]).to(dec_input.device)
    # Decoder padding mask
    dec_inp_padding_mask = _gen_padding_mask(dec_input, trg_pad).to(dec_input.device)
    # Update look ahead mask
    # Example:
    #     input = [[1, 2, 3, 0, 0]]
    #     output = [[0, 1, 1, 1, 1]
    #               [0, 0, 1, 1, 1]
    #               [0, 0, 0, 1, 1]
    #               [0, 0, 0, 1, 1]
    #               [0, 0, 0, 1, 1]]
    dec_look_ahead_mask = torch.maximum(dec_look_ahead_mask, dec_inp_padding_mask).to(torch.uint8)
    dec_look_ahead_mask = dec_look_ahead_mask.to(dec_input.device)

    return enc_padding_mask, dec_look_ahead_mask


# Test
if __name__ == '__main__':
    inp = torch.tensor([[1, 2, 0, 0, 0], [1, 3, 3, 4, 0]])
    targ = torch.tensor([[1, 2, 3, 0, 0], [1, 3, 3, 4, 5]])
    x, y = gen_mask(inp, targ)
    print(x)
    print(x.shape)
    print(y)
    print(y.shape)