import numpy as np
import torch


def _get_angles(d_model: int) -> np.array:
    """
    Calculate the angle for each dimension.

    Args:
        - d_model (int): The dimension (of linear projection for the query, key and value matrices).

    Returns (np.array): The angle array in shape (1, d_model).
    """
    angles = np.arange(d_model)
    angles[1::2] = angles[0::2]
    angles = 1 / (10000 ** (angles / d_model))
    angles = np.expand_dims(angles, axis=0)
    return angles


def calc_positional_encoding(last_pos, d_model):
    """
    Calculate the positional encoding for all positions from 0 to pos - 1.

    Args:
        - last_pos (int): Position to calculate positional encoding.
        - d_model (int): The dimension (of linear projection for the query, key and value matrices).

    Returns (torch.Tensor): The positional encoding in shape (1, last_pos, d_model).
    """
    angles = _get_angles(d_model)
    last_pos = np.expand_dims(np.arange(last_pos), axis=1)
    pos_angles = last_pos.dot(angles)
    pos_angles[:, 0::2] = np.sin(pos_angles[:, 0::2])
    pos_angles[:, 1::2] = np.cos(pos_angles[:, 1::2])
    pos_angles = np.expand_dims(pos_angles, axis=0)
    return torch.from_numpy(pos_angles).float()


# Test
if __name__ == '__main__':
    print(calc_positional_encoding(50, 512).shape)