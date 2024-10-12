import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from typing import Iterable, List, Set
from scipy.ndimage import distance_transform_edt

# Boundary loss from https://github.com/LIVIAETS/boundary-loss
def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)   

class SurfaceLoss():
    '''
    Boundary loss implementation 
    Inputs:
    @probs: probability maps provded from the output of the network 
    @dc: distance maps computed when the dataset class is initialized
    outputs:
    @loss: boundary loss
    @description: 
    the loss finetunes the probability maps by the groundtruth distance map representations.
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)
        multipled = multipled/multipled.max()
      
        loss = multipled.mean()

        return loss

def get_distance_map(mask_tensor):
    """
    Calculate the distance map from a binary tensor (segmentation mask).

    Parameters:
    mask_tensor (torch.Tensor): A binary tensor with shape (B, C, H, W), where B is the batch size,
                               C is the number of channels, and non-zero pixels represent the object of interest.

    Returns:
    distance_map (torch.Tensor): A tensor of shape (B, 1, H, W) representing the distance map. Each pixel
                                value indicates the distance to the nearest non-zero pixel in the mask.
    """
    mask_numpy = mask_tensor.cpu().numpy().squeeze(axis=1)
    mask_numpy = ~mask_numpy.astype(bool)
    distance_maps = [distance_transform_edt(mask) for mask in mask_numpy]

    distance_map_tensor = torch.from_numpy(np.array(distance_maps)).cuda()
    distance_map_tensor = distance_map_tensor.unsqueeze(1)

    return distance_map_tensor


class LogCoshDiceLoss(nn.Module):
    """
    L_{lc-dce} = log(cosh(DiceLoss)
    """
    def __init__(self, use_softmax=True):
        super(LogCoshDiceLoss, self).__init__()
        self.use_softmax = use_softmax

    def forward(self, output, target, epsilon=1e-6):
        num_classes = output.shape[1]
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            output = F.softmax(output, dim=1)
        one_hot_target = F.one_hot(target.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert output.shape == one_hot_target.shape
        numerator = 2. * torch.sum(output * one_hot_target, dim=(-2, -1))  # Shape [batch, n_classes]
        denominator = torch.sum(output + one_hot_target, dim=(-2, -1))
        return torch.log(torch.cosh(1 - torch.mean((numerator + epsilon) / (denominator + epsilon))))



