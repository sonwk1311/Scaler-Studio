import torch
from torch import Tensor, nn

from real_esrgan.utils.color import rgb_to_ycbcr_torch
from real_esrgan.utils.ops import check_tensor_shape

__all__ = [
    "mse_torch",
    "MSE",
]


def mse_torch(
        raw_tensor: Tensor,
        dst_tensor: Tensor,
        only_test_y_channel: bool,
        data_range: float = 1.0,
        eps: float = 1e-8,
) -> Tensor:
    r"""PyTorch implements the MSE (Mean Squared Error, mean square error) function

    Args:
        raw_tensor (Tensor): tensor flow of images to be compared, RGB format, data range [0, 1]
        dst_tensor (Tensor): reference image tensor flow, RGB format, data range [0, 1]
        only_test_y_channel (bool): Whether to test only the Y channel of the image
        data_range (float, optional): Maximum value range of images. Default: 1.0
        eps (float, optional): Deviation prevention denominator is 0. Default: 1e-8

    Returns:
        mse_metrics (Tensor): MSE metrics

    """
    # Convert RGB tensor data to YCbCr tensor, and only extract Y channel data
    if only_test_y_channel and raw_tensor.shape[1] == 3 and dst_tensor.shape[1] == 3:
        raw_tensor = rgb_to_ycbcr_torch(raw_tensor, True)
        dst_tensor = rgb_to_ycbcr_torch(dst_tensor, True)

    mse_metrics = torch.mean((raw_tensor * data_range - dst_tensor * data_range) ** 2 + eps, dim=[1, 2, 3])

    return mse_metrics


class MSE(nn.Module):
    r"""PyTorch implements the MSE (Mean Squared Error, mean square error) function"""

    def __init__(self, crop_border: int = 0, only_test_y_channel: bool = True, **kwargs) -> None:
        """

        Args:
            crop_border (int, optional): how many pixels to crop border. Default: 0
            only_test_y_channel (bool, optional): Whether to test only the Y channel of the image. Default: ``True``

        Returns:
            mse_metrics (Tensor): MSE metrics
        """

        super(MSE, self).__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel
        self.kwargs = kwargs

    def forward(self, raw_tensor: Tensor, dst_tensor: Tensor) -> Tensor:
        # Check if two tensor scales are similar
        check_tensor_shape(raw_tensor, dst_tensor)

        # crop pixel boundaries
        if self.crop_border > 0:
            raw_tensor = raw_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]
            dst_tensor = dst_tensor[..., self.crop_border:-self.crop_border, self.crop_border:-self.crop_border]

        mse_metrics = mse_torch(raw_tensor, dst_tensor, self.only_test_y_channel, **self.kwargs)

        return mse_metrics
