import torch
from real_esrgan.models.vgg_feature_extractor import VGGFeatureExtractor
from torch import Tensor, nn

__all__ = [
    "FeatureLoss",
]


class FeatureLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            arch_name: str,
            layer_weight_dict: dict[str, float] = None,
            normalize: bool = True,
    ) -> None:
        super(FeatureLoss, self).__init__()
        if layer_weight_dict is None:
            layer_weight_dict = {
                "relu1_2": 0.1,
                "relu2_2": 0.1,
                "relu3_4": 1.0,
                "relu4_4": 1.0,
                "relu5_4": 1.0,
            }

        self.vgg_feature_extractor = VGGFeatureExtractor(
            arch_name=arch_name,
            layer_name_list=list(layer_weight_dict.keys()),
            normalize=normalize)
        self.layer_weight_dict = layer_weight_dict

        self.loss_function = nn.L1Loss()

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        assert inputs.size() == target.size(), "Two tensor must have the same size"

        inputs_features = self.vgg_feature_extractor(inputs)
        target_features = self.vgg_feature_extractor(target.detach())

        loss = 0.
        for k in inputs_features.keys():
            loss += self.layer_weight_dict[k] * self.loss_function(inputs_features[k], target_features[k])

        return torch.Tensor(loss).to(device=inputs.device)
