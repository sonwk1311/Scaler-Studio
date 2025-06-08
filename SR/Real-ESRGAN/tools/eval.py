import argparse
from pathlib import Path

from omegaconf import OmegaConf

from real_esrgan.engine.evaler import Evaler
from real_esrgan.utils.envs import select_device


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        metavar="FILE",
        help="path to config file",
    )
    return parser.parse_args()


def main() -> None:
    opts = get_opts()
    config_path = opts.config_path

    config_dict = OmegaConf.load(config_path)
    # merge _BASE_ config
    base_config_path = config_dict.get("_BASE_", False)
    if base_config_path:
        base_config_dict = OmegaConf.load(Path(config_path).absolute().parent / Path(base_config_path))
        config_dict = OmegaConf.merge(base_config_dict, config_dict)
    device = select_device(config_dict.DEVICE)

    evaler = Evaler(config_dict, device)
    psnr, ssim, niqe = evaler.evaluate()
    print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, NIQE: {niqe:.2f}")


if __name__ == "__main__":
    main()
