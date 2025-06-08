import argparse

from omegaconf import OmegaConf

from real_esrgan.apis.super_resolution import SuperResolutionInferencer


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

    inferencer = SuperResolutionInferencer(config_dict)
    inferencer.warmup()
    inferencer.inference()


if __name__ == "__main__":
    main()
