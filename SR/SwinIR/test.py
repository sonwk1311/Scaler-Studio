import argparse
import os
import time
from typing import Any

import cv2
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

import model
from dataset import CUDAPrefetcher, PairedImageDataset
from imgproc import tensor_to_image
from utils import build_iqa_model, load_pretrained_state_dict, make_directory, AverageMeter, ProgressMeter, Summary


def load_dataset(config: Any, device: torch.device) -> CUDAPrefetcher:
    test_datasets = PairedImageDataset(config["TEST"]["DATASET"]["PAIRED_TEST_GT_IMAGES_DIR"],
                                       config["TEST"]["DATASET"]["PAIRED_TEST_LR_IMAGES_DIR"])
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                 shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                 num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                 pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                 drop_last=False,
                                 persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])
    test_test_data_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return test_test_data_prefetcher


def build_model(config: Any, device: torch.device) -> nn.Module | Any:
    g_model = model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["G"]["CHANNELS"])
    g_model = g_model.to(device)

    # compile model
    if config["MODEL"]["G"]["COMPILED"]:
        g_model = torch.compile(g_model)

    return g_model


def test(
        g_model: nn.Module,
        test_data_prefetcher: CUDAPrefetcher,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        device: torch.device,
        config: Any,
) -> tuple[float, float]:
    if config["TEST"]["SAVE_IMAGE"] and config["TEST"]["SAVE_DIR_PATH"] is None:
        raise ValueError("Image save location cannot be empty!")

    if config["TEST"]["SAVE_IMAGE"]:
        save_dir_path = os.path.join(config["SAVE_DIR_PATH"], config["EXP_NAME"])
        make_directory(save_dir_path)
    else:
        save_dir_path = None

    # Calculate the number of iterations per epoch
    batches = len(test_data_prefetcher)
    # Interval printing
    if batches > 100:
        print_freq = 100
    else:
        print_freq = batches
    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    psnres = AverageMeter("PSNR", ":4.2f", Summary.AVERAGE)
    ssimes = AverageMeter("SSIM", ":4.4f", Summary.AVERAGE)
    progress = ProgressMeter(len(test_data_prefetcher),
                             [batch_time, psnres, ssimes],
                             prefix=f"Test: ")

    # set the model as validation model
    g_model.eval()

    with torch.no_grad():
        # Initialize data batches
        batch_index = 0

        # Set the data set iterator pointer to 0 and load the first batch of data
        test_data_prefetcher.reset()
        batch_data = test_data_prefetcher.next()

        # Record the start time of verifying a batch
        end = time.time()

        while batch_data is not None:
            # Load batches of data
            gt = batch_data["gt"].to(device, non_blocking=True)
            lr = batch_data["lr"].to(device, non_blocking=True)

            # Reasoning
            sr = g_model(lr)

            # Calculate the image sharpness evaluation index
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)

            # record current metrics
            psnres.update(psnr.item(), sr.size(0))
            ssimes.update(ssim.item(), ssim.size(0))

            # Record the total time to verify a batch
            batch_time.update(time.time() - end)
            end = time.time()

            # Output a verification log information
            if batch_index % print_freq == 0:
                progress.display(batch_index)

            # Save the processed image after super-resolution
            if config["TEST"]["SAVE_IMAGE"] and batch_data["image_name"] is None:
                raise ValueError("The image_name is None, please check the dataset.")
            if config["TEST"]["SAVE_IMAGE"]:
                image_name = os.path.basename(batch_data["image_name"][0])
                sr_image = tensor_to_image(sr, False, False)
                sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_dir_path, image_name), sr_image)

            # Preload the next batch of data
            batch_data = test_data_prefetcher.next()

            # Add 1 to the number of data batches
            batch_index += 1

    # Print the performance index of the model at the current Epoch
    progress.display_summary()

    return psnres.avg, ssimes.avg


def main() -> None:
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/test/SWINIRNet_DEFAULT_SR_X4.yaml",
                        required=True,
                        help="Path to test config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    device = torch.device("cuda", config["DEVICE_ID"])
    test_data_prefetcher = load_dataset(config, device)
    g_model = build_model(config, device)
    psnr_model, ssim_model = build_iqa_model(
        config["SCALE"],
        config["ONLY_TEST_Y_CHANNEL"],
        device,
    )

    # Load model weights
    g_model = load_pretrained_state_dict(g_model, config["MODEL"]["G"]["COMPILED"], config["MODEL_WEIGHTS_PATH"])

    # Create a directory for saving test results
    save_dir_path = os.path.join(config["SAVE_DIR_PATH"], config["EXP_NAME"])
    if config["SAVE_IMAGE"]:
        make_directory(save_dir_path)

    test(g_model,
         test_data_prefetcher,
         psnr_model,
         ssim_model,
         device,
         config)


if __name__ == "__main__":
    main()
