from pathlib import Path

import cv2
from omegaconf import DictConfig
from tqdm import tqdm

from real_esrgan.utils.events import LOGGER
from .super_resolution import SuperResolutionInferencer

__all__ = [
    "VideoSuperResolutionInferencer",
]


class VideoSuperResolutionInferencer(SuperResolutionInferencer):
    def __init__(self, config_dict: DictConfig) -> None:
        super().__init__(config_dict)
        # load inference config
        self.upscale_factor = config_dict.UPSCALE_FACTOR

    def inference(self) -> None:
        # get video information
        video_capture = cv2.VideoCapture(self.inputs)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        num_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        sr_video_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) * self.upscale_factor),
                         int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) * self.upscale_factor)

        sr_video_path = self.output / Path(self.inputs).with_suffix(".avi").name
        sr_video_writer = cv2.VideoWriter(str(sr_video_path), cv2.VideoWriter_fourcc("M", "P", "E", "G"), fps, sr_video_size)

        success, frame = video_capture.read()
        pbar = tqdm(range(int(num_frames)), desc="Processing")
        for _ in pbar:
            if success:
                lr_tensor = self.pre_process(frame)
                sr_tensor = self.model(lr_tensor)
                sr_image = self.post_process(sr_tensor)

                sr_video_writer.write(sr_image)
                success, frame = video_capture.read()

        LOGGER.info(f"SR image save to `{sr_video_path}`")
