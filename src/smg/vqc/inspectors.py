import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

import cv2
import click
import numpy as np
import torch


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VideoProp:
    fps: int
    width: int
    height: int
    n_frame: int


class Detector(ABC):
    def __init__(self):
        super(Detector, self).__init__()
        self._video_prop: Optional[VideoProp] = None
        self._score: Optional[np.array] = None

    def _get_video_meta_info(self, cap: cv2.VideoCapture):
        self._video_prop = VideoProp(
            fps=int(cap.get(cv2.CAP_PROP_FPS)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )

        logger.info(f"video props: {self._video_prop}")

    def process_video(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        output_path = Path(video_path.stem + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self._get_video_meta_info(cap)

        i_frame = 0

        self._score = np.zeros(self._video_prop.n_frame, dtype=np.float)

        while True:
            ret, current_frame = cap.read()

            if not ret:
                cap.release()
                self._save_result(output_path)
                break

            self._score[i_frame] = self.get_score(current_frame)

            if i_frame % 1_000 == 0:
                logger.info(f"processed {i_frame:,} frames")

            i_frame += 1

        logger.info(f"read {i_frame:,} frames")

    @abstractmethod
    def get_score(self, frame: np.array) -> float:
        pass

    def _save_result(self, output_path: Path = None):
        if output_path is None:
            output_path = Path(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        output_path.mkdir(exist_ok=True, parents=True)
        np.save(output_path / "score.npy", self._score)
        import matplotlib.pyplot as plt
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        ax.plot(self._score)
        fig.savefig(output_path / 'score.png')


class NoiseDetector(Detector):
    def __init__(self):
        super(NoiseDetector, self).__init__()

    def get_score(self, frame: np.array):
        return cv2.Laplacian(frame, cv2.CV_64F).var()


class MosaicDetector(Detector):
    def __init__(self):
        super(MosaicDetector, self).__init__()

    def get_score(self, frame: np.array) -> float:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = torch.tensor(frame)

        mu = torch.nn.functional.avg_pool2d(frame.unsqueeze(0).float(), kernel_size=8, stride=8)[0]
        mu2 = torch.nn.functional.avg_pool2d(frame.unsqueeze(0).float()**2, kernel_size=8, stride=8)[0]
        mosaic_mask = mu2 == mu ** 2
        if torch.unique(mu[mosaic_mask]).nelement() < 20:
            return 0
        return torch.sum(mosaic_mask).item()


@click.command()
@click.argument("video_path", type=str)
def main(video_path: str):
    detector = NoiseDetector()
    detector.process_video(Path(video_path))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s")
    main()
