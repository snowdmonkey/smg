import logging

import cv2
import numpy as np
import matplotlib.pyplot as plt

from smg.vqc.inspectors import VideoProp
from smg.vqc.metrics.blockiness import BlockinessPerra
from smg.utils import iter_frames


logger = logging.getLogger(__name__)

video_path = "data/103191001678.mxf"

cap = cv2.VideoCapture(video_path)

video_prop = VideoProp(
    fps=int(cap.get(cv2.CAP_PROP_FPS)),
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    n_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
)

score = np.zeros(shape=video_prop.n_frame, dtype=np.float32)

metric = BlockinessPerra(height=video_prop.height, width=video_prop.width)

for i, frame in enumerate(iter_frames(cap)):
    blockiness = metric.process(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    score[i] = blockiness

    if i % 1_000 == 0:
        # logger.info(f"processed {i:,} frames")
        print(f"processed {i:,} frames")

np.save("demo/blockiness/103191001678.npy", score)

fig = plt.Figure()
ax = fig.add_subplot(111)
ax.plot(score)
fig.savefig("demo/blockiness/103191001678.png")
