from pathlib import Path

import cv2
import numpy as np

output_folder = Path("/Users/liuxuefe/PycharmProjects/smg/1031910016782020-06-17-20-41-54")
output_folder.mkdir(exist_ok=True, parents=True)


cap = cv2.VideoCapture("/Users/liuxuefe/PycharmProjects/smg/data/103191001678.mxf")

i = 0

while True:

    ret, frame = cap.read()

    if not ret:
        cap.release()
        break
    elif i in (
            36047, 36197, 35612, 35687, 35882, 36152, 36002, 35957, 35657, 36122, 36092,
            36017, 35912, 35672, 35972, 35642, 35927, 35942,36032, 35627):
        cv2.imwrite(str(output_folder / f"{i}.png"), frame)

    if i % 1000 == 0:
        print(i)
    i += 1
