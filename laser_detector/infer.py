from pathlib import Path

import cv2
import numpy as np
import torch
from hourglassnet import HourglassNet
from tqdm import tqdm

checkpoint = Path("checkpoints") / "2023-11-10T16:13:51.380865.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = HourglassNet(nstack=2, inp_dim=256, oup_dim=1)
model.to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

cap = cv2.VideoCapture("data/videos/2023-11-10_video0.mp4")

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

input_size = (512, 512)

pbar = tqdm()
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, dsize=input_size)

    model_input = np.moveaxis(frame, 2, 0)  # HWC to CHW
    model_input = model_input[np.newaxis, :]  # batch dimension
    model_input = model_input.astype(np.float32) / 255
    model_input = torch.Tensor(model_input)
    model_input = model_input.to(device)

    with torch.no_grad():
        outputs = model(model_input)
    heatmap = outputs[0, 0, 0]

    vis = heatmap.cpu().numpy()
    vis = np.clip(vis, 0, 1)
    vis *= 255
    vis = vis.astype(np.uint8)

    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vis = cv2.resize(vis, dsize=input_size)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis = np.hstack((frame, vis))

    cv2.imshow("frame", vis)
    if cv2.waitKey(1) == ord("q"):
        break

    pbar.update()

cap.release()
cv2.destroyAllWindows()
