import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from hourglassnet import HourglassNet
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to Torch checkpoint")
    parser.add_argument("video", type=str, help="Path to video")
    return parser.parse_args()


def main(checkpoint, video):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = HourglassNet(nstack=1, inp_dim=256, oup_dim=1)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    input_size = (384, 512)

    pbar = tqdm()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, dsize=input_size[::-1])

        model_input = np.moveaxis(frame, 2, 0)  # HWC to CHW
        model_input = model_input[np.newaxis, :]  # batch dimension
        model_input = model_input.astype(np.float32) / 255
        model_input = torch.Tensor(model_input)
        model_input = model_input.to(device)

        with torch.no_grad():
            outputs = model(model_input)
        heatmap = outputs[0, -1, 0]

        vis = heatmap.cpu().numpy()
        vis = np.clip(vis, 0, 1)
        vis *= 255
        vis = vis.astype(np.uint8)

        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vis = cv2.resize(vis, dsize=input_size[::-1], interpolation=0)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        vis = np.hstack((frame, vis))

        cv2.imshow("frame", vis)
        if cv2.waitKey(1) == ord("q"):
            break

        pbar.update()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
