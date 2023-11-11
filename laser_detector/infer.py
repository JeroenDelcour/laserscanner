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


def main(checkpoint, video, threshold=0.9, cutoff=0.01):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_size = (192, 2048)
    output_size = (48, 512)

    model = HourglassNet(nstack=1, inp_dim=256, oup_dim=1)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video)

    cv2.namedWindow("vis", cv2.WINDOW_NORMAL)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    writer = cv2.VideoWriter(
        "vis.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps=int(cap.get(cv2.CAP_PROP_FPS)),
        frameSize=(width, height),
    )

    y_scale = height / output_size[0]
    x_scale = width / output_size[1]
    y_grid = np.repeat(
        np.arange(output_size[0])[:, np.newaxis], repeats=output_size[1], axis=1
    )
    x_range = np.arange(output_size[1])

    pbar = tqdm()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        model_input = cv2.resize(frame, dsize=input_size[::-1])
        model_input = np.moveaxis(model_input, 2, 0)  # HWC to CHW
        model_input = model_input[np.newaxis, :]  # batch dimension
        model_input = model_input.astype(np.float32) / 255
        model_input = torch.Tensor(model_input)
        model_input = model_input.to(device)

        with torch.no_grad():
            outputs = model(model_input)
        heatmap = outputs[0, -1, 0].cpu().numpy().astype(float)

        # Visualize points
        heatmap[heatmap < cutoff] = 1e-12
        # mask = heatmap.max(axis=0) > threshold
        y_coords = np.average(y_grid, axis=0, weights=heatmap)
        mask = heatmap[y_coords.round().astype(int), x_range] > threshold
        y_coords = y_coords[mask]
        x_coords = x_range[mask].astype(float)
        y_coords *= y_scale
        x_coords *= x_scale
        vis = frame
        for x, y in zip(x_coords, y_coords):
            vis = cv2.circle(
                vis,
                center=(round(x), round(y)),
                radius=0,
                color=(0, 0, 255),
                thickness=2,
            )

        # # Visualize heatmap
        # vis = np.clip(heatmap, 0, 1)
        # vis *= 255
        # vis = vis.astype(np.uint8)
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # vis = cv2.resize(vis, dsize=(width, height), interpolation=0)
        # vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        # vis = np.hstack((frame, vis))

        cv2.imshow("vis", vis)
        if cv2.waitKey(1) == ord("q"):
            break

        writer.write(frame)

        pbar.update()

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
