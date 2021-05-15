import numpy as np
import asyncio
import random
import websockets
import json
import cv2
import cv2.aruco
import math
from pprint import pprint


def detect_laser(image, lower_threshold=10):
    gray = image[:, :, 2].astype(float) - np.mean(image[:, :, :2], axis=-1)
    gray -= gray.min()
    gray /= gray.max() / 255
    gray = np.clip(gray, 0, 255)
    gray = gray.astype(np.uint8)
    frame_shifts = np.argmax(gray, axis=0)
    for i in range(len(frame_shifts)):
        y = frame_shifts[i]
        if gray[y, i] < lower_threshold:
            frame_shifts[i] = 0
            continue
    return frame_shifts


def rodrigues_rotation(v, k, theta):
    return v*np.cos(theta) + (np.cross(k, v)*np.sin(theta)) + k*(np.dot(k, v))*(1.0-np.cos(theta))


async def position(websocket, path):
    # cap = cv2.VideoCapture("test.h264")
    cap = cv2.VideoCapture("http://raspberrypi.local:8000/stream.mjpg")

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard_create(11, 8, 0.015, 0.01125, aruco_dict)

    fx = fy = width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cx = width / 2
    cy = width / 2
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.zeros(5)

    cv2.namedWindow("", cv2.WINDOW_NORMAL)

    prev_rvec, prev_tvec = None, None
    success, image = cap.read()
    vis = np.zeros_like(image)
    while True:
        success, image = cap.read()
        if not success:
            break
        vis[:] = image

        frame_shifts = detect_laser(image)
        vis[frame_shifts, np.arange(image.shape[1])] = (0, 255, 0)

        corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict)
        # image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
        corners, ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
            image, board, corners, ids, rejected, cameraMatrix=intrinsic_matrix, distCoeffs=dist_coeffs
        )
        if not corners:
            cv2.imshow("", vis)
            cv2.waitKey(1)
            continue
        retval, corners, ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, image, board
        )
        image = cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)

        # rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
        #     corners, markerLength=0.05, cameraMatrix=intrinsic_matrix, distCoeffs=dist_coeffs)
        # for rvec, tvec, id_ in zip(rvecs, tvecs, ids):
        #     if id_ == 77:
        #         image = cv2.aruco.drawAxis(image, intrinsic_matrix, dist_coeffs, rvec, tvec, 0.1)

        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            corners, ids, board, intrinsic_matrix, dist_coeffs, rvec=prev_rvec, tvec=prev_tvec, useExtrinsicGuess=prev_tvec is not None)

        cv2.imshow("", vis)
        cv2.waitKey(1)

        if not success:
            prev_rvec = None
            prev_tvec = None
            continue
        prev_rvec = rvec
        prev_tvec = tvec

        # try:
        #     target_idx = list(ids).index(77)
        # except ValueError:
        #     continue

        # T = tvecs[target_idx][0]
        # R = rvecs[target_idx][0]
        T = tvec[:, 0]
        R = rvec[:, 0]

        # invert
        R = -R
        theta = np.linalg.norm(R)
        Rnorm = R / theta  # normalized
        T = rodrigues_rotation(T, Rnorm, theta)

        message = {
            "camera": {
                "position": {
                    "x": T[0],
                    "y": T[1],
                    "z": T[2],
                },
                "rotation": {
                    "x": Rnorm[0],
                    "y": Rnorm[1],
                    "z": Rnorm[2],
                    "angle": theta,
                }
            }
        }

        await websocket.send(json.dumps(message))

        await asyncio.sleep(1/30)


async def main():
    server = await websockets.serve(position, "localhost", 5678)
    await server.wait_closed()


asyncio.run(main())
