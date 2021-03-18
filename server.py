import numpy as np
import asyncio
import random
import websockets
import json
import cv2
import cv2.aruco
import math
from pprint import pprint


def rodrigues_rotation(v, k, theta):
    return v*np.cos(theta) + (np.cross(k, v)*np.sin(theta)) + k*(np.dot(k, v))*(1.0-np.cos(theta))


async def position(websocket, path):
    # cap = cv2.VideoCapture("/home/jeroen/Downloads/VID_20210316_144205.mp4")
    cap = cv2.VideoCapture("http://192.168.178.10:8080/video")

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)

    fx = fy = width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cx = width / 2
    cy = width / 2
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.zeros(5)

    cv2.namedWindow("", cv2.WINDOW_NORMAL)

    while True:
        success, image = cap.read()
        if not success:
            break

        corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict)
        image = cv2.aruco.drawDetectedMarkers(image, corners, ids)

        if not corners:
            cv2.imshow("", image)
            cv2.waitKey(1)
            continue
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners, markerLength=0.05, cameraMatrix=intrinsic_matrix, distCoeffs=dist_coeffs)
        for rvec, tvec, id_ in zip(rvecs, tvecs, ids):
            if id_ == 77:
                image = cv2.aruco.drawAxis(image, intrinsic_matrix, dist_coeffs, rvec, tvec, 0.1)

        cv2.imshow("", image)
        cv2.waitKey(1)

        try:
            target_idx = list(ids).index(77)
        except ValueError:
            continue

        T = tvecs[target_idx][0]
        R = rvecs[target_idx][0]

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
