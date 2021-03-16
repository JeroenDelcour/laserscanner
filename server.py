import numpy as np
import asyncio
import random
import websockets
import json
import cv2
import cv2.aruco
import math


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


# async def position(websocket, path):
#     while True:
#         message = {
#             "camera": {
#                 "position": {
#                     "x": random.random(),
#                     "y": random.random(),
#                     "z": random.random(),
#                 },
#                 "rotation": {
#                     "x": 0,
#                     "y": 0,
#                     "z": 0,
#                 }
#             }
#         }
#         await websocket.send(json.dumps(message))
#         await asyncio.sleep(1)

async def position(websocket, path):
    cap = cv2.VideoCapture("/home/jeroen/Downloads/VID_20210316_144205.mp4")

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_100)

    fx = fy = width = 1280
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
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        cv2.imshow("", image)
        cv2.waitKey(1)

        if not corners:
            continue
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners, markerLength=0.12, cameraMatrix=intrinsic_matrix, distCoeffs=dist_coeffs)
        tvecs = tvecs[0]
        rvecs = rvecs[0]
        
        r_euler_angles = [rotationMatrixToEulerAngles(cv2.Rodrigues(rvec)[0].T) for rvec in rvecs]
        
        message = {
            "camera": {
                "position": {
                    "x": -tvecs[0][0],
                    "y": -tvecs[0][1],
                    "z": -tvecs[0][2],
                },
                "rotation": {
                    "x": r_euler_angles[0][0],
                    "y": r_euler_angles[0][1],
                    "z": r_euler_angles[0][2],
                }
            }
        }

        await websocket.send(json.dumps(message))

        # await asyncio.sleep(1/30)


async def main():
    server = await websockets.serve(position, "localhost", 5678)
    await server.wait_closed()


asyncio.run(main())
