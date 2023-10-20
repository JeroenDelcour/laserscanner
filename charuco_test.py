import asyncio
import cv2
import time
import json
import websockets
import numpy as np

from pyquaternion import Quaternion
from picamera2 import Picamera2

async def sense(websocket):
    focal_length = 3.04e-3  # meters
    vertical_resolution = 800
    horizontal_resolution = 600
    fx = fy = 1355 / 2
    cx, cy = 0.5 * horizontal_resolution, 0.5 * vertical_resolution
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    distortion_coefficients = np.array(
        [
            0.22128342058032355,
            -0.5663376863990286,
            -0.0001804474513748153,
            -0.001201953225667692,
            0.2602535953452802,
        ],
        dtype=np.float32,
    )


    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    charuco_board = cv2.aruco.CharucoBoard_create(
        squaresX=5, squaresY=7, squareLength=0.015, markerLength=0.011, dictionary=aruco_dict
    )

    # image = charuco_board.generateImage(outSize=(600, 500), marginSize=40, borderBits=1)
    # cv2.imwrite("charuco_board.png", image)

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (800, 600), "format": "BGR888"})
    # config = picam2.create_preview_configuration({"size": (800, 600), "format": "BGR888"})
    picam2.align_configuration(config)
    print(config["main"])
    picam2.configure(config)

    picam2.start()

    while True:
        image = picam2.capture_array("main")

        corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary=aruco_dict)
        if len(corners) == 0:
            continue

        retval, corners, ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, image, charuco_board
        )
        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            corners,
            ids,
            charuco_board,
            camera_matrix,
            distortion_coefficients,
            rvec=None,
            tvec=None,
            useExtrinsicGuess=False,
        )
        if not success:
            continue
        
        # tvec and quat of the board in the camera frame
        angle = np.linalg.norm(rvec[:,0])
        axis = rvec[:,0] / angle
        quat = Quaternion(axis=axis, radians=angle)

        # invert so we have the camera from the board's frame
        quat = quat.inverse
        tvec = -tvec

        # rotate 90 degrees around X axis so Y is up instead of Z
        dq = Quaternion(axis=(1, 0, 0), degrees=90)
        quat = dq.inverse * quat
        tvec[:,0] = dq.rotate(tvec[:,0])


        message = {
            "quaternion": {
                "x": quat.x,
                "y": quat.y,
                "z": quat.z,
                "w": quat.w,
            },
            "translation": {
                "x": tvec[0, 0],
                "y": tvec[1, 0],
                "z": tvec[2, 0],
            },
        }

        await websocket.send(json.dumps(message))


async def main():
    async with websockets.serve(sense, host=None, port=8765):
        print("Server is ready")
        await asyncio.Future()


asyncio.run(main())
