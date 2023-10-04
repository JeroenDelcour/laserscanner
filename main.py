import numpy as np
import cv2
import ujson as json
import websockets
import asyncio

from laserscanner import LaserScanner


async def scan(websocket, path):
    cap = cv2.VideoCapture("http://raspberrypi.local:8000/stream.mjpg")
    laserscanner = LaserScanner()
    images = np.zeros(shape=(3, laserscanner.vertical_resolution,
                             laserscanner.horizontal_resolution, 3), dtype=np.uint8)
    image_exposures = ["", "", ""]
    images_idx = 0
    cv2.namedWindow("stream", cv2.WINDOW_NORMAL)
    while True:
        success, image = cap.read()
        if not success:
            break
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        images[images_idx] = image
        image_exposures[images_idx] = "long" if image[:,
                                                      laserscanner.horizontal_resolution // 2].mean() > 20 else "short"
        images_idx = (images_idx + 1) % len(images)

        # if image_exposures == ["long", "short", "long"]:
        laserscanner.update_pose(images[0])
        T0, R0, theta0 = laserscanner.T, laserscanner.R, laserscanner.theta
        points = laserscanner.scan(images[1])
        laserscanner.update_pose(images[2])
        T1, R1, theta1 = laserscanner.T, laserscanner.R, laserscanner.theta
        T = (T0 + T1) / 2
        R = (R0 + R1) / 2
        theta = (theta0 + theta1) / 2

        message = {
            "camera": {
                "position": {
                    "x": float(T[0]),
                    "y": float(T[1]),
                    "z": float(T[2]),
                },
                "rotation": {
                    "x": float(R[0]),
                    "y": float(R[1]),
                    "z": float(R[2]),
                    "angle": theta,
                }
            },
            "points": points.tolist()
        }
        await websocket.send(json.dumps(message))

        cv2.waitKey(1)


async def main():
    server = await websockets.serve(scan, "localhost", 5678)
    await server.wait_closed()

asyncio.run(main())
