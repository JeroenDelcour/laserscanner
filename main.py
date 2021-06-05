import numpy as np
import cv2
import ujson as json
import websockets
import asyncio

from laserscanner import LaserScanner, rodrigues_rotation


async def scan(websocket, path):
    cap = cv2.VideoCapture("http://raspberrypi.local:8000/stream.mjpg")
    laserscanner = LaserScanner()
    images = np.zeros(shape=(3, laserscanner.vertical_resolution,
                             laserscanner.horizontal_resolution, 3), dtype=np.uint8)
    image_exposures = ["", "", ""]
    images_idx = 0

    cv2.namedWindow("", cv2.WINDOW_NORMAL)

    while True:
        success, image = cap.read()
        if not success:
            break
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        images[images_idx] = image
        image_exposures[images_idx] = "long" if image[:,
                                                      laserscanner.horizontal_resolution // 2].mean() > 20 else "short"
        images_idx = (images_idx + 1) % len(images)

        if image_exposures == ["long", "short", "long"]:
            laserscanner.update_pose(images[0])
            tvec0, rvec0 = laserscanner._tvec, laserscanner._rvec

            laser_screenpoints = laserscanner.find_laser(images[1])

            laserscanner.update_pose(images[2])
            tvec1, rvec1 = laserscanner._tvec, laserscanner._rvec

            tvec = (tvec0 + tvec1) / 2
            rvec = (rvec0 + rvec1) / 2

            fx = fy = 1355
            cx, cy = laserscanner.horizontal_resolution / 2, laserscanner.vertical_resolution / 2
            points = np.zeros(shape=(len(laser_screenpoints), 3))
            points[:, :2] = laser_screenpoints  # - np.array([cx, cy])
            # points[:, :2] /= np.array([fx, fy])
            points[:, 2] = 1
            points = points.T
            print(points[:, int(cx)])

            # image frame to camera frame
            camera_points = np.linalg.inv(laserscanner.camera_matrix) @ points
            # camera_points2 = np.ones(shape=(3, points.shape[1]), dtype=np.float32)
            # camera_points2[:2, :] = ((points[:2].T - np.array([cx, cy])) / np.array([fx, fy])).T
            # camera_points = camera_points2
            print(camera_points[:, int(cx)])

            # # camera frame to world frame

            # rot_matrix, _ = cv2.Rodrigues(rvec)
            # # transform = np.hstack([rot_matrix, tvec])
            # # print(transform)
            # points = rot_matrix.T @ points - rot_matrix.T @ tvec
            # T = rot_matrix.T @ tvec
            # Tx, Ty, Tz = T.squeeze()
            # print(points.shape)
            # dx, dy, dz = points
            # X = (-Tz / dz) * dx + Tx
            # Y = (-Tz / dz) * dy + Ty
            # print(X[int(cx)], Y[int(cx)])
            # points = points.T
            # print(points[:, int(cx)])

            # points *= laserscanner.pixel_size
            # print(points[500])

            # define calibration plane in camera frame
            rot_matrix, _ = cv2.Rodrigues(rvec)
            points_w = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
            points_c = rot_matrix.T @ points_w - rot_matrix.T @ tvec
            d = 1
            D = np.linalg.det(points_c)
            # a = -d / D * np.hstack([np.ones([3,1]), D[:,[1,2]]])
            # b = -d / D * np.hstack([D[:, 0], np.ones([3, 1]), D[:, 2]])
            # c = -d / D * np.hstack([D[:, [2, 3]], np.ones([3, 1])])
            a = -d / D * np.linalg.det(np.array([[1, points_c[0, 1], points_c[0, 2]],
                                                 [1, points_c[1, 1], points_c[1, 2]],
                                                 [1, points_c[2, 1], points_c[2, 2]]]))
            b = -d / D * np.linalg.det(np.array([[points_c[0, 0], 1, points_c[0, 2]],
                                                 [points_c[1, 0], 1, points_c[1, 2]],
                                                 [points_c[2, 0], 1, points_c[2, 2]]]))
            c = -d / D * np.linalg.det(np.array([[points_c[0, 0], points_c[0, 1], 1],
                                                 [points_c[1, 0], points_c[1, 1], 1],
                                                 [points_c[2, 0], points_c[2, 1], 1]]))

            # intersect rays with calibration plane
            s = -d / (a*camera_points[0, :] + b*camera_points[1, :] + c)  # scale factor
            print(s.shape)
            world_points = s * camera_points
            print(world_points[:, int(cx)])

            print()

            T = tvec[:, 0]
            R = -rvec[:, 0]
            theta = np.linalg.norm(rvec)
            R = R / theta
            T = rodrigues_rotation(T, R, theta)

            # object_points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float32)
            # rot_matrix = cv2.Rodrigues(rvec)
            # object_points_in_camera_frame = rot_matrix @ object_points + tvec
            # image_points, jacobian = cv2.projectPoints(
            #     object_points_in_camera_frame, rvec, tvec, laserscanner.camera_matrix, laserscanner.distortion_coefficients)

            # vis = image[1].copy()
            # for image_point in image_points:
            #     vis = cv2.drawMarker(vis, image_point, (0, 255, 0))
            # cv2.imshow("", vis)

            # M = cv2.getPerspectiveTransform(image_points, object_points_in_camera_frame)

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
                        "angle": float(theta),
                    }
                },
                "points": laserscanner.points.tolist()
            }
            await websocket.send(json.dumps(message))


async def main():
    server = await websockets.serve(scan, "localhost", 5678)
    await server.wait_closed()

asyncio.run(main())
