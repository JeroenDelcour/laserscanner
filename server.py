import numpy as np
import asyncio
import random
import websockets
import ujson as json
import cv2
import cv2.aruco
import math
from pprint import pprint


ANGLE = np.deg2rad(75)
BASELINE = 118.478e-3  # meters
FOCAL_LENGTH = 3.04e-3  # meters
# PIXEL_SIZE = 2 * 1.12e-6  # meters
# VERTICAL_RESOLUTION = 1232
# HORIZONTAL_RESOLUTION = 1640
VERTICAL_RESOLUTION = 1232
HORIZONTAL_RESOLUTION = 1640
PIXEL_SIZE = 3.68e-3 / HORIZONTAL_RESOLUTION
fx = fy = 1355
cx, cy = HORIZONTAL_RESOLUTION // 2, VERTICAL_RESOLUTION // 2
INTRINSIC_MATRIX = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
DIST_COEFFS = np.array([0.22128342058032355,
                        -0.5663376863990286,
                        -0.0001804474513748153,
                        -0.001201953225667692,
                        0.2602535953452802], dtype=np.float32)


points = np.zeros(shape=(HORIZONTAL_RESOLUTION, 3))
_points_arange = np.arange(len(points))
screen_points = np.zeros(shape=(len(points), 2), dtype=np.float32)
screen_points[:, 0] = _points_arange

aruco_parameters = cv2.aruco.DetectorParameters_create()
aruco_parameters.errorCorrectionRate = 0  # default is 0.6
aruco_parameters.maxErroneousBitsInBorderRate = 0  # default is 0.35
# aruco_parameters.perspectiveRemovePixelPerCell = 4  # default is 4
# aruco_parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13  # default is 0.13
# aruco_parameters.adaptiveThreshWinSizeMin = 3  # default is 3
# aruco_parameters.adaptiveThreshWinSizeMax = 23  # default is 23
# aruco_parameters.adaptiveThreshWinSizeStep = 10
# aruco_parameters.adaptiveThreshConstant = 7
aruco_parameters.minMarkerPerimeterRate = 0.03  # default is 0.03, 0.15 works well
# aruco_parameters.maxMarkerPerimeterRate = 4.0
# aruco_parameters.minCornerDistanceRate = 0.05
# aruco_parameters.polygonalApproxAccuracyRate = 0.05  # default is 0.05


def detect_laser(image, lower_threshold=10):
    # image = image.astype(float)
    # gray = image[:, :, 2] - 0.5 * (image[:, :, 0] + image[:, :, 1])
    gray = image[:, :, 2]
    # gray -= np.mean(image[:, :, :2], axis=-1)
    # gray -= gray.min()
    # gray /= gray.max() / 255
    # gray = np.clip(gray, 0, 255)
    # gray = gray.astype(np.uint8)
    shifts = np.argmax(gray, axis=0)
    # filter out too short segments
    # d = np.abs(shifts[1:] - shifts[:-1])
    # shifts[1:][d > 5] = -1
    # filter out too dark segments
    shifts[gray[shifts, _points_arange] < lower_threshold] = -1
    return shifts


def get_laserpoints(shifts):
    shifts = shifts.astype(float)
    shifts[shifts == -1] = np.nan
    screen_points[:, 1] = shifts
    undistorted_screen_points = cv2.undistortPoints(
        screen_points[np.newaxis, :], INTRINSIC_MATRIX, DIST_COEFFS, P=INTRINSIC_MATRIX)
    undistorted_screen_points = undistorted_screen_points.squeeze()
    shifts[:] = undistorted_screen_points[:, 1]
    shifts -= 0.5*VERTICAL_RESOLUTION  # measure from optical center
    shifts *= -1  # flip
    shifts *= PIXEL_SIZE  # pixels to meters
    z = points[:, 2] = (BASELINE * FOCAL_LENGTH * np.tan(ANGLE)) / (FOCAL_LENGTH - shifts * np.tan(ANGLE))
    y = points[:, 1] = z * shifts / FOCAL_LENGTH  # similar triangles
    x = points[:, 0]
    x[:] = undistorted_screen_points[:, 0] - 0.5 * len(shifts)  # in pixels
    x[:] *= PIXEL_SIZE  # pixels to meters
    x[:] = z * x / FOCAL_LENGTH
    points[np.isnan(shifts), :] = -1
    return points


def rodrigues_rotation(v, k, theta):
    return v*np.cos(theta) + (np.cross(k, v)*np.sin(theta)) + k*(np.dot(k, v))*(1.0-np.cos(theta))


@profile
async def position(websocket, path):
    # cap = cv2.VideoCapture(2)
    # cap.set(3, HORIZONTAL_RESOLUTION)  # width
    # cap.set(4, VERTICAL_RESOLUTION)  # height
    # cap.set(5, 30)  # framerate
    cap = cv2.VideoCapture("http://raspberrypi.local:8000/stream.mjpg")

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard_create(11, 8, 0.015, 0.01125, aruco_dict)

    # fx = fy = width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # cx = width / 2
    # cy = width / 2
    # intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # dist_coeffs = np.zeros(5)

    cv2.namedWindow("", cv2.WINDOW_NORMAL)

    prev_rvec, prev_tvec = None, None
    success, image = cap.read()
    vis = np.zeros_like(image)
    T = [0., 0., 0.]
    Rnorm = [1., 0., 0.]
    theta = 0.
    points = np.array([0.]*HORIZONTAL_RESOLUTION)
    while True:
        success, image = cap.read()
        if not success:
            break
        vis[:] = image

        sample = image[:, HORIZONTAL_RESOLUTION//2].mean()
        if sample < 20:
            shifts = detect_laser(image)
            vis[shifts, np.arange(image.shape[1])] = (0, 255, 0)
            points = get_laserpoints(shifts)
        # points = np.zeros(shape=(len(shifts), 3))
        # points[:, 0] = np.arange(len(shifts))  # x
        # points[:, 2] = 0.3  # z (depth)

        else:
            scale = 0.5
            corners, ids, rejected = cv2.aruco.detectMarkers(
                cv2.resize(image, dsize=None, fx=scale, fy=scale), aruco_dict)
            corners = [c / scale for c in corners]
            # image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
            corners, ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
                image, board, corners, ids, rejected, cameraMatrix=INTRINSIC_MATRIX, distCoeffs=DIST_COEFFS
            )
            if corners:
                retval, corners, ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, image, board
                )
                vis = cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids)

                # rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
                #     corners, markerLength=0.05, cameraMatrix=intrinsic_matrix, distCoeffs=dist_coeffs)
                # for rvec, tvec, id_ in zip(rvecs, tvecs, ids):
                #     if id_ == 77:
                #         image = cv2.aruco.drawAxis(image, intrinsic_matrix, dist_coeffs, rvec, tvec, 0.1)

                success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    corners, ids, board, INTRINSIC_MATRIX, DIST_COEFFS, rvec=prev_rvec, tvec=prev_tvec, useExtrinsicGuess=prev_tvec is not None)

                # gray = image[:, :, 2].astype(float) - np.mean(image[:, :, :2], axis=-1)
                # gray -= gray.min()
                # gray /= gray.max() / 255
                # gray = np.clip(gray, 0, 255)
                # gray = gray.astype(np.uint8)
                # vis = np.hstack([vis, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)])

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
            },
            "points": points.tolist()
        }
        message = json.dumps(message)
        await websocket.send(message)

        # cv2.imshow("", vis)
        # cv2.waitKey(1)

        # await asyncio.sleep(1/30)


async def main():
    server = await websockets.serve(position, "localhost", 5678)
    await server.wait_closed()


asyncio.run(main())
