import argparse
from copy import copy

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, required=True, help="Video source (usually 0, 2, or 4)"
    )
    parser.add_argument(
        "--save_board",
        action="store_true",
        help="Save Charuco board to PNG file and exit",
    )
    parser.add_argument("--test", action="store_true", help="Test saved calibration")
    args = parser.parse_args()
    return args


def main(
    source,
    test=False,
    max_init_error=0.8,
    min_improvement_fraction=0.001,
    save_board=False,
):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard(
        size=(5, 7), squareLength=0.036, markerLength=0.027, dictionary=aruco_dict
    )
    board.setLegacyPattern(True)

    if save_board:
        # generate charuco board
        board_img = board.generateImage(outSize=(2000, 2000), marginSize=100)
        cv2.imwrite("charuco_board.png", board_img)
        return

    parameters = cv2.aruco.DetectorParameters()
    allCharucoCorners = []
    allCharucoIds = []
    allImagePoints = []
    allObjectPoints = []
    cameraMatrix = None
    distCoeffs = None
    last_error = None

    if test:
        cameraMatrix = np.loadtxt("cameraMatrix.txt")
        distCoeffs = np.loadtxt("distCoeffs.txt")

    try:
        source = int(source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(source)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set number of frames to buffer
    if not cap.isOpened():
        raise ValueError("Could not open camera stream")
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    frame_counter = 0

    while True:
        retval, frame = cap.read()
        if retval is False:
            continue
        frame_counter += 1
        undistorted = frame

        if not test:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                frame, dictionary=aruco_dict, parameters=parameters
            )
            if ids is not None:
                frame = cv2.aruco.drawDetectedMarkers(
                    image=frame, corners=corners, ids=ids
                )
                # OpenCV's tutorial recommends disabling corner interpolation when the camera has not been calibrated yet
                (
                    retval,
                    charucoCorners,
                    charucoIds,
                ) = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners, markerIds=ids, image=frame, board=board
                )
                frame = cv2.aruco.drawDetectedCornersCharuco(
                    image=frame, charucoCorners=charucoCorners, charucoIds=charucoIds
                )
                if charucoIds is not None and len(charucoIds) >= 6:
                    objPoints, imgPoints = board.matchImagePoints(
                        detectedCorners=charucoCorners, detectedIds=charucoIds
                    )

                    # estimate camera intrinsic parameters
                    (
                        error,
                        newCameraMatrix,
                        newDistCoeffs,
                        rvecs,
                        tvecs,
                    ) = cv2.calibrateCamera(
                        allObjectPoints + [objPoints],
                        allImagePoints + [imgPoints],
                        imageSize=frame.shape[:2],
                        cameraMatrix=copy(cameraMatrix),
                        distCoeffs=copy(distCoeffs),
                        flags=cv2.CALIB_FIX_INTRINSIC
                        | cv2.CALIB_FIX_ASPECT_RATIO
                        | cv2.CALIB_ZERO_TANGENT_DIST,
                    )
                    if (last_error is None and error < max_init_error) or (
                        last_error is not None
                        and error < last_error * (1 - min_improvement_fraction)
                    ):
                        last_error = error
                        allCharucoCorners.append(charucoCorners)
                        allCharucoIds.append(charucoIds)
                        allImagePoints.append(imgPoints)
                        allObjectPoints.append(objPoints)
                        cameraMatrix = newCameraMatrix
                        distCoeffs = newDistCoeffs

                    if cameraMatrix is not None:
                        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                            charucoCorners=charucoCorners,
                            charucoIds=charucoIds,
                            board=board,
                            cameraMatrix=cameraMatrix,
                            distCoeffs=distCoeffs,
                            rvec=None,
                            tvec=None,
                        )
                        frame = cv2.drawFrameAxes(
                            image=frame,
                            cameraMatrix=cameraMatrix,
                            distCoeffs=distCoeffs,
                            rvec=rvec,
                            tvec=tvec,
                            length=0.050,
                        )

            for i, text in enumerate(
                (
                    f"Calibration frames: {len(allImagePoints)}",
                    f"Projection error RMS: {last_error}",
                )
            ):
                frame = cv2.putText(
                    img=frame,
                    text=text,
                    org=(50, 50 + i * 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            print(len(allCharucoCorners), last_error)

        if cameraMatrix is not None:
            h, w = frame.shape[:2]
            # newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
            #     cameraMatrix, distCoeffs, (w, h), 1, (w, h)
            # )
            undistorted = cv2.undistort(frame, cameraMatrix, distCoeffs)
            # add black border to crop the image
            # x, y, w, h = roi
            # undistorted[:y, :x] = 0
            # undistorted[y+h:, :x] = 0
            # undistorted[:y, x+w:] = 0
            # undistorted[y+h:, x+w:] = 0

        vis = np.hstack((frame, undistorted))

        cv2.imshow("vis", vis)
        if cv2.waitKey(1) in (27, ord("q")):  # ESC or Q key pressed
            if not test:
                np.savetxt("cameraMatrix.txt", cameraMatrix)
                np.savetxt("distCoeffs.txt", distCoeffs)
            break


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
