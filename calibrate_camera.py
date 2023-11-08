import argparse

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
    args = parser.parse_args()
    return args


def main(source, max_init_error=0.5, min_improvement_fraction=0.01, save_board=False):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard(
        size=(5, 7), squareLength=0.036, markerLength=0.027, dictionary=aruco_dict
    )

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

    try:
        source = int(source)
    except ValueError:
        pass
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError("Could not open camera stream")
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    while True:
        retval, frame = cap.read()

        corners, ids, rejected = cv2.aruco.detectMarkers(
            frame, dictionary=aruco_dict, parameters=parameters
        )
        if ids is not None:
            frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids)
            # OpenCV's tutorial recommends disabling corner interpolation when the camera has not been calibrated yet
            retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
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
                error, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
                    allObjectPoints + [objPoints],
                    allImagePoints + [imgPoints],
                    imageSize=frame.shape[:2],
                    cameraMatrix=cameraMatrix,
                    distCoeffs=distCoeffs,
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

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) in (27, ord("q")):  # ESC or Q key pressed
            np.savetxt("cameraMatrix.txt", cameraMatrix)
            np.savetxt("distCoeffs.txt", distCoeffs)
            break


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
