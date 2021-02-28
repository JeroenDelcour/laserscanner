import numpy as np
import cv2
from cv2 import aruco
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


def _find_board(image, charucoboard_dict, board):
    # aruco detection parameters tuned for small markers
    parameters = cv2.aruco.DetectorParameters_create()
    # parameters.errorCorrectionRate = 1  # default is 0.6
    # parameters.adaptiveThreshWinSizeMin = 3
    # parameters.adaptiveThreshWinSizeMax = 23
    # parameters.adaptiveThreshWinSizeStep = 10
    # parameters.adaptiveThreshConstant = 7
    # parameters.minMarkerPerimeterRate = 0.01  # default is 0.03
    # parameters.maxMarkerPerimeterRate = 4.0
    # parameters.minCornerDistanceRate = 0.05
    corners, ids, rejected = cv2.aruco.detectMarkers(image, charucoboard_dict, parameters=parameters)
    corners, ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
        image, board, corners, ids, rejected
    )
    if not corners:
        return None, None
    retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
        corners, ids, image, board
    )
    return charucoCorners, charucoIds


charucoboard_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
squares_x = 11
squares_y = 8
square_length = 22  # mm
marker_length = 18  # mm
board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, charucoboard_dict)

width = 1280
fx = fy = width
cx = width / 2
cy = width / 2
intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
distortion_coefficients = np.zeros(5)

cap = cv2.VideoCapture("video.h264")

shifts = []

for i in range(66):
    success, image = cap.read()
while True:
    success, image = cap.read()
    if not success:
        break

    if np.median(image) < 10:  # short exposure image
        gray = image[:,:,2].astype(float) - np.mean(image[:,:,:2], axis=-1)
        gray -= gray.min()
        gray /= gray.max() / 255
        gray = np.clip(gray, 0, 255)
        gray = gray.astype(np.uint8)
        image[:,:,0] = gray
        image[:,:,1] = gray
        image[:,:,2] = gray
        frame_shifts = np.argmax(gray, axis=0)
        for i in range(len(frame_shifts)):
            y = frame_shifts[i]
            if image[y, i, 2] < 10:
                frame_shifts[i] = 0
                continue
            image[y, i] = (0, 255, 0)
        shifts.append(frame_shifts)
    else:  # long exposure image
        charucoCorners, charucoIds = _find_board(image, charucoboard_dict, board)
        if charucoCorners is not None:
            cv2.aruco.drawDetectedCornersCharuco(image, charucoCorners, charucoIds)
            rvec, tvec = None, None
            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charucoCorners, charucoIds, board, intrinsic_matrix, distortion_coefficients, rvec, tvec
            )
            if valid:
                print(tvec, rvec)

    cv2.imshow("", image)
    # key = cv2.waitKey(1000//60)
    key = cv2.waitKey(0)
    if key == 27:  # ESC pressed
        break

shifts = np.array(shifts)
shifts = shifts.astype(float)
shifts[shifts == 0] = np.nan

plt.imshow(shifts, cmap="magma")
plt.show()

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# X = range(shifts.shape[1])
# Y = range(shifts.shape[0])
# X, Y = np.meshgrid(X, Y)
# Z = shifts
# ax.plot_surface(X, Y, Z, antialiased=False)
# plt.show()
