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

while True:
    success, image = cap.read()
    if not success:
        break

    if np.median(image) < 10:  # short exposure image
        gray = image[:, :, 2].astype(float) - np.mean(image[:, :, :2], axis=-1)
        gray -= gray.min()
        gray /= gray.max() / 255
        gray = np.clip(gray, 0, 255)
        gray = gray.astype(np.uint8)
        image[:, :, 0] = gray
        image[:, :, 1] = gray
        image[:, :, 2] = gray
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
    key = cv2.waitKey(1)
    if key == 27:  # ESC pressed
        break

shifts = np.array(shifts)
shifts = shifts.astype(float)
shifts[shifts == 0] = np.nan
shifts -= 0.5*720  # measure from optical center
shifts *= -1  # flip
shifts *= 0.001121951  # pixels to mm

# def line(p1, p2):
#     A = (p1[1] - p2[1])
#     B = (p2[0] - p1[0])
#     C = (p1[0]*p2[1] - p2[0]*p1[1])
#     return A, B, -C

# def intersection(L1, L2):
#     D  = L1[0] * L2[1] - L1[1] * L2[0]
#     Dx = L1[2] * L2[1] - L1[1] * L2[2]
#     Dy = L1[0] * L2[2] - L1[2] * L2[0]
#     if D != 0:
#         x = Dx / D
#         y = Dy / D
#         return x,y
#     else:
#         return False

# laser_origin = [-baseline, 0]
# laser_vector = [-baseline + np.cos(angle), 0 + np.sin(angle)]
# camera_origin = [0, 0]
# depths = []
# laser_line = line(laser_origin, laser_vector)
# for shift in shifts:
#     d = []
#     for y in shift:
#         shift_vector = [y, focal_length]
#         camera_line = line(camera_origin, shift_vector)
#         x, y = intersection(laser_line, camera_line)
#         depth = y
#         d.append(depth)
#         if not depth:
#             d[-1] = np.nan
#     depths.append(d)
# depths = np.array(depths)

angle = np.deg2rad(70)
baseline = 118.478  # mm
focal_length = 3.04  # mm
z = (baseline * focal_length * np.tan(angle)) / (focal_length - shifts * np.tan(angle))
y = z * shifts / focal_length  # similar triangles
x = np.zeros_like(shifts)
x[:] = np.arange(shifts.shape[1]) - 0.5 * shifts.shape[1]  # in pixels
x *= 0.001121951  # pixels to mm
# x = np.arange(len(shifts)) - 0.5 * len(shifts)  # in pixels
x = z * x / focal_length
depths = z

# assume camera moves steadily along Y axis
for i in range(len(y)):
    y[i] -= i


plt.imshow(depths, cmap="magma")
plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# X = range(shifts.shape[1])
# Y = range(shifts.shape[0])
# X, Y = np.meshgrid(X, Y)
# Z = shifts
# ax.plot_surface(X, Y, Z, antialiased=False)

# plot a random sample to keep framerate managable
every = 20
ax.scatter(x[:, ::every].flatten(), y[:, ::every].flatten(), -z[:, ::every].flatten(), c=-z[:, ::every].flatten())
plt.show()
#
