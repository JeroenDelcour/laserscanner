import numpy as np
import cv2
import cv2.aruco


def rodrigues_rotation(v, k, theta):
    return v*np.cos(theta) + (np.cross(k, v)*np.sin(theta)) + k*(np.dot(k, v))*(1.0-np.cos(theta))


class LaserScanner:
    def __init__(self):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
        self.charuco_board = cv2.aruco.CharucoBoard_create(11, 8, 0.015, 0.01125, self.aruco_dict)

        self.laser_angle = np.deg2rad(75)
        self.baseline = 118.478e-3  # meters
        self.focal_length = 3.04e-3  # meters
        self.vertical_resolution = 1640
        self.horizontal_resolution = 1232
        self.pixel_size = 3.68e-3 / self.horizontal_resolution
        fx = fy = 1355
        cx, cy = 0.5 * self.horizontal_resolution, 0.5 * self.vertical_resolution
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.distortion_coefficients = np.array([0.22128342058032355,
                                                 -0.5663376863990286,
                                                 -0.0001804474513748153,
                                                 -0.001201953225667692,
                                                 0.2602535953452802], dtype=np.float32)
        self.position_lock = False

        # pre-allocate memory
        self.points = np.zeros(shape=(self.horizontal_resolution, 3))
        self._points_arange = np.arange(len(self.points))
        self._screen_points = np.zeros(shape=(len(self.points), 2), dtype=np.float32)
        self._screen_points[:, 0] = self._points_arange
        self._tvec = np.zeros(shape=(3, 1), dtype=np.float32)
        self._rvec = np.zeros(shape=(3, 1), dtype=np.float32)

        self.T = np.zeros(3)
        self.R = np.zeros(3)
        self.theta = 0

    def update_pose(self, image, resize=0.5):
        """ Update pose (self.T and self.R) from charuco board in image. Returns True if successfull, False otherwise. """
        corners, ids, rejected = cv2.aruco.detectMarkers(cv2.resize(
            image, dsize=None, fx=resize, fy=resize), self.aruco_dict)
        corners = [c / resize for c in corners]
        corners, ids, rejected, recovered = cv2.aruco.refineDetectedMarkers(
            image, self.charuco_board, corners, ids, rejected, cameraMatrix=self.camera_matrix, distCoeffs=self.distortion_coefficients)
        if not corners:
            self.position_lock = False
            return False
        retval, corners, ids = cv2.aruco.interpolateCornersCharuco(corners, ids, image, self.charuco_board)
        success, _, _ = cv2.aruco.estimatePoseCharucoBoard(
            corners, ids, self.charuco_board, self.camera_matrix, self.distortion_coefficients, rvec=self._rvec, tvec=self._tvec, useExtrinsicGuess=self.position_lock)
        if not success:
            self.position_lock = False
            return False
        # invert
        self.T = self._tvec[:, 0]
        self.R = -self._rvec[:, 0]
        self.theta = np.linalg.norm(self.R)
        self.R = self.R / self.theta
        self.T = rodrigues_rotation(self.T, self.R, self.theta)

        self.position_lock = True
        return True

    def find_laser(self, image, lower_threshold=10):
        gray = image[:, :, 2]
        shifts = np.argmax(gray, axis=0)
        # filter out too dark segments
        shifts[gray[shifts, self._points_arange] < lower_threshold] = -1
        return shifts

    def scan(self, image):
        """ Scan laser in image and return 3D points relative to camera. """
        shifts = self.find_laser(image).astype(np.float32)
        shifts[shifts == -1] = np.nan
        self._screen_points[:, 1] = shifts
        undistorted_screenpoints = cv2.undistortPoints(
            self._screen_points[np.newaxis, :], self.camera_matrix, self.distortion_coefficients, P=self.camera_matrix).squeeze()
        shifts[:] = undistorted_screenpoints[:, 1]
        shifts -= 0.5 * self.vertical_resolution  # measure from optical center
        # shifts *= -1  # flip
        shifts *= self.pixel_size  # pixels to meters
        z = self.points[:, 2] = (self.baseline * self.focal_length * np.tan(self.laser_angle)
                                 ) / (self.focal_length - shifts * np.tan(self.laser_angle))
        y = self.points[:, 1] = z * shifts / self.focal_length  # similar triangles
        x = self.points[:, 0]
        x[:] = undistorted_screenpoints[:, 0] - 0.5 * len(shifts)  # in pixels
        x[:] *= self.pixel_size  # pixels to meters
        x[:] = z * x / self.focal_length
        self.points[np.isnan(shifts), :] = -1
        return self.points
