import asyncio
import cv2
import time
import json
import websockets
import numpy as np

from pyquaternion import Quaternion
from picamera2 import Picamera2

from MPU6050 import MPU6050

class IMU:
    def __init__(self):
        i2c_bus = 1
        device_address = 0x68
        # The offsets are different for each device and should be changed
        # accordingly using a calibration procedure
        x_accel_offset = -2308
        y_accel_offset = -1525
        z_accel_offset = 2565
        x_gyro_offset = 36
        y_gyro_offset = 19
        z_gyro_offset = -23
        self.mpu = MPU6050(
            i2c_bus,
            device_address,
            x_accel_offset,
            y_accel_offset,
            z_accel_offset,
            x_gyro_offset,
            y_gyro_offset,
            z_gyro_offset,
            0,
        )

        self.mpu.dmp_initialize()
        self.mpu.set_DMP_enabled(True)
        self.packet_size = self.mpu.DMP_get_FIFO_packet_size()

    def read(self):
        """
        Reads quaternion and acceleration vector from the IMU.
        """
        self.mpu.reset_FIFO()

        # wait for correct available data length, should be a VERY short wait
        FIFO_count = self.mpu.get_FIFO_count()
        while FIFO_count < self.packet_size:
            FIFO_count = self.mpu.get_FIFO_count()

        # read a packet from FIFO
        FIFO_buffer = self.mpu.get_FIFO_bytes(self.packet_size)
        quat = self.mpu.DMP_get_quaternion(FIFO_buffer)
        acc_raw = self.mpu.DMP_get_acceleration_int16(FIFO_buffer)
        quat_raw = self.mpu.DMP_get_quaternion_int16(FIFO_buffer)
        grav = self.mpu.DMP_get_gravity(quat)
        acceleration = self.mpu.DMP_get_linear_accel(acc_raw, grav)
        quat = Quaternion(w=quat.w, x=quat.x, y=quat.y, z=quat.z)
        acceleration = np.array([acceleration.x, acceleration.y, acceleration.z])

        # scale acceleration to meters per second squared
        acceleration /= 8192.0

        # transform from IMU to camera frame
        imu2cam = Quaternion(axis=[0, 0, 1], degrees=180)
        acceleration = imu2cam.rotate(acceleration)
        # rotate 90 degrees around X axis so Y is up
        y_up = Quaternion(axis=[1, 0, 0], degrees=-90)
        quat = y_up * quat * imu2cam
        
        return quat, acceleration


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

    imu = IMU()

    cam_quat = Quaternion()
    tvec = np.zeros(shape=(3,1))

    imu_correction = Quaternion()

    ###
    # x = x + v*dt + alpha*dk
    # v = v + a*dt +  beta*dk
    ###

    q_correction = Quaternion()

    while True:
        imu_quat, imu_acceleration = imu.read()
        imu_quat = q_correction * imu_quat

        # rotate acceleration vector from IMU frame to world frame
        imu_acceleration = imu_quat.rotate(imu_acceleration)

        image = picam2.capture_array("main")

        corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary=aruco_dict)
        if len(corners) > 0:

            retval, corners, ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, image, charuco_board
            )
            success, rvec, new_tvec = cv2.aruco.estimatePoseCharucoBoard(
                corners,
                ids,
                charuco_board,
                camera_matrix,
                distortion_coefficients,
                rvec=None,
                tvec=None,
                useExtrinsicGuess=False,
            )
            if success:
                # tvec and quat of the board in the camera frame
                tvec = new_tvec
                angle = np.linalg.norm(rvec[:,0])
                axis = rvec[:,0] / angle
                cam_quat = Quaternion(axis=axis, radians=angle)

                # invert so we have the camera from the board's frame
                cam_quat = cam_quat.inverse
                tvec = -tvec

                # rotate 90 degrees around X axis so Y is up instead of Z
                dq = Quaternion(axis=(1, 0, 0), degrees=-90)
                cam_quat = dq * cam_quat
                tvec[:,0] = cam_quat.rotate(tvec[:,0])

                # orientation correction
                alpha = 0.5
                dq = cam_quat * imu_quat.inverse
                q_correction = Quaternion(axis=dq.axis, angle=alpha * dq.angle) * q_correction

        message = {
            "cam_quaternion": {
                "x": cam_quat.x,
                "y": cam_quat.y,
                "z": cam_quat.z,
                "w": cam_quat.w,
            },
            "imu_quaternion": {
                "x": imu_quat.x,
                "y": imu_quat.y,
                "z": imu_quat.z,
                "w": imu_quat.w,
            },
            "translation": {
                "x": tvec[0, 0],
                "y": tvec[1, 0],
                "z": tvec[2, 0],
            },
            "acceleration": {
                "x": imu_acceleration[0],
                "y": imu_acceleration[1],
                "z": imu_acceleration[2],
            }
        }

        await websocket.send(json.dumps(message))


async def main():
    async with websockets.serve(sense, host=None, port=8765):
        print("Server is ready")
        await asyncio.Future()


asyncio.run(main())
