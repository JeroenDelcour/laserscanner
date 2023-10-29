import asyncio
import json
import websockets
from pprint import pprint

from pyquaternion import Quaternion

import board
import busio
from adafruit_bno08x import (
    BNO_REPORT_LINEAR_ACCELERATION,
    BNO_REPORT_GAME_ROTATION_VECTOR,
)
from adafruit_bno08x.i2c import BNO08X_I2C

i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
bno = BNO08X_I2C(i2c)

bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)


async def sense(websocket):
    quat = Quaternion()
    while True:
        # read sensor
        acceleration = bno.linear_acceleration
        x, y, z, w = bno.game_quaternion
        quat = Quaternion(w, x, y, z)

        # rotate 90 degrees around X axis so Y is up
        dq = Quaternion(axis=[1, 0, 0], degrees=-90)
        quat = dq * quat

        # rotate acceleration vector from IMU frame to world frame
        acceleration = quat.rotate(acceleration)

        # send message
        message = {
            "quaternion": {
                "x": quat.x,
                "y": quat.y,
                "z": quat.z,
                "w": quat.w,
            },
            "acceleration": {
                "x": acceleration[0],
                "y": acceleration[1],
                "z": acceleration[2],
            },
        }
        # pprint(message)

        await websocket.send(json.dumps(message))


async def main():
    async with websockets.serve(sense, host=None, port=8765):
        print("Server is ready")
        await asyncio.Future()


asyncio.run(main())
