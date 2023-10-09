import asyncio
import json
import math
import websockets
import time

from mpu6050 import mpu6050
from pyquaternion import Quaternion

sensor = mpu6050(0x68)

state = {
            "quaternion": Quaternion()
        }

async def sense(websocket):
    t_prev = time.time()
    while True:
        gyro_data = sensor.get_gyro_data()

        now = time.time()
        dt = now - t_prev
        t_prev = now

        gyro_data = {k: math.radians(v) for k, v in gyro_data.items()}
        angle = math.sqrt(gyro_data["x"] ** 2 + gyro_data["y"] ** 2 + gyro_data["z"] ** 2) * dt

        if angle != 0:
            gyro_data["x"] /= angle
            gyro_data["y"] /= angle
            gyro_data["z"] /= angle

        dq = Quaternion(axis=[gyro_data["x"], gyro_data["y"], gyro_data["z"]], angle=angle)
        state["quaternion"] = dq * state["quaternion"]

        print(state["quaternion"])

        message = {"quaternion":
                    {
                        "x": state["quaternion"][1],
                        "y": state["quaternion"][2],
                        "z": state["quaternion"][3],
                        "w": state["quaternion"][0],
                    }
                }

        await websocket.send(json.dumps(message))

async def main():
    async with websockets.serve(sense, host=None, port=8765):
        await asyncio.Future()

asyncio.run(main())
