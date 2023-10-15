import asyncio
import json
import math
import websockets
from pprint import pprint

from MPU6050 import MPU6050
from MPU6050.Quaternion import Quaternion

i2c_bus = 1
device_address = 0x68
# The offsets are different for each device and should be changed
# accordingly using a calibration procedure
x_accel_offset = -1874
y_accel_offset = 255
z_accel_offset = 1046
x_gyro_offset = 46
y_gyro_offset = 26
z_gyro_offset = -27
mpu = MPU6050(
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

mpu.dmp_initialize()
mpu.set_DMP_enabled(True)
packet_size = mpu.DMP_get_FIFO_packet_size()
FIFO_buffer = [0] * 64
FIFO_count_list = list()


async def sense(websocket):
    quat = Quaternion()
    while True:
        FIFO_count = mpu.get_FIFO_count()
        mpu_int_status = mpu.get_int_status()

        # If overflow is detected by status or fifo count we want to reset
        if (FIFO_count == 1024) or (mpu_int_status & 0x10):
            mpu.reset_FIFO()
            # print("overflow!")
        # Check if fifo data is ready
        elif mpu_int_status & 0x02:
            # Wait until packet_size number of bytes are ready for reading, default
            # is 42 bytes
            while FIFO_count < packet_size:
                FIFO_count = mpu.get_FIFO_count()
            FIFO_buffer = mpu.get_FIFO_bytes(packet_size)
            quat = mpu.DMP_get_quaternion(FIFO_buffer)

            message = {
                "quaternion": {
                    "x": quat.y,  # switch x, y, z values around so that Y is up
                    "y": quat.z,
                    "z": quat.x,
                    "w": quat.w,
                },
            }
            pprint(message)

            await websocket.send(json.dumps(message))


async def main():
    async with websockets.serve(sense, host=None, port=8765):
        print("Server is started")
        await asyncio.Future()


asyncio.run(main())
