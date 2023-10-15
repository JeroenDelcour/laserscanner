import asyncio
import json
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


async def sense(websocket):
    quat = Quaternion()
    while True:
        # Clear the buffer so as we can get fresh values
        # The sensor is running a lot faster than our sample period
        mpu.reset_FIFO()

        # wait for correct available data length, should be a VERY short wait
        FIFO_count = mpu.get_FIFO_count()
        while FIFO_count < packet_size:
            FIFO_count = mpu.get_FIFO_count()

        # read a packet from FIFO
        FIFO_buffer = mpu.get_FIFO_bytes(packet_size)
        quat = mpu.DMP_get_quaternion(FIFO_buffer)
        acc_raw = mpu.DMP_get_acceleration_int16(FIFO_buffer)
        quat_raw = mpu.DMP_get_quaternion_int16(FIFO_buffer)
        grav = mpu.DMP_get_gravity(quat)
        acceleration = mpu.DMP_get_linear_accel(acc_raw, grav)
        print(acceleration.z / 8192)

        # send message
        message = {
            "quaternion": {
                "x": quat.y,  # switch x, y, z values around so that Y is up
                "y": quat.z,
                "z": quat.x,
                "w": quat.w,
            },
            "acceleration": {
                "x": acceleration.y / 8192,
                "y": acceleration.z / 8192,
                "z": acceleration.x / 8192,
            },
        }
        # pprint(message)

        await websocket.send(json.dumps(message))


async def main():
    async with websockets.serve(sense, host=None, port=8765):
        print("Server is started")
        await asyncio.Future()


asyncio.run(main())
