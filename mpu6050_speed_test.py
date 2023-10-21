import smbus
import time


# class MPU6050:
#     def __init__(self, device_address=0x68):
#         self._device_address = device_address
#         self._bus = smbus.SMBus(1)
# 
#         #some MPU6050 Registers and their Address
#         self._PWR_MGMT_1   = 0x6B
#         self._SMPLRT_DIV   = 0x19
#         self._CONFIG       = 0x1A
#         self._GYRO_CONFIG  = 0x1B
#         self._INT_ENABLE   = 0x38
#         self._ACCEL_XOUT_H = 0x3B
#         self._ACCEL_YOUT_H = 0x3D
#         self._ACCEL_ZOUT_H = 0x3F
#         self._GYRO_XOUT_H  = 0x43
#         self._GYRO_YOUT_H  = 0x45
#         self._GYRO_ZOUT_H  = 0x47
# 
#         # set clock source to X gyro
#         self._write(self._PWR_MGMT_1, 1)
# 
#     def _write(self, register, value):
#         self._bus.write_byte_data(self._device_address, register, value)

PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47

def MPU_Init():
    #write to sample rate register
    bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
    
    #Write to power management register
    bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)
    
    #Write to Configuration register
    bus.write_byte_data(Device_Address, CONFIG, 0)
    
    #Write to Gyro configuration register
    bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)
    
    #Write to interrupt enable register
    bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
    #Accelero and Gyro value are 16-bit
    high = bus.read_byte_data(Device_Address, addr)
    low = bus.read_byte_data(Device_Address, addr+1)

    #concatenate higher and lower value
    value = ((high << 8) | low)
    
    #to get signed value from mpu6050
    if(value > 32768):
            value = value - 65536
    return value


bus = smbus.SMBus(1)    # or bus = smbus.SMBus(0) for older version boards
Device_Address = 0x68   # MPU6050 device address

MPU_Init()

print (" Reading Data of Gyroscope and Accelerometer")

DMP_packet_size = 42

t_prev = time.perf_counter_ns()
while True:
    
    # #Read Accelerometer raw value
    # acc_x = read_raw_data(ACCEL_XOUT_H)
    # acc_y = read_raw_data(ACCEL_YOUT_H)
    # acc_z = read_raw_data(ACCEL_ZOUT_H)
    # 
    # #Read Gyroscope raw value
    # gyro_x = read_raw_data(GYRO_XOUT_H)
    # gyro_y = read_raw_data(GYRO_YOUT_H)
    # gyro_z = read_raw_data(GYRO_ZOUT_H)
    # 
    # #Full scale range +/- 250 degree/C as per sensitivity scale factor
    # Ax = acc_x/16384.0
    # Ay = acc_y/16384.0
    # Az = acc_z/16384.0
    # 
    # Gx = gyro_x/131.0
    # Gy = gyro_y/131.0
    # Gz = gyro_z/131.0

    # bus.write_byte_data(Device_Address, 0x6A, 196)  # reset FIFO buffer
    # FIFO_count_H = bus.read_byte_data(Device_Address, 0x72)
    # FIFO_count_L = bus.read_byte_data(Device_Address, 0x73)
    # FIFO_count = (FIFO_count_H << 8) | FIFO_count_L
    # print(FIFO_count, FIFO_count / 42)

    for i in range(DMP_packet_size):
        bus.read_byte_data(Device_Address, 0x74)
    

    # print ("Gx=%.2f" %Gx, u'\u00b0'+ "/s", "\tGy=%.2f" %Gy, u'\u00b0'+ "/s", "\tGz=%.2f" %Gz, u'\u00b0'+ "/s", "\tAx=%.2f g" %Ax, "\tAy=%.2f g" %Ay, "\tAz=%.2f g" %Az)   
    # sleep(1)

    now = time.perf_counter_ns()
    dt = now - t_prev
    t_prev = now
    print(1/(dt * 1e-9), "Hz")
