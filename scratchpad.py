import cv2
import numpy as np
from copy import deepcopy

cap = cv2.VideoCapture(2)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height
cap.set(5, 90)  # framerate

vis = np.zeros(shape=(720, 1280*3, 3), dtype=np.uint8)
out0 = vis[:, 1280:1280*2]
out1 = vis[:, 1280*2:]
# kernel = np.ones((3,3), dtype=np.float32)
# kernel /= kernel.sum()
template = cv2.getGaussianKernel(ksize=21, sigma=4).astype(np.float32)
# template = np.hstack([template, template])
gray_float = np.zeros(shape=(720, 1280), dtype=np.float32)
res = None
print(template)
while True:
    success, image = cap.read()
    vis[:, :1280] = image
    gray_float[:] = image[:, :, 2] - np.mean(image[:, :, :2], axis=-1, dtype=np.float32)
    gray = deepcopy(gray_float)
    gray -= gray.min()
    gray /= gray.max() / 255
    gray = np.clip(gray, 0, 255)
    gray = gray.astype(np.uint8)
    out0[:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if res is None:
        res = cv2.matchTemplate(gray_float, templ=template, method=cv2.TM_CCOEFF)
    else:
        res[:] = cv2.matchTemplate(gray_float, templ=template, method=cv2.TM_CCOEFF)
    shifts = np.argmax(res, axis=0)
    values = res[shifts, np.arange(res.shape[1])]
    mask = values > 10
    print(values.min(), values.mean(), values.max())
    res -= res.min()
    res /= res.max() / 255
    res = np.clip(res, 0, 255)
    out1[:res.shape[0], :res.shape[1]] = cv2.cvtColor(res.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    out1[shifts[mask], np.arange(res.shape[1])[mask]] = (0, 255, 0)
    cv2.imshow("", vis)
    cv2.waitKey(1)
