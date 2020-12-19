import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body

import time

body_estimation = Body('model/body_pose_model.pth')

img = 'images/wl-man-3.png'
frame = cv2.imread(img)  # B,G,R order

start = time.time()

candidate, subset = body_estimation(frame)
canvas = copy.deepcopy(frame)

print((subset))
print((candidate))

canvas = util.draw_bodypose(canvas, candidate, subset)

end = time.time()
print("Elapsed time: {} seconds".format(end - start))

# cv2.imshow('Preview', canvas)
# cv2.waitKey(0)

plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()



