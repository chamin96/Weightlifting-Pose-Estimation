import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body

import time

body_estimation = Body('model/body_pose_model.pth')

input_filename = 'snatch_side_view'
capture = cv2.VideoCapture("videos/{}.mp4".format(input_filename))

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))

output_filename = "{}-output".format(input_filename)

# Define the codec and create VideoWriter object.
out = cv2.VideoWriter("results/{}.avi".format(output_filename), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

frame_count = 0
start = time.time()

while True:
    isTrue, frame = capture.read()

    if isTrue == True:
        frame_count += 1
        print('Processing %dth frame...' %frame_count)

        candidate, subset = body_estimation(frame)
        canvas = copy.deepcopy(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # Write the frame into the file 'output.avi'
        out.write(canvas)

        # Display the resulting frame    
        cv2.imshow('Preview',canvas)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('d'):
          break

    # Break the loop
    else:
      break

capture.release()
out.release()

cv2.destroyAllWindows()

end = time.time()
print("Elapsed time: {} seconds".format(end - start))

print('Done')