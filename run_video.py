"""

Run score model for a single video.

"""
import time
import csv
import copy
import cv2
import numpy as np

from src import util
from src.body import Body
from scoring_model import scoring

body_estimation = Body("model/body_pose_model.pth")

INPUT_FILENAME = "men85kg_0001"
capture = cv2.VideoCapture("85kg_men_test/{}.mp4".format(INPUT_FILENAME))

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))

OUTPUT_FILENAME = "{}-output".format(INPUT_FILENAME)

# Define the codec and create VideoWriter object.
out = cv2.VideoWriter(
    "results/{}.avi".format(OUTPUT_FILENAME),
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    10,
    (frame_width, frame_height),
)

# Creating a dictionary from the output.csv file
reader = csv.reader(open("85kg_men_results/{}/output.csv".format(INPUT_FILENAME), "r"))
dict_csv = {}
next(reader)  # skip header
for row in reader:
    k, v = row
    dict_csv[int(k)] = v

frame_count = 0
start = time.time()

kneeAngles = []
barPositions = []
barAngles = []

while True:
    isTrue, frame = capture.read()

    if isTrue:

        print("Processing %dth frame..." % frame_count)

        candidate, subset = body_estimation(frame)
        canvas = copy.deepcopy(frame)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        if len(candidate) > 0:
            coordinates = np.delete(candidate, (2, 3), axis=1)

            # angle calculation
            action = dict_csv.get(frame_count)
            print("Action:", action)
            # calculate bar position of the frame
            if len(coordinates) > 15:
                barPositions.append(
                    util.calcBarPosition(
                        coordinates[4][0],
                        coordinates[4][1],
                        coordinates[7][0],
                        coordinates[7][1],
                    )
                )

                # calculate bar angle of the frame
                barAngles.append(
                    util.calcBarAngle(
                        coordinates[4][0],
                        coordinates[4][1],
                        coordinates[7][0],
                        coordinates[7][1],
                    )
                )

                # calculate initial knee angle
                if action == "setupsnatch" or action == "setupclean":
                    kneeAngles.append(
                        util.calcInitialKneeAngle(
                            coordinates[11], coordinates[12], coordinates[13]
                        )
                    )

        # Write the frame into the file 'output.avi'
        out.write(canvas)

        # Display the resulting frame
        cv2.imshow("Preview", canvas)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord("d"):
            break

        frame_count += 1

    # Break the loop
    else:
        break

# calculate average initial knee angle
initialKneeAngle = sum(kneeAngles) / len(kneeAngles)

# calculate average of bar angles
avgBarAngle = sum(barAngles) / len(barAngles)

print("Initial Knee Angle(Avg):", initialKneeAngle)
# print('Bar Position', barPositions)
print("Average Bar Angle:", avgBarAngle)

print("------ SCORE REPORT ------")
knee_score = scoring.kneeAngleScore(initialKneeAngle)
bar_score = scoring.barAngleScore(avgBarAngle)
overall_score = scoring.overallScore(knee_score, bar_score)

print("Knee Angle Score:", knee_score)
print("Bar Angle Score:", bar_score)
print("Overall Score:", overall_score)

score_list = []
field_names = ["file", "kAngle", "bAngle", "kScore", "bScore", "oScore"]
temp_scores = {
    "file": INPUT_FILENAME,
    "kAngle": initialKneeAngle,
    "bAngle": avgBarAngle,
    "kScore": knee_score,
    "bScore": bar_score,
    "oScore": overall_score,
}
score_list.append(temp_scores.copy())

with open("./score_report.csv", "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(score_list)

capture.release()
out.release()

cv2.destroyAllWindows()

end = time.time()
print("Elapsed time: {} seconds".format(end - start))

# util.toDataframe()
print("Done")
