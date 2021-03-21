"""

Run multiple videos in directory at once.

"""

from os import walk
import copy
import csv
import time
import numpy as np
import cv2

from src import util
from src.body import Body
from scoring_model import scoring

SRC_DIR = "85kg_men_test"
CSV_DIR = "./85kg_men_results"
DST_DIR = "./result_score/{}".format(SRC_DIR)
body_estimation = Body("model/body_pose_model.pth")
score_list = []
field_names = ["file", "kAngle", "bAngle", "kScore", "bScore", "oScore"]

_, _, video_files = next(walk(SRC_DIR))
count = 1

for file in video_files:
    print("Processing {}/{} file...".format(count, len(video_files)))
    count += 1
    capture = cv2.VideoCapture("{}/{}".format(SRC_DIR, file))
    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))

    OUTPUT_FILENAME = "{}-output".format(file)

    # Define the codec and create VideoWriter object.
    out = cv2.VideoWriter(
        "results/{}/{}.avi".format(SRC_DIR, OUTPUT_FILENAME),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        10,
        (frame_width + int(frame_width / 4), frame_height),
    )

    # Creating a dictionary from the output.csv file
    sub_dir = file.replace(".mp4", "")
    reader = csv.reader(open("{}/{}/output.csv".format(CSV_DIR, sub_dir), "r"))
    dict_csv = {}
    next(reader)  # skip header

    for row in reader:
        k, v = row
        dict_csv[int(k)] = v

    frame_count = 0
    start = time.time()

    kneeAngles = []
    larmAngles = []
    rarmAngles = []
    lkneeAngles = []
    rkneeAngles = []
    barPositions = []
    barAngles = []

    while True:
        isTrue, frame = capture.read()

        if isTrue:
            candidate, subset = body_estimation(frame)
            canvas = copy.deepcopy(frame)
            canvas = util.draw_bodypose(canvas, candidate, subset)

            if len(candidate) > 0:
                coordinates = np.delete(candidate, (2, 3), axis=1)

                # util.toTempList(candidate)
                coordinates = np.delete(candidate, (2, 3), axis=1)

                # angle calculation
                action = dict_csv.get(frame_count)

                # draw action text
                canvas = util.draw_action(canvas, action)

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
                    bar_angle = util.calcBarAngle(
                        coordinates[4][0],
                        coordinates[4][1],
                        coordinates[7][0],
                        coordinates[7][1],
                    )
                    if bar_angle > 0:
                        barAngles.append(bar_angle)

                    # draw bar angle
                    canvas = util.draw_bar_angle(canvas, bar_angle)

                    # calculate initial knee angle
                    if action == "setupsnatch" or action == "setupclean":
                        knee_angle = util.calcInitialKneeAngle(
                            coordinates[11], coordinates[12], coordinates[13]
                        )
                        if knee_angle > 0:
                            kneeAngles.append(knee_angle)
                        # draw knee angle
                        canvas = util.draw_knee_angle(canvas, knee_angle)

                    # calculate arm/knee angles at the end
                    if action == "standsnatch" or action == "recoveryjerk":
                        l_arm_angle = util.calcArmAngle(
                            coordinates[5], coordinates[6], coordinates[7]
                        )
                        if l_arm_angle > 100:
                            larmAngles.append(l_arm_angle)

                        r_arm_angle = util.calcArmAngle(
                            coordinates[2], coordinates[3], coordinates[4]
                        )
                        if r_arm_angle > 100:
                            rarmAngles.append(r_arm_angle)

                        # draw arm angle
                        canvas = util.draw_arm_angle(canvas, l_arm_angle, r_arm_angle)

                        l_knee_angle = util.calcInitialKneeAngle(
                            coordinates[11], coordinates[12], coordinates[13]
                        )
                        if l_knee_angle > 100:
                            lkneeAngles.append(l_knee_angle)

                        r_knee_angle = util.calcInitialKneeAngle(
                            coordinates[8], coordinates[9], coordinates[10]
                        )
                        if r_knee_angle > 100:
                            rkneeAngles.append(r_knee_angle)

                        # draw knee angle
                        canvas = util.draw_knee_angle_end(
                            canvas, l_knee_angle, r_knee_angle
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
    if len(kneeAngles) > 0:
        initialKneeAngle = sum(kneeAngles) / len(kneeAngles)
    else:
        initialKneeAngle = -1

    # calculate average of bar angles
    if len(barAngles) > 0:
        avgBarAngle = sum(barAngles) / len(barAngles)
    else:
        avgBarAngle = -1

    # calculate average of l arm angles
    avglArmAngle = sum(larmAngles) / len(larmAngles)

    # calculate average of r arm angles
    avgrArmAngle = sum(rarmAngles) / len(rarmAngles)

    # calculate average of l legs angles
    avglLegAngle = sum(lkneeAngles) / len(lkneeAngles)

    # calculate average of r legs angles
    avgrLegAngle = sum(rkneeAngles) / len(rkneeAngles)

    if initialKneeAngle > 0 and avgBarAngle > 0:
        print("Initial Knee Angle(Avg):", initialKneeAngle)
        print("Average Bar Angle:", avgBarAngle)
        print("Average Left Arm Angle:", avglArmAngle)
        print("Average Right Arm Angle:", avgrArmAngle)
        print("Average Left Leg Angle:", avglLegAngle)
        print("Average Right Leg Angle:", avgrLegAngle)

        print("------ SCORE REPORT ------")
        knee_score = scoring.kneeAngleScore(initialKneeAngle)
        bar_score = scoring.barAngleScore(avgBarAngle)
        arms_score = scoring.armsAngleScore(avglArmAngle, avgrArmAngle)
        legs_score = scoring.legsAngleScore(avglLegAngle, avgrLegAngle)
        overall_score = scoring.overallScore(
            knee_score, bar_score, legs_score, arms_score
        )

        print("Knee Angle Score:", knee_score)
        print("Bar Angle Score:", bar_score)
        print("Legs Angle Score:", legs_score)
        print("Arms Angle Score:", arms_score)
        print("Overall Score:", overall_score)

        canvas = util.draw_score_report(
            canvas,
            initialKneeAngle,
            avgBarAngle,
            avglArmAngle,
            avgrArmAngle,
            avglLegAngle,
            avgrLegAngle,
            knee_score,
            bar_score,
            legs_score,
            arms_score,
            overall_score,
        )

        # Write the frame into the file 'output.avi'
        for i in range(100):
            out.write(canvas)
            # Display the resulting frame
            cv2.imshow("Preview", canvas)
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord("d"):
                break

        field_names = [
            "file",
            "kAngle",
            "bAngle",
            "larmAngle",
            "rarmAngle",
            "llegAngle",
            "rlegAngle",
            "legScore",
            "armScore",
            "kScore",
            "bScore",
            "oScore",
        ]
        temp_scores = {
            "file": file,
            "kAngle": initialKneeAngle,
            "bAngle": avgBarAngle,
            "larmAngle": avglArmAngle,
            "rarmAngle": avgrArmAngle,
            "llegAngle": avglLegAngle,
            "rlegAngle": avgrLegAngle,
            "legScore": legs_score,
            "armScore": arms_score,
            "kScore": knee_score,
            "bScore": bar_score,
            "oScore": overall_score,
        }
        score_list.append(temp_scores.copy())

    else:
        print("Angles were not detected!")

    capture.release()
    out.release()

    cv2.destroyAllWindows()

    end = time.time()
    print("Elapsed time: {} seconds".format(end - start))
    print("Finished processing {}".format(file))

with open("{}/score_report.csv".format(DST_DIR), "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(score_list)

print("Done")
