import os
import cv2 as cv
import json

video_uid = [f for f in os.listdir("full_scale") if f.startswith("d250")][0]

with open("annotations/narration.json") as f:
    narration = json.load(f)
    narration = narration[video_uid[:-4]]

import pprint

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(video_uid)
pp.pprint(narration["narration_pass_2"]["narrations"][:5])

cap = cv.VideoCapture(os.path.join("full_scale", video_uid))
# cap = cv.VideoCapture("trim.mp4")
fps = int(cap.get(cv.CAP_PROP_FPS))
print(fps)
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
size = (w, h)
frame_num = 0

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*"XVID")
out = cv.VideoWriter(f"narrations/{video_uid[:-4]}.avi", fourcc, fps, size)

query_time_offset = 0.0210286
queries = {
    "fps": 1.875,
    "num_frames": 900,
    "timestamps": [
        [0, 81],
        [104, 112],
        [117, 135],
        [282, 289],
        [13, 15],
        [402, 424],
        [425, 453],
        [477, 494],
        [14, 24],
        [226, 230],
        [428, 428],
        [577, 633],
        [744, 750],
    ],
    "exact_times": [
        [0.0, 43.6657],
        [55.809, 60.26],
        [62.70855, 72.21],
        [150.49668, 154.489],
        [7.16, 8.518],
        [214.789, 226.429],
        [227.014, 241.869],
        [254.78526, 263.679],
        [7.49754, 13.0],
        [120.68154, 123.0],
        [228.57423, 228.57423],
        [307.73553, 338.0],
        [397.04556, 400.0],
    ],
    "sentences": [
        "how many frying pans can i see on the shelf?",
        "what colour bowl did i carry from the plate stand.?",
        "in what location did i see the basket?",
        "what did i pour in the bowl?",
        "where was the soap before i picked it up?",
        "what colour spoon did i carry from the plate stand?",
        "where was the container before i picked it up?",
        "what colour tray did i pick?",
        "where did i put the soap.?",
        "where did i put a plastic peg.?",
        "what did i put in a bowl.?",
        "where did i put a meat container.?",
        "what did i put in the fridge?",
    ],
    "annotation_uids": [
        "f3083484-a6c0-45cb-a40c-b1c2cb470443",
        "f3083484-a6c0-45cb-a40c-b1c2cb470443",
        "f3083484-a6c0-45cb-a40c-b1c2cb470443",
        "f3083484-a6c0-45cb-a40c-b1c2cb470443",
        "f3083484-a6c0-45cb-a40c-b1c2cb470443",
        "f3083484-a6c0-45cb-a40c-b1c2cb470443",
        "f3083484-a6c0-45cb-a40c-b1c2cb470443",
        "f3083484-a6c0-45cb-a40c-b1c2cb470443",
        "3fcc7a1d-c136-41d6-9d63-0f8b4b90b8f0",
        "3fcc7a1d-c136-41d6-9d63-0f8b4b90b8f0",
        "3fcc7a1d-c136-41d6-9d63-0f8b4b90b8f0",
        "3fcc7a1d-c136-41d6-9d63-0f8b4b90b8f0",
        "3fcc7a1d-c136-41d6-9d63-0f8b4b90b8f0",
    ],
    "query_idx": [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4],
}
queries["exact_times"] = [ [s+query_time_offset, e+query_time_offset] for [s,e] in queries["exact_times"] ]

captions = narration["narration_pass_2"]["narrations"]
caption_idx = 0
current_caption = captions[caption_idx]
display_duration = 1  # duration for which caption is displayed post-event in seconds
video_time = 0
font_scale = 2
thickness = 3
while video_time < int(60 * 5):  # cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    video_time = cap.get(cv.CAP_PROP_POS_MSEC) / 1000

    if caption_idx < len(captions):
        # caption_timestamp = captions[caption_idx]["_unmapped_timestamp_sec"]
        caption_timestamp = captions[caption_idx]["timestamp_sec"]
        d = video_time - caption_timestamp
        if 0 <= d < display_duration:
            cv.putText(
                frame,
                captions[caption_idx]["narration_text"],
                (600, 1350),
                cv.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(255, 0, 0),
                thickness=thickness,
            )
            # cv.putText(frame, captions[caption_idx]["narration_text"], (w//3, 4*h//5), cv.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,0,0), thickness=2)
        elif d >= display_duration:
            caption_idx += 1

    for i, (exact_time, sentence) in enumerate(zip(queries["exact_times"], queries["sentences"])):
        s,e = exact_time
        if s <= video_time <= e:
            cv.putText(
                frame,
                sentence,
                (800, (i+1) * 100),
                cv.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(0, 0, 255),
                thickness=thickness,
            )

    cv.putText(
        frame,
        f"Video time: {video_time:.2f}",
        (100, 100),
        cv.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=(255, 0, 0),
        thickness=thickness,
    )
    # cv.imshow('video', frame)
    out.write(frame)
    # if cv.waitKey(5) == ord('q'):
    #     break

cap.release()
out.release()
cv.destroyAllWindows()
