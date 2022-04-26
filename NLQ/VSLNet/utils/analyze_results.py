import argparse
import collections
import csv
import json
import math
import os
import numpy as np

import torch
import tqdm
import pickle

CANONICAL_VIDEO_FPS = 30.0
FEATURE_WINDOW_SIZE = 16.0
FEATURES_PER_SEC = CANONICAL_VIDEO_FPS / FEATURE_WINDOW_SIZE


def get_nearest_frame(time, floor_or_ceil=None):
    """Obtain the nearest frame for a given time, video fps, and feature window."""
    return floor_or_ceil(int(time * CANONICAL_VIDEO_FPS / FEATURE_WINDOW_SIZE))

def compute_IoU(pred, gt):
    """Compute the IoU given predicted and ground truth windows."""
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap

def process_question(question):
    """Process the question to make it canonical."""
    return question.strip(" ").strip("?").lower() + "?"

def load_json(read_path):
    with open(read_path, "r") as file_id:
        data = json.load(file_id)
    return data

def parse_results(args):
    results = load_json(args['results_path'])
    narrations = load_json(args['narration_path'])
    data = load_json(args['dataset_path'])

    output = []
    csv_columns = ["clip_uid", "narration", "query", "IOU"]

    results = results["results"]
    for r in results:
        video_id=None
        for v in data["videos"]:
            for c in v["clips"]:
                temp_dict = {}
                if c["clip_uid"]==r["clip_uid"]:
                    temp_dict["clip_uid"] = r["clip_uid"]
                    video_id = v["video_uid"]
                    narration = narrations[video_id]

                    if narration["status"]=="redacted":
                        temp_dict["narration"] = "redacted"
                        
                    else:
                        window_size = 11
                        temp_dict["narration"]=[]
                        for n in narration["narration_pass_2"]["narrations"]:
                            n_time_stamp = n["timestamp_sec"]

                            start_sec = c["clip_start_sec"]
                            end_sec = c["clip_end_sec"]

                            if start_sec <= n_time_stamp <= end_sec:
                                clip_start = get_nearest_frame(start_sec, math.floor)
                                clip_end = get_nearest_frame(end_sec, math.floor)
                                n_frame = get_nearest_frame(n_time_stamp, math.floor)-start_sec
                                sw, ew = n_frame - window_size//2, n_frame + window_size//2 + 1
                                s, e = max(0, sw), min(((clip_end+1)-clip_start), ew)

                                if ((e-s)>0):
                                    temp_dict["narration"].append(n["narration_text"])

                    for a in c["annotations"]:
                        if a["annotation_uid"]==r["annotation_uid"]:
                            query = a["language_queries"][int(r["query_idx"])]
                            temp_dict["query"] = query["query"]
                            gt_time = [[query["clip_start_sec"], query["clip_end_sec"]]]



                    predicted_times = r["predicted_times"]
                    
                    temp_dict["IOU"] = compute_IoU(predicted_times[0],gt_time)
                    output.append(temp_dict)


                    break
                else:
                    continue
            if video_id!=None:
                break

    try:
        with open(args["output_path"], 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in output:
                writer.writerow(data)
    except IOError:
        print("I/O error")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path", required=True, help="Path to validation dataset"
    )    
    parser.add_argument(
        "--results_path", required=True, help="Path to results"
    )
    parser.add_argument(
        "--narration_path", required=True, help="Path to narrations"
    )
    parser.add_argument(
        "--output_path", required=True, help="Path to save csv"
    )          
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    
    parse_results(parsed_args)