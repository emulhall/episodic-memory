#! /usr/bin/env python
"""
Prepare Ego4d episodic memory NLQ for model training.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import collections
import csv
import json
import math
import os

import torch
import tqdm


CANONICAL_VIDEO_FPS = 30.0
FEATURE_WINDOW_SIZE = 16.0
FEATURES_PER_SEC = CANONICAL_VIDEO_FPS / FEATURE_WINDOW_SIZE


def get_nearest_frame(time, floor_or_ceil=None):
    """Obtain the nearest frame for a given time, video fps, and feature window."""
    return floor_or_ceil(int(time * CANONICAL_VIDEO_FPS / FEATURE_WINDOW_SIZE))


def process_question(question):
    """Process the question to make it canonical."""
    return question.strip(" ").strip("?").lower() + "?"

def load_narrations(path):
    read_path=os.path.join(path)
    with open(read_path, "r") as file_id:
        raw_data = json.load(file_id)

    return raw_data

def process_narration(narration_data, video_uid, time_stamps):
    narrations = narration_data[video_uid]
    start, end =time_stamps
    output=[]
    for n in narrations.keys():
        if n=="status":
            continue
        n_start = narrations[n]["summaries"][0]["start_sec"]
        n_end = narrations[n]["summaries"][0]["end_sec"]

        for narration in narrations[n]['narrations']:
            frame = narration['timestamp_frame']

            if (start <= frame) and (end >= frame):
                output.append(narration['narration_text'])

    return output


def reformat_data(split_data, narration_data,test_split=False,max_size=750):
    """Convert the format from JSON files.
    fps, num_frames, timestamps, sentences, exact_times,
    annotation_uids, query_idx.
    """
    formatted_data = {}
    clip_video_map = {}
    for video_datum in split_data["videos"][:max_size]:
        for clip_datum in video_datum["clips"]:
            clip_uid = clip_datum["clip_uid"]
            clip_video_map[clip_uid] = (
                video_datum["video_uid"],
                clip_datum["video_start_sec"],
                clip_datum["video_end_sec"],
            )
            clip_duration = clip_datum["video_end_sec"] - clip_datum["video_start_sec"]
            num_frames = get_nearest_frame(clip_duration, math.ceil)
            new_dict = {
                "fps": FEATURES_PER_SEC,
                "num_frames": num_frames,
                "timestamps": [],
                "exact_times": [],
                "sentences": [],
                "annotation_uids": [],
                "query_idx": [],
                "narrations": []
            }

            for ann_datum in clip_datum["annotations"]:
                for index, datum in enumerate(ann_datum["language_queries"]):
                    if not test_split:
                        start_time = float(datum["clip_start_sec"])
                        end_time = float(datum["clip_end_sec"])
                    else:
                        # Random placeholders for test set.
                        start_time = 0.
                        end_time = 0.

                    if "query" not in datum or not datum["query"]:
                        continue
                    new_dict["sentences"].append(process_question(datum["query"]))
                    new_dict["annotation_uids"].append(ann_datum["annotation_uid"])
                    new_dict["query_idx"].append(index)
                    new_dict["exact_times"].append([start_time, end_time]),
                    start_frame = get_nearest_frame(start_time, math.floor)
                    end_frame = get_nearest_frame(end_time, math.ceil)
                    new_dict["timestamps"].append(
                        [
                            start_frame,
                            end_frame,
                        ]
                    )
                    new_dict["narrations"].append(process_narration(narration_data, video_datum["video_uid"], [start_frame, end_frame]))
            formatted_data[clip_uid] = new_dict
    return formatted_data, clip_video_map


def convert_ego4d_dataset(args, narration_data):
    """Convert the Ego4D dataset for VSLNet."""
    # Reformat the splits to train vslnet.
    all_clip_video_map = {}
    for split in ("train", "val", "test"):
        read_path = args[f"input_{split}_split"]
        print(f"Reading [{split}]: {read_path}")
        with open(read_path, "r") as file_id:
            raw_data = json.load(file_id)
        if split=="train":
            max_size=1
        else:
            max_size=1
        data_split, clip_video_map = reformat_data(raw_data, narration_data,split == "test", max_size)
        all_clip_video_map.update(clip_video_map)
        num_instances = sum(len(ii["sentences"]) for ii in data_split.values())
        print(f"# {split}: {num_instances}")

        os.makedirs(args["output_save_path"], exist_ok=True)
        save_path = os.path.join(args["output_save_path"], f"{split}.json")
        print(f"Writing [{split}]: {save_path}")
        with open(save_path, "w") as file_id:
            json.dump(data_split, file_id)

    # Extract visual features based on the all_clip_video_map.
    feature_sizes = {}
    os.makedirs(args["clip_feature_save_path"], exist_ok=True)
    progress_bar = tqdm.tqdm(all_clip_video_map.items(), desc="Extracting features")
    for clip_uid, (video_uid, start_sec, end_sec) in progress_bar:
        feature_path = os.path.join(args["video_feature_read_path"], f"{video_uid}.pt")
        feature = torch.load(feature_path)

        # Get the lower frame (start_sec) and upper frame (end_sec) for the clip.
        clip_start = get_nearest_frame(start_sec, math.floor)
        clip_end = get_nearest_frame(end_sec, math.ceil)
        clip_feature = feature[clip_start : clip_end + 1]
        feature_sizes[clip_uid] = clip_feature.shape[0]
        feature_save_path = os.path.join(
            args["clip_feature_save_path"], f"{clip_uid}.pt"
        )
        torch.save(clip_feature, feature_save_path)

    save_path = os.path.join(args["clip_feature_save_path"], "feature_shapes.json")
    with open(save_path, "w") as file_id:
        json.dump(feature_sizes, file_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_train_split", required=True, help="Path to Ego4d train split"
    )
    parser.add_argument(
        "--input_val_split", required=True, help="Path to Ego4d val split"
    )
    parser.add_argument(
        "--input_test_split", required=True, help="Path to Ego4d test split"
    )
    parser.add_argument(
        "--output_save_path", required=True, help="Path to save the output jsons"
    )
    parser.add_argument(
        "--video_feature_read_path", required=True, help="Path to read video features"
    )
    parser.add_argument(
        "--clip_feature_save_path",
        required=True,
        help="Path to save clip video features",
    )
    parser.add_argument(
        "--narration_read_path", required=True, help="Path to read narrations"
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    
    narration_data=load_narrations(parsed_args['narration_read_path'])
    convert_ego4d_dataset(parsed_args, narration_data)
    #load_narrations(parsed_args['narration_read_path'])
