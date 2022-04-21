import os
import json
import random
from tqdm import tqdm

from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import cv2 as cv

# video_uid = [f for f in os.listdir("full_scale") if f.startswith("77cc")][0][:-4]

with open("annotations/nlq_train.json") as f:
    annotations = json.load(f)

with open("annotations/narration.json") as f:
    narrations = json.load(f)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
acc, acc3, acc5, acc10 = [], [], [], []
dist, d = [], []

# annotation = random.choice(annotations["videos"])
# for foo in annotations["videos"]:
#     if foo["video_uid"] == "c707920d-846e-4dc2-9f16-71705507669f":
#         annotation = foo
#         break
"""
Total: 754 videos, 5 videos redacted.
"""
# print(len(annotations["videos"]))
# c = 0
# for annotation in annotations["videos"]:
#     video_uid = annotation["video_uid"]
#     keys = narrations[video_uid]
#     # if "narration_pass_2" not in keys:
#     if keys["status"] == "redacted":
#         print(keys)
#         c += 1
# print(c)
# import ipdb; ipdb.set_trace()
skipped = 0
# for annotation in tqdm(random.sample(annotations["videos"], 20)):
for annotation in tqdm(annotations["videos"]):
    video_uid = annotation["video_uid"]
    queries = []
    for clip in annotation["clips"]:
        for ann in clip["annotations"]:
            if "language_queries" in ann:
                queries.extend([(q["query"], q["video_start_sec"], q["video_end_sec"]) for q in ann["language_queries"] if "query" in q])

    narration = narrations[video_uid]
    if narration["status"] == "redacted":
        skipped += 1
        continue
    captions = [ (n["narration_text"], n["timestamp_sec"]) for n in narration["narration_pass_2"]["narrations"] ]

    # acc.append(1)
    # acc3.append(1)
    # acc5.append(1)
    # acc10.append(1)
    # continue

    # print(f"video_uid: {video_uid}")
    # print(f"{len(queries)} natural language queries found.")
    # print(f"{len(captions)} captions found.")

    with torch.no_grad():
        narration_embeddings = []
        for caption in captions:
            inputs = tokenizer(caption[0], return_tensors="pt")
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            embedding = last_hidden_states.squeeze().mean(0)
            embedding /= embedding.norm()
            narration_embeddings.append(embedding)

        for query in queries:
            try:
                inputs = tokenizer(query[0], return_tensors="pt")
            except:
                print(f"Could not tokenize {query[0]}, skipping ..")
                continue
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            embedding = last_hidden_states.squeeze().mean(0)
            embedding /= embedding.norm()

            dot_products = torch.tensor([torch.dot(emb, embedding) for emb in narration_embeddings])
            idx = torch.argsort(dot_products, descending=True)

            # idx = random.sample(range(len(captions)), len(captions))

            for i in [3, 5, 10]:
                acc_ = 0.
                for idx_ in idx[:i]:
                    if (query[1] <= captions[idx_][1] <= query[2]):
                        acc_ = 1.
                        break
                if i == 3:
                    acc3.append(acc_)
                elif i == 5:
                    acc5.append(acc_)
                else:
                    acc10.append(acc_)

            m = idx[0]

            acc_ = 1. if (query[1] <= captions[m][1] <= query[2]) else 0.
            acc.append(acc_)

            # dist_ = 0 if acc_ else min([abs(query[1] - captions[m][1]), abs(query[2] - captions[m][1])])
            # dist.append(dist_)

            # d_ = dist_ / (query[2] - query[1])
            # d.append(d_)

            # print("--------")
            # print(query[0])
            # print(captions[m][0])
            # print(bool(acc_))
            # print(f"{dist_:.02f}")
            # print(f"{d_:.02f}")
            # print("--------")

print(f"Accuracy: {np.mean(acc):.02f}")
print(f"Accuracy @ 3: {np.mean(acc3):.02f}")
print(f"Accuracy @ 5: {np.mean(acc5):.02f}")
print(f"Accuracy @ 10: {np.mean(acc10):.02f}")
print(f"Skipped: {skipped}")
# print(f"Median dist: {np.median(dist):.02f}, Mean dist: {np.mean(dist):.02f}")
# print(f"Median d: {np.median(d):.02f}, Mean d: {np.mean(d):.02f}")
