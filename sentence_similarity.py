import os
import cv2 as cv
import json
import random

from transformers import BertTokenizer, BertModel
import torch

# video_uid = [f for f in os.listdir("full_scale") if f.startswith("77cc")][0][:-4]

with open("annotations/nlq_train.json") as f:
    annotations = json.load(f)
    annotation = random.choice(annotations["videos"])
    video_uid = annotation["video_uid"]
    queries = []
    for clip in annotation["clips"]:
        for ann in clip["annotations"]:
            if "language_queries" in ann:
                queries.extend(ann["language_queries"])

    print(f"video_uid: {video_uid}")
    print(f"{len(queries)} natural language queries found.")

with open("annotations/narration.json") as f:
    narration = json.load(f)
    narration = narration[video_uid]
    captions = narration["narration_pass_2"]["narrations"]


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

with torch.no_grad():
    narration_embeddings = []
    for caption in captions:
        inputs = tokenizer(caption["narration_text"], return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = last_hidden_states.squeeze().mean(0)
        embedding /= embedding.norm()
        narration_embeddings.append(embedding)

    for query in queries:
        query = query["query"]
        inputs = tokenizer(query, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = last_hidden_states.squeeze().mean(0)
        embedding /= embedding.norm()

        m = torch.argmax(torch.tensor([torch.dot(emb, embedding) for emb in narration_embeddings]))

        print("--------")
        print(query)
        print(captions[m]["narration_text"])
