import clip
import json
from model_archs import ClipCaptionModel, ClipCaptionPrefix
from caption_prediction import generate2
import cv2
import os
import cv2
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, BertModel, BertTokenizer
from tqdm import tqdm, trange
# from google.colab import files
# import skimage.io as io
import PIL.Image
# from IPython.display import Image

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]

D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


CUDA = get_device
acc, acc3, acc5, acc10 = [], [], [], []

# Load model architecture
device = CUDA(0)
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load model weights
model_path = f"model_wieghts.pt"
prefix_length = 10
model = ClipCaptionModel(prefix_length)
model.load_state_dict(torch.load(model_path, map_location=CPU))
model = model.eval()
model = model.to(device)

# Read Video and generate Captions
video_path = "/Users/abhirajmohan/ego4d_data/v1/full_scale/d250521e-5197-44aa-8baa-2f42b24444d2.mp4"
video = cv2.VideoCapture(video_path)
captions = []
fps = int(video.get(cv2.CAP_PROP_FPS))
while video.isOpened():
    r, frame = video.read()
    pil_image = PIL.Image.fromarray(frame)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
    captions.append(generated_text_prefix)
captions = captions[::fps]

# Read queries and answers
with open("annotations/nlq_train.json") as f:
    annotations = json.load(f)
video_uid = "d250521e-5197-44aa-8baa-2f42b24444d2"  # read from the video path
annotation = [a for a in annotations['videos'] if a['video_uid'] == video_uid][0]
queries = []
for clip in annotation["clips"]:
    for ann in clip["annotations"]:
        if "language_queries" in ann:
            queries.extend([(q["query"],
                             q["video_start_sec"],
                             q["video_end_sec"]) for q in ann["language_queries"] if "query" in q])

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")


with torch.no_grad():
    narration_embeddings = []
    for caption in captions:
        inputs = tokenizer(caption, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embedding = last_hidden_states.squeeze().mean(0)
        embedding /= embedding.norm()
        narration_embeddings.append(embedding)

with torch.no_grad():
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

        for i in [3, 5, 10]:
            acc_ = 0.
            for idx_ in idx[:i]:
                if query[1] <= captions[idx_][1] <= query[2]:
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

print(np.mean(acc), np.mean(acc3), np.mean(acc5), np.mean(acc10))
