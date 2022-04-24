import clip
import json
import time
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
from pprint import pprint
import pickle

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
DEBUG_SEGFAULT = False

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
ego4d_dataroot = "/home/jayant/big_drive/ego4d_data/v1"
video_path = lambda video_uid: os.path.join(ego4d_dataroot, "full_scale", f"{video_uid}.mp4")
# video_uid = "d250521e-5197-44aa-8baa-2f42b24444d2"  # read from the video path

# inc_t specifies the captioning fps in terms of caption_every_{inc_t}_seconds
# inc_t=5 implies caption one frame every 5s
total_captioned_duration = 0
def generate_image_captions_for_video(video_uid, inc_t=5):
    global total_captioned_duration
    if DEBUG_SEGFAULT:
        tqdm.write(video_path(video_uid))
    video = cv2.VideoCapture(video_path(video_uid))
    captions = []
    fps = video.get(cv2.CAP_PROP_FPS)
    nframes = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = (nframes / fps)
    cur_t = 0
    st = time.time()
    while cur_t < duration:
        if DEBUG_SEGFAULT:
            tqdm.write(f"Reading frame {cur_t:.02f} | {duration:.02f} ..")
        video.set(cv2.CAP_PROP_POS_MSEC, cur_t * 1000) # convert to milliseconds
        r, frame = video.read()
        if DEBUG_SEGFAULT:
            tqdm.write("Done")
        if not r:
            print(f"Could not read frame even though cur_t={cur_t:.02f} < duration={duration:.02f}")
            break
        pil_image = PIL.Image.fromarray(frame[..., ::-1])
        image = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
        captions.append((generated_text_prefix, cur_t))
        cur_t += inc_t
    et = time.time()
    total_captioned_duration += duration / 3600
    tqdm.write(f"Captioning took {et-st:.0f}s | Total captioned duration @ freq {inc_t}s : {total_captioned_duration:.02f} hours")
    return captions

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Read queries and answers
with open(os.path.join(ego4d_dataroot, "annotations", "nlq_train.json")) as f:
    annotations = json.load(f)

# annotation = [a for a in annotations['videos'] if a['video_uid'] == video_uid][0]
for ii, annotation in tqdm(enumerate(annotations["videos"]), total=len(annotations["videos"])):
    video_uid = annotation["video_uid"]
    if video_uid in [ 
            "250d59fc-e1e3-479e-946d-f108d4f38275", 
            "21dabbc1-d8b9-4631-846b-4e8a0fc60048", 
            "40359e3f-931e-4c0b-8d8b-a87d1773320f",
            "0ca23a40-6daf-4503-bfa2-f315a79b7317",
            "d8c894ab-7b08-4983-9e80-fdb5d6ee0202",
            "155f8d74-4c5c-4821-a18b-fceaa9c6199c",
            "dfdfde56-5e0f-4e8f-aa5b-73fae9336086",
            "b367b7d2-180e-48c6-b4b3-34e770acbbd4",
            "968139e2-987e-4615-a2d4-fa2e683bae8a",
            "f0917e7a-e945-473d-95a6-12a1b4cb4e27",
            "e4dc253e-e5be-4b1e-89c2-ab1a47c486b0",
            "7f0320b1-b866-4c80-99bf-42125d99b99e",
            "8d3ac72b-5e56-4bb8-9fb2-7b57c8c9f530",
            ]:
        continue
    
    captions_savepath = os.path.join("video_captions", f"{video_uid}.pkl")
    if os.path.exists(captions_savepath):
        with open(captions_savepath, 'rb') as f:
            captions = pickle.load(f)
    else:
        captions = generate_image_captions_for_video(video_uid)
        with open(captions_savepath, 'wb') as f:
            pickle.dump(captions, f)

    queries = []
    for clip in annotation["clips"]:
        for ann in clip["annotations"]:
            if "language_queries" in ann:
                queries.extend([(q["query"],
                                 q["video_start_sec"],
                                 q["video_end_sec"]) for q in ann["language_queries"] if "query" in q])


    with torch.no_grad():
        narration_embeddings = []
        for caption in captions:
            inputs = bert_tokenizer(caption[0], return_tensors="pt")
            outputs = bert_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            embedding = last_hidden_states.squeeze().mean(0)
            embedding /= embedding.norm()
            narration_embeddings.append(embedding)

    with torch.no_grad():
        for query in queries:
            try:
                inputs = bert_tokenizer(query[0], return_tensors="pt")
            except:
                print(f"Could not tokenize {query[0]}, skipping ..")
                continue
            outputs = bert_model(**inputs)
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

    if ii % 10 == 0:
        tqdm.write(f"{np.mean(acc):f}, {np.mean(acc3):f}, {np.mean(acc5):f}, {np.mean(acc10):f}")

print(np.mean(acc), np.mean(acc3), np.mean(acc5), np.mean(acc10))
