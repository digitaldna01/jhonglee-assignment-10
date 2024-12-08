import numpy as np
import pandas as pd
import os
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from open_clip import create_model_and_transforms, tokenizer, get_tokenizer
import torch.nn.functional as F

file_path = "coco_images_resized/"

df = pd.read_pickle('image_embeddings.pickle')

MODEL_NAME = 'ViT-B-32'
PRETRAINED_SOURCE = 'openai'
model, _, preprocess = create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_SOURCE)
tokenizer = get_tokenizer(MODEL_NAME)
model.eval()

def embed_image(image_query):
    
    # check if the file is image or not
    if isinstance(image_query, str):
        image = preprocess(Image.open(image_query)).unsqueeze(0)
    else:
        image = Image.open(io.BytesIO(image_query.read())) 
        image = preprocess(image).unsqueeze(0)
        
    image_embedding = F.normalize(model.encode_image(image))
    return image_embedding

def embed_text(text_query):
    tokenizer = get_tokenizer('ViT-B-32')
    model.eval()
    text = tokenizer([text_query])
    text_embedding = F.normalize(model.encode_text(text)) ### encode text
    
    return text_embedding

def embed_hybrid(image_query, text_query, hybrid_weight):
    image_embedding = embed_image(image_query)
    text_embedding = embed_text(text_query)
    
    hybrid_weight = torch.tensor(hybrid_weight, dtype=torch.float32)

    hybrid_embedding = F.normalize(hybrid_weight * text_embedding + (1.0 - hybrid_weight) * image_embedding)

    return hybrid_embedding

def get_top_images(query_embedding):
    impaths = []
    similarities = []
        # Calculate cosine similarities
    cosine_similarities = F.cosine_similarity(query_embedding, torch.tensor(df['embedding'].tolist()))
    print(cosine_similarities)

    # Find the index of the highest similarity
    top_indices = torch.topk(cosine_similarities, 5).indices.tolist()
    
    for index in top_indices:
        impath = df['file_name'].iloc[index]
        similarity = cosine_similarities[index].item()
        impaths.append(impath)
        similarities.append(similarity)
    
    return impaths, similarities