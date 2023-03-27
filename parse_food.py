import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main(clip_model_type: str, data_path: str, token_limit: int, test_size: int):
    #preprocessing
    df = pd.read_csv(data_path + 'Captions.csv')
    df.iloc[0] = df.iloc[0].astype(np.int64)
    df.iloc[1] = df.iloc[1].astype(str)
    df = df.rename(columns={df.columns[0]:'image_id', df.columns[1]:'caption'})
    df = df[df['caption'].map(len) < token_limit]
    _, X_test = train_test_split(df, test_size=test_size, random_state=42)
    data = X_test

    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./{clip_model_name}_RN_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        print(data)
        d = data.iloc[i]
        img_id = d['image_id']
        filename = f"./dataset/{int(img_id)}.jpg"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--token_limit', type=int, default=240)
    parser.add_argument('--test_size', type=int, default=0.3)
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.data_path, args.token_limit, args.test_size))