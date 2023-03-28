# CLIP Prefix Captioning.

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://choosealicense.com/licenses/mit/)

## Description  

In this article, we are utilising ["ClipCap: CLIP Prefix for Image Captioning"](https://github.com/rmokady/CLIP_prefix_caption) that is based on the [CLIP](https://github.com/openai/CLIP) model to finetune for our food-based dataset.

If you are unfamiliar with either resource, please visit the links above first. Our code is largely an extension of the former and we take no credit for any of the authors' code.



## COCO Examples (Replace with our own finetuned images and captions)




 Images        | Finetuned with Food Dataset           | Original  
| ------------- |:-------------:| -----:|
| ![Inf1](Images/inf1.jpg)   | A healthy diet of fruits, vegetables, nuts, seeds, nuts, seeds, and whole grains. | A table topped with lots of different types of fruits and vegetables. |
| ![Inf2](Images/inf2.jpg)      | Pastries and muffins sitting on a table next to each other on a blue background.      |   a close up of muffins on a napkin on a table |
| ![Inf3](Images/inf3.jpg)| A bowl of pasta with noodles and cheese on a wooden table.     |    A bowl of spaghetti and a fork on a table. |
| ![Inf4](Images/inf4.jpg) | A bowl of soup with vegetables and noodles on a plate.    |    A bowl of soup, a bowl of vegetables, and a bowl of rice.|

## Prerequisites for Fine-tuning

Clone, create environment and install dependencies:  
```
git clone https://github.com/rmokady/CLIP_prefix_caption && cd CLIP_prefix_caption
conda env create -f environment.yml
conda activate clip_prefix_caption
```



## Fine-tuning on your own data

Extract CLIP features using `parse_food.py` (output is `./RN50x4_RN_train.pkl`):
```
python parse_food.py --clip_model_type RN50x4 --data_path <captions dir> --token_limit <max token length> --test_size <% of dataset>
```
Train with fine-tuning of GPT2:
```
python train.py --data ./data/ViT-B_32_train.pkl --out_dir ./coco_train/
```

Train only transformer mapping network:
```
python train.py --only_prefix --data ./RN50x4_RN_train.pkl --out_dir ./model_checkpoints --mapping_type transformer  --num_layers 8 --prefix_length 40 --prefix_length_clip 40 --is_rn
```


## Model Architecture

![Model](Images/modelarchitecture.png)

Our best inference results were obtained by only training the ResNet based transformer while keeping CLIP and GPT2 frozen. 

Our final checkpoint was trained on a subset `test_size = 0.3` of the [food dataset](https://www.kaggle.com/datasets/zeynaloy/food-related-pictures-dataset-with-captions) from Kaggle.

An important parameter to note at inference is the `Temperature`. -- Varying `Temperature` can lead to significantly different results.

![Inf1](Images/inf1.jpg)

| `Temperature`= 1 | `Temperature`= 0.1|
| ---|:---|
| A healthy diet of fruits, vegetables, nuts, seeds, nuts, seeds, and whole grains.| A healthy eating diet with healthy foods and healthy fats. |

