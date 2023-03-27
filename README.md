# CLIP prefix captioning.

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> 

## Fine-tuning implementation for the paper ["ClipCap: CLIP Prefix for Image Captioning"](https://arxiv.org/abs/2111.09734)


## Description  
Here, we are providing guidance for fine-tuning ["ClipCap: CLIP Prefix for Image Captioning"](https://arxiv.org/abs/2111.09734) that is itself based on the [CLIP](https://github.com/openai/CLIP) model. If you are unfamiliar with either resource, please visit the links above first. Our code is largely an extension of the former and we take no credit for any of the authors' code.

## COCO Examples (Replace with our own finetuned images and captions)

<table>
  <tr>
    <td><img src="Images/COCO_val2014_000000562207.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000165547.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000579664.jpg" ></td>
  </tr>
  <tr>
    <td>A couple of people standing next to an elephant. </td>
     <td>A wooden table sitting in front of a window.</td>
     <td>A bunch of bananas sitting on top of a table.</td>
  </tr>
 </table>
 
 <table>
  <tr>
    <td><img src="Images/COCO_val2014_000000060623.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000386164.jpg" ></td>
    <td><img src="Images/COCO_val2014_000000354533.jpg" ></td>
  </tr>
  <tr>
    <td>A woman holding a plate with a piece of cake in front of her face. </td>
     <td>A wooden table topped with lots of wooden utensils.</td>
     <td>A red motorcycle parked on top of a dirt field.</td>
  </tr>
 </table>


## Pre-trained Weights for the official paper
[COCO](https://drive.google.com/file/d/1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX/view?usp=sharing) and [Conceptual Captions](https://drive.google.com/file/d/14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT/view?usp=sharing) pretrained models are available for the authors' mlp mapping network. For the transformer (without fine-tuning GPT-2), [COCO](https://drive.google.com/file/d/1GYPToCqFREwi285wPLhuVExlz7DDUDfJ/view?usp=sharing) pretrained model is provided.

## Prerequisites for Fine-tuning

Clone, create environment and install dependencies:  
```
git clone https://github.com/rmokady/CLIP_prefix_caption && cd CLIP_prefix_caption
conda env create -f environment.yml
conda activate clip_prefix_caption
```

## Fine-tuning on your own data

Extract CLIP features using (output is `data/ViT-B_32_train.pkl`):
```
python parse_food.py --clip_model_type ViT-B/32
```
Train with fine-tuning of GPT2:
```
python train.py --data ./data/ViT-B_32_train.pkl --out_dir ./coco_train/
```

Train only transformer mapping network:
```
python train.py --only_prefix --data ./data/ViT-B_32_train.pkl --out_dir ./data_train/ --mapping_type transformer  --num_layers 8 --prefix_length 40 --prefix_length_clip 40
```

**If you wish to use ResNet-based CLIP:** 

```
python parse_food.py --clip_model_type RN50x4
```
```
python train.py --only_prefix --data ./data/RN50x4_train.pkl --out_dir ./data_train/ --mapping_type transformer  --num_layers 8 --prefix_length 40 --prefix_length_clip 40 --is_rn
```

## Citation
If you use the author's code for your research, please cite:
```
@article{mokady2021clipcap,
  title={ClipCap: CLIP Prefix for Image Captioning},
  author={Mokady, Ron and Hertz, Amir and Bermano, Amit H},
  journal={arXiv preprint arXiv:2111.09734},
  year={2021}
}
```


## Acknowledgments
This repository is heavily based on [CLIP](https://github.com/openai/CLIP) and [Hugging-faces](https://github.com/huggingface/transformers) repositories.
For training, the authors of the original paper used the data of [COCO dataset](https://cocodataset.org/#home) and [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/).

## Contact
For any inquiry regarding finetuning, please contact us at:

For any inquiry regarding the implementation of the original paper, please contact the authors at their email addresses: ron.mokady@gmail.com or amirhertz@mail.tau.ac.il.


