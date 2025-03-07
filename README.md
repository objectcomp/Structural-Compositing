# Harmonizing Structural Cues for Image Compositing
This repository provides a diffusion-based framework for controllable image compositing, enabling structural harmonization and seamless blending of foreground and background elements with precise control over the foreground object. Use our code and models to generate high-quality, realistic composited images for both creative and practical applications.

## 🚀 Features

   ##### 🔹 Dedicated Encoders - Specially designed to extract structural cues from each condition for structure-aware compositing.
   ##### 🔹 Adaptive Gating - Composites foreground and background structural cues adaptively.
   ##### 🔹 Specialized Feature Injection - Incorporates customized feature injection block for injecting structural cues.


## 📥 Installation
create an environment using by running following command.
```python
conda env create -f environment.yaml
```

## 🔹 Training
For training on your own dataset. Follow the following steps.

### 1. Dataset Folder Structure:

 ```python
Compositing/
├─ ckpt/
├─ dataset/
│   ├─ img/
│   └─ cond/
       ├─ midas/
       ├─ hed/
       └─ canny/
       └─ list.txt
```
### 2. Structural Annotations:
Run the condition detectors in ./annotator/ to extract structural masks for every directory .cond/hed, .cond/canny, .cond/midas.
See the list.txt file to get an idea on how to structure the annotations. 

### 3. Pre-Trained weights
 1️. First click here -> [Stable Diffusion](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) to download pre-train model weights. You want the file "v1-5-pruned.ckpt",  place 
    it in ./ckpt folder. 
 2. Now, initialize the weights for compositing using the following command.

```python
python utils/init_weights.py init_comp ckpt/v1-5-pruned.ckpt configs/comp_v15.yaml ckpt/init_comp.ckpt
```
This prepares the initial weights by integrating base Stable Diffusion weights to our model.

### 4. Start Training
Once the dataset is organized and model wieghts are initilized, run:

```python
python src/train/train.py
```

## ✅ Inference
To perform inference, refer to the src/inference.py file.
