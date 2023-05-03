# affectnet
Facial Expression Recognition (FER) and computing Valence &amp; Arousal through Transfer Learning on AffectNet dataset. Includes the entire source code for data preprocessing, model training, analysis, and visualization.

## Contents
1. [ Abstract ](#abs)
2. [ Methodology Diagram ](#m_dig)
3. [ Introduction ](#intro)
4. [ Setup Instructions ](#setup)
5. [ Usage Instructions ](#usage)
6. [ Quantitative Results ](#quant_res)
7. [ Training Graphs ](#graphs)
8. [ Visual Results ](#vis_res)
9. [ Author ](#auth)

<a name="abs"></a>
## Abstract
This project focuses on developing a facial expression recognition system that can effectively detect and recognize expressions while also computing their valence and arousal values. The proposed system employs transfer learning, utilizing pre-trained convolutional neural network (CNN) models as a starting point to reduce the required training time and improve the system's performance. To evaluate the effectiveness of the proposed approach, several well-known CNN baselines were employed, including VGG19, ResNet152, EfficientNet, \& ConvNext. These models were fine-tuned and evaluated on the AffectNet dataset, which is a large-scale dataset of facial expressions annotated with valence and arousal labels. The dataset consists of over 280,000 images, including eight facial expression categories. The performance of the developed system was evaluated using several evaluation metrics, including, but not limited to, accuracy, F1-score, root mean square error (RMSE), AUC, \& Correlation. At the end, we also implemented a custom CNN architecture (CVNet) for a comparison with transfer learning from above-mentioned baselines.

<a name="m_dig"></a>
## Methodology Diagram
![methodology diagram](https://user-images.githubusercontent.com/41828100/235878998-978072d8-3658-47fc-a5cc-6d40dcf52396.jpg)

<a name="intro"></a>
## Introduction
The report provides an overview of the AFFECT dataset and the task of recognizing facial expressions, valence, and arousal from facial images. Affective computing is a field that aims to develop devices that can recognize and simulate human emotions. Facial expressions are one of the most important nonverbal channels used by humans to convey their internal emotions. The dataset provided contains cropped and resized images along with the location of facial landmarks, categorical emotion labels, and continuous valence and arousal values.

The report discusses the evaluation metrics used for categorical and continuous classification tasks, such as accuracy, F1-score, Cohen's kappa, area under curve (AUC), root mean square error (RMSE), correlation, sign agreement metric, and concordance correlation coefficient (CCC). The task is to use CNN architectures to recognize facial expressions, valence, and arousal from the given dataset. The report also emphasizes the use of best practices and modular implementation of the image classification pipeline.

Facial expressions are a critical aspect of human communication and can convey a wide range of emotions and affective states. The ability to recognize and interpret facial expressions can be important in various fields, such as psychology, medicine, and social robotics. Affective computing is a subfield of computer science that seeks to develop computational systems capable of recognizing, interpreting, and expressing emotions and affective states. The dimensional model of affect is a widely used approach in affective computing, which quantifies emotions in terms of two dimensions: valence and arousal. Valence refers to how positive or negative an event is, while arousal reflects whether an event is exciting/agitating or calm/soothing. The valence-arousal circumplex is a visual representation of this model, where emotions are plotted on a two-dimensional space defined by these values.

Deep learning has shown promising results in various computer vision tasks, including facial expression recognition. Convolutional Neural Networks (CNNs) are a type of deep neural networks that are particularly well-suited for image classification tasks. Several CNN architectures have been proposed in recent years, such as VGG, ResNet, Inception, Xception, MobileNet, EfficientNet, SENet, and DenseNet. These architectures differ in terms of their depth, width, and complexity, and have achieved state-of-the-art performance on various classification benchmarks.

<a name="setup"></a>
## Setup Instructions
These are instructions for setting up the environment for running this code.

The code performs facial expression recognition on the AffectNet dataset and computes valence and arousal values. The pretrained baselines used in this project are: VGG-19, ResNet-152, Swin-V2, EfficientNet, RegNet, and ConvNext.

### Requirements:
To run this code, you will need the following:

1- Python</br>
2- OpenCV</br>
3- PyTorch</br>
4- Pandas</br>
5- scikit-learn

There are 2 methods for running this code. Let us take a brief look at each one of them.

### Method 1 (On Cloud):

Download the Python notebook and open it up inside Google Colaboratory. The datasets will be mounted from your Google Drive account.

Just run the notebook and wait for the models to train themselves. Make sure to run only that model's block which you intend to train. These models take around 1 hour to run a single epoch on the AffectNet dataset. During training, each epoch is saved in checkpoints and the best performing model is also dumped in Google Drive at the end.

### Method 2 (On Device):

Clone this repository to your local machine with the following command:</br>
<pre>git clone https://github.com/athar-usama/affectnet.git</pre>

Install Python and Jupyter Notebook. You can download them from the official websites:
#### Python: https://www.python.org/downloads/
#### Jupyter Notebook: https://jupyter.org/install

Install OpenCV, PyTorch, Pandas, and scikit-learn libraries using pip:</br>
<pre>pip install opencv-python pytorch pandas scikit-learn</pre>
<pre>pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118</pre>

Launch Jupyter Notebook from the command line:</br>
<pre>jupyter notebook</pre>

Open the .ipynb file in Jupyter Notebook and run the code.

<a name="usage"></a>
## Usage Instructions
The code is designed to work with the AffectNet dataset. However, the tar files for train and validation set as well as the test set must be present on a folder inside the Google Drive which is to be mounted. These folders will then be automatically extracted inside the notebook. Therefore, there is no need to download the dataset locally.

All you need is to run the code and wait for the model to be trained.

<a name="quant_res"></a>
## Quantitative Results

### Categorical Metrics
![categorical](https://user-images.githubusercontent.com/41828100/235882389-946c7fdd-1d90-413b-9d93-e9bf7081d186.jpg)

### Continuous Domain Metrics
![continuous](https://user-images.githubusercontent.com/41828100/235882481-9284e709-f342-4c40-a7a5-df08524ecb1c.jpg)

<a name="graphs"></a>
## Training Graphs

### Accuracy Curves
![accuracy](https://user-images.githubusercontent.com/41828100/235883903-35d46db9-1a78-44c0-979d-9a0a8d4d2f39.png)

### Loss Curves
![loss](https://user-images.githubusercontent.com/41828100/235883932-4cb0fee9-a721-407c-ba42-aa5e5a92e079.png)

<a name="vis_res"></a>
## Visual Results

### Correctly Classified Images
![correct](https://user-images.githubusercontent.com/41828100/235882220-54723491-3dd6-4b61-b466-57eb8487a39f.png)

### Incorrectly Classified Images
![incorrect](https://user-images.githubusercontent.com/41828100/235882267-bfc4a5c2-80f7-4290-8edf-1af6382db181.png)

<a name="auth"></a>
## Author
Usama Athar atharusama99@gmail.com
