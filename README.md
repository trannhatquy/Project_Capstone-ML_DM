# Project_Capstone-ML_DM

Basic Image Captioning Model with Pytorch Lightning using CNN and LSTM

This repository show how to apply neural network (CNN, LSTM) using pytorch lightning to generate a caption from a given image. The dataset we use is Flickr8k Image Caption dataset. The model contains encoder to encode the image into a latent vector and then use that vector in the decoder for generate text.
<br/> <br/>

<img src="utility/model.png"/>
<br/><br/>

Our architecture looks like the image on the top, but have a little difference
## Dataset Source

Flickr8k Dataset : https://www.kaggle.com/nunenuh/flickr8k <br/>

## How to run the project 
1. Clone the repository 

2. Download the dataset in the link above, unzip it then put it in the data folder 

3. Install all the requirements in requirements.txt 

4. Open the train_inference.ipynb and train, test the model
