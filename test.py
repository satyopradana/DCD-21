# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from PIL import Image
from random import sample
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.models import Model
from pathlib import Path
Image.LOAD_TRUNCATED_IMAGES = True
import requests
from io import BytesIO
import os
import streamlit as st
import pickle

# Read the data files
listing_data = pd.read_csv("current_farfetch_listings.csv")

# drop the unnames: 0 column
listing_data.drop('Unnamed: 0', axis=1, inplace=True)

# Drop priceInfo.installmentsLabel
listing_data.drop('priceInfo.installmentsLabel', axis=1, inplace=True)

# Drop the column merchandiseLabel
listing_data.drop('merchandiseLabel', axis=1, inplace=True)

# fill the null values in priceInfo.discountLabel with 0
listing_data['priceInfo.discountLabel'] = listing_data['priceInfo.discountLabel'].fillna(0)

# drop the size column
listing_data.drop('availableSizes', axis=1, inplace=True)

# Extracting the Image 
def load_images():
    
    # Store the directory path in a variable
    cutout_img_dir = "C:/Users/SATYO/Music/DCD_21/cutout-img/cutout"
    model_img_dir = "C:/Users/SATYO/Music/DCD_21/model-img/model"
    
    # list the images in these directories
    cutout_images = os.listdir(cutout_img_dir)
    model_images = os.listdir(model_img_dir)
    
    # load 10 Random Cutout Images: Sample out 10 images randomly from the above list
    sample_cutout_images = sample(cutout_images,10)
    fig = plt.figure(figsize=(10, 5))
    
    print("==============Cutout Images==============")
    for i, img_name in enumerate(sample_cutout_images):
        plt.subplot(2, 5, i+1)
        img_path = os.path.join(cutout_img_dir, img_name)
        loaded_img = image.load_img(img_path)
        img_array = image.img_to_array(loaded_img, dtype='int')
        plt.imshow(img_array)
        plt.axis('off')
        
    plt.show()
    print()
    # load 10 Random Model Images: Sample out 10 images randomly from the above list
    sample_model_images = sample(model_images,10)
    fig = plt.figure(figsize=(10,5))
    
    print("==============Model Images==============")
    for i, img_name in enumerate(sample_model_images):
        plt.subplot(2, 5, i+1)
        img_path = os.path.join(model_img_dir, img_name)
        loaded_img = image.load_img(img_path)
        img_array = image.img_to_array(loaded_img, dtype='int')
        plt.imshow(img_array)
        plt.axis('off')
        
    plt.show()


# Join the images with path and add in the dataframe

# Store the directory path in a varaible
cutout_img_dir = "C:/Users/SATYO/Music/DCD_21/cutout-img/cutout"
model_img_dir = "C:/Users/SATYO/Music/DCD_21/model-img/model"

# list the directories
cutout_images = os.listdir(cutout_img_dir)
model_images = os.listdir(model_img_dir)


def extractImageName(x):
    
    # 1. Invert the image path
    x_inv = x[ :: -1]
    
    # 2. Find the index of '/'
    slash_idx = x_inv.find('/')
    
    # 3. Extract the text after the -slash_idx
    return x[-slash_idx : ] 

listing_data['cutOutimageNames'] = listing_data['images.cutOut'].apply(lambda x : extractImageName(x))
listing_data['modelimageNames'] = listing_data['images.model'].apply(lambda x : extractImageName(x))


# Extract only those data points for which we have images
listing_data = listing_data[listing_data['cutOutimageNames'].isin(cutout_images)]
listing_data = listing_data[listing_data['modelimageNames'].isin(model_images)]

# Reset the index
listing_data.reset_index(drop=True, inplace=True)

# Add entire paths to cutOut and modelImages
listing_data['cutOutImages_path'] = cutout_img_dir + '/' + listing_data['cutOutimageNames']
listing_data['modelImages_path'] = model_img_dir + '/' + listing_data['modelimageNames']

# Drop the cutOutimageNames, cutOutimageNames
listing_data.drop(['cutOutimageNames', 'cutOutimageNames'], axis=1, inplace=True)

# Plot the images along with product descriptions, price and brand
random_idx = np.random.randint(low = 0, high = listing_data.shape[0] - 1)
cutOut_img_path = listing_data.iloc[random_idx]['cutOutImages_path']
model_img_path = listing_data.iloc[random_idx]['modelImages_path']
price = listing_data.iloc[random_idx]['priceInfo.formattedFinalPrice']
desc = listing_data.iloc[random_idx]['shortDescription']
brand = listing_data.iloc[random_idx]['brand.name']

# Load the images
cutOut_img = image.load_img(cutOut_img_path)
cutOut_img_arr = image.img_to_array(cutOut_img, dtype='int')
model_img = image.load_img(model_img_path)
model_img_arr = image.img_to_array(model_img, dtype='int')

# Plot the images
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
print("The images show a {}".format(desc))
print("Price: {}".format(price))
print("Brand: {}".format(brand))
print()
ax[0].imshow(cutOut_img_arr)
ax[1].imshow(model_img_arr)
ax[0].axis('off')
ax[1].axis('off')
plt.show()


def testModel(input_file):
    
    # Testing the architectures on external images
    
    '''Read the inserted url'''
    
    testing_img = Image.open(input_file)
    
    
    testing_features = resnet_feature_extractor.extract_features(testing_img)
    
    similarity_images_resnet = {}
    for idx, feat in image_features_resnet.items():

        # Compute the similarity using Euclidean Distance
        similarity_images_resnet[idx] = np.sum((queryFeatures_Resnet - feat)**2) ** 0.5

    # Extracting the top 10 similar images
    similarity_resnet_sorted = sorted(similarity_images_resnet.items(), key = lambda x : x[1], reverse=False)
    top_10_indexes_resnet = [idx for idx, _ in similarity_resnet_sorted][ : 8]
    
    # Plotting the images
    top_10_similar_imgs_Resnet = listing_data.iloc[top_10_indexes_resnet]['modelImages_path']
    brand_Resnet = listing_data.iloc[top_10_indexes_resnet]['brand.name']
    price_Resnet = listing_data.iloc[top_10_indexes_resnet]['priceInfo.formattedFinalPrice']
    desc_Resnet = listing_data.iloc[top_10_indexes_resnet]['shortDescription']
    
    print("===================== QUERY IMAGE ==========================")
    plt.figure(figsize=(4,4))
    testing_img_arr = image.img_to_array(testing_img, dtype='int')
    plt.imshow(testing_img_arr)
    plt.show()
    

    fig = plt.figure(figsize=(10,5))
    print("===================== SIMILAR IMAGES ==========================")
    for i, (img_path, brand) in enumerate(zip(top_10_similar_imgs_Resnet,desc_Resnet)):
        plt.subplot(2, 4, i+1)
        img = image.load_img(img_path)
        img_arr = image.img_to_array(img, dtype='int')
        plt.imshow(img_arr)
        plt.xlabel(price)
        plt.title(brand)
        plt.axis('off')
    plt.show()

testModel.pickle = "testModel.pkl"

st.title("Image Similarity For E-Commerce")
st.subheader("Image Search")
st.markdown("---")
input_file = st.file_uploader("Input File")
camera_input = st.camera_input("Camera Input")

st.image()


with open(testModel.pickle, 'rb') as file:  
    testModel.pickle = pickle.load(file)

st.write(testModel)
st.image()

st.success("Done")