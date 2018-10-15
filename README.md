### (in progress)

## Introduction

In this project I resolve image-classification problem of detecting age class (young, middle or old) based on person's face images. I use convolutional neural networks, which is well-proven approach in deep learning to work with image datasets.
I intentionally decided to work on small training dataset, which is common real-world project use case.

All needed theoretical knowledge about Convolutional Neural Networks to fully understand each step in the project is covered in this [Andrew Ng course on Coursera](https://www.coursera.org/learn/convolutional-neural-networks/home/welcome). I also recommend chapter 5 in book [Deep Learning with Python by François Chollet](https://www.manning.com/books/deep-learning-with-python), which helped me to complete this project.  

### Dataset description

Indian Movie Face database (IMFDB) is a large unconstrained face database consisting of 34512 images of 100 Indian actors collected from more than 100 videos. All the images are manually selected and cropped from the video frames resulting in a high degree of variability interms of scale, pose, expression, illumination, age, resolution, occlusion, and makeup. IMFDB is the first face database that provides a detailed annotation of every image in terms of age, pose, gender, expression and type of occlusion that may help other face related applications. For more information and link to download can be found [here](http://cvit.iiit.ac.in/projects/IMFDB/). 

<img src="images/dataset_intro.PNG" width="900">

### Prepare dataset by sample small dataset size and organize file structure. 

For simplicity I dataset is downloaded from [datahack.analyticsvidhya.com](https://datahack.analyticsvidhya.com/contest/practice-problem-age-detection/). The labels has been already converted to a multiclass problem with classes as Young, Middle and Old.

Since I will use [`ImageDataGenerator.flow_from_directory()`](https://keras.io/preprocessing/image/) to preprocess data, it expects data to be organized in following directiories and subdirectories.  



```
train_young_dir = os.path.join(train_dir, 'young')
os.mkdir(train_young_dir)

validation_young_dir = os.path.join(validation_dir, 'young')
os.mkdir(validation_young_dir)
```

<img src="images/files_structure.PNG" width="100">

Then I random sample small available data to have assumpted small data size. 

```
# Young class

labels_young = labels.loc[labels.Class == 'YOUNG']
labels_young = labels_young.sample(frac=1)

for i, row in labels_young.iloc[:1600].iterrows():
    src = os.path.join(train_all, row.ID)
    dst = os.path.join(train_young_dir, row.ID)
    shutil.copy(src, dst)
    
for i, row in labels_young.iloc[1600:2400].iterrows():
    src = os.path.join(train_all, row.ID)
    dst = os.path.join(validation_young_dir, row.ID)
    shutil.copy(src, dst)
```

```
Total training young images: 1600
Total training middle images: 1600
Total training old images: 1600
Total validation young images: 800
Total validation middle images: 800
Total validation old images: 796
```

For full code go to  [data_preparing.ipynb](https://github.com/ksulima/Age_Detection_Convolutional_NN/blob/master/notebooks/data_preparing.ipynb)

## Steps:
- Prepare dataset by sample small dataset size and organize file structure. `data_preparing.ipynb`
- Image preprocessing with ImageDataGenerator implemented in Keras.
- Build CNN baseline model
- Include some regularization techniques to model (Dropout and BatchNormalization) 
- Use Data augmentation to mitigate overfitting.
- Use some well-known convolutional networks with weights pre-trained on ImageNet to build model with higher performance. 
- Fine-tuning weights in a few top layers network.
- Conclusions
