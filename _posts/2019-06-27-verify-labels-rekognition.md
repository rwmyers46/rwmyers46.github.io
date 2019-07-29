---
title: "Image Labeling with AWS Rekognition"
date: 2019-06-27
tags: [AWS Rekognition, Image Preprocessing, Boto3]
excerpt: "Amazon Web Services, Machine Learning, Data Science"
---

<img src="/images/rekognition/photo-pile.jpg" alt="drawing" height="200"/>
<figcaption>Photo by @jontyson on Unsplash</figcaption>

* AWS Rekognition can be leveraged as an image processing utility for a variety of data science applications
* Verifying image labels with Rekognition is a simple, time-saving hack when building neural networks

The Rekognition API is an Amazon Web Services (AWS) microservice designed for computer vision applications. As part of AWS's Machine Learning suite, Rekognition provides scalable, on-demand image and video AI processing. Rekognition has been successfully applied to everything from flagging adult content to facial recognition for law enforcement. So as an AWS Solutions Architect studying data science, I wanted to see if Rekognition could also help build a better convolutional neural network. *Spoiler Alert - it can!*

<img src="/images/rekognition-2.png"/>
<figcaption>AWS Rekognition</figcaption>

When building a neural network for image detection, ensuring that training images are consistent with the desired label is the simplest way to increase model accuracy. Just as a Japanese character masquerading as the 27th letter would it confuse a child learning the English alphabet, training neural networks with mislabeled data will reduce a model's performance.

Training neural networks for image recognition requires at least 800 high-quality photos per class. For example, if you're building some AI to distinguish between two animals, plan on gathering about 3,000 images (20% will be useless or redundant and you'll need an extra 30% for the null class).

Compared with numerical or categorical data, which can be screened for inconsistent types and null values with minimal code, working with imagery is more intricate:

1. How do you know all training images are representative of your target label?
2. Does your training imagery contain multiple instances of the target label, extraneous objects or other noise likely to confuse your model?
3. Are there redundant images with unique filenames in your directory? ([solution: see related post](https://rwmyers46.github.io/image-duplicates/))

Traditionally, these questions could not be confidently answered without a human manually reviewing each and every image. Oftentimes, this time-intensive, repetitive work was hired out via platforms like AWS Mechanical Turk, but even delegation takes time.

So in lieu of spending a weekend tediously reviewing thousands of training images, I decided to enlist AWS Rekognition for the task of verifying image labels. It worked flawlessly - and took less than a minute.

##### Step 1 - Instantiate S3 and Rekognition Boto3 Clients:

Before proceeding, install Boto 3 and separate images by label in an S3 bucket directory.

In the code block below, we first instantiate `s3_client` with boto3 and set `bucket_name` to the S3 bucket containing the training imagery. If the images are in folder, we must also set the `prefix` variable to the precise path.  

 ```python
  s3_client = boto3.client('s3')

  bucket_name = 'your-bucket-name'
  prefix = '/images-directory-path/'

  rek_client = boto3.client(
      "rekognition",
      aws_access_key_id = "your_key_id",
      aws_secret_access_key = "your_secret_key",
      region_name = "us-east-1"
  )
```
##### Step 2 - Select Test Images:

Next we build a list containing those labels that Rekognition is associating with your desired label. To do this, select a few images in your bucket that exemplify the target label - ideally as single instances excluding other extraneous objects. Copy these image paths to the `test_images` list. In this example, I used 3 test images for a whitetail deer:

```python
  test_images = ['image-1-path, image-2-path, image-3-path']
```
##### Step 3 - Run Label Verification with Rekognition:

Unless you would like to modify the Confidence or MaxLabels parameters, the code below with handle the rest. The script requires user confirmation before deleting the files (note: I have yet to find a file mistakenly on deck for deletion).

```python
# Generate a test_labels list by loading photos consecutively
# and capturing Rekognition's response:

animal_list = []
test_labels = {}
keyString_list = []
bad_pics = 0

for img in test_images:
    response = rek_client.detect_labels(
        Image={
            'S3Object': {
                'Bucket': bucket_name,
                'Name': img,
            }
        },
        MaxLabels = 10,
    )
    for label in response['Labels']:
        if label['Confidence'] > 85:
            animal_list.append(label['Name'])

# create a set of unique image labels from our test images

test_labels = set(animal_list)

# create object type botocore.paginate.PageIterator
# from images in s3 bucket:

paginator = s3_client.get_paginator('list_objects_v2')
result = paginator.paginate(Bucket = bucket_name, Prefix = prefix)

# unpack the image file keystrings from the paginator results:

for page in result:
    if "Contents" in page:
        for key in page[ "Contents" ]:
            keyString = key[ "Key" ]
            keyString_list.append(keyString)

# call Rekognition with the file's keyString:

            try:
                rek_response = rek_client.detect_labels(
                    Image={
                        'S3Object': {
                            'Bucket': bucket_name,
                            'Name': keyString,
                        }
                    },
                    MaxLabels = 10,
                )
# append response labels > 85% confidence to labels_list:

                labels_list = []
                for label in rek_response['Labels']:
                    if label['Confidence'] > 85:
                        labels_list.append(label['Name'])
```
In the last segment below, I also wanted to remove photos containing people. With Rekognition, this task was as simple as adding the `or ('Person' in test_labels)` clause to the conditional statement.

```python
# compare labels_list to test_labels and remove images
# lacking evidence of our desired subject:

    labels_list = set(labels_list)
    if (not labels_list.intersection(test_labels)) or ('Person' in test_labels):
        s3_client.delete_object(Bucket = bucket_name, Key = keyString)
        bad_pics += 1
except:
    print('Bad image:', keyString)

print('{} images processed'.format(len(keyString_list)))
print('Deleted {} images.'.format(bad_pics))
```
##### Pink Elephants & Whitetail Deer:

So if AWS Rekognition has perfected computer vision, why build your own models at all? While Rekognition is a 90% solution for most image and video analysis, specific use cases require custom models. Wildlife species identification is one example case currently outside of Rekognition's scope.

<img src="/images/article-deer.jpg"/>

Testing Rekognition with images of a Whitetail Deer returns "impala", an aesthetically similar animal, but from a different taxonomy family and indigenous to another continent. Considering that dozens of Whitetail Deer sub-species are also spread throughout the Americas, we can conclude that hunting guides are safely outside disruption's path.

So while Rekognition is broadly accurate, more specific applications may require a model with deeper discernment. But for verifying image labels, Rekognition is a fast, accurate, and inexpensive alternative to traditional methods.
