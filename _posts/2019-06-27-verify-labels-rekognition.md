---
title: "Label Verification with AWS Rekognition"
date: 2019-06-27
tags: [AWS Rekognition, machine learning, boto3]
excerpt: "Amazon Web Services, Machine Learning, Data Science"
---
<figure>
<img src="/images/rekognition/photo-pile.jpg" alt="drawing" height="200"/>
<figcaption>Photo by @jontyson on Unsplash </figcaption>
</figure>

figure {
    display: inline-block;
    border: 1px dotted gray;
    margin: 20px; 
}
figure img {
    vertical-align: top;
}
figure figcaption {
    border: 1px dotted blue;
    text-align: center;
}

When training neural networks with images downloaded from the web, it is common for 10-25% of those photos to contain mislabeled categories. For example, when gathering images from Microsoft's Cognitive Services API to train a model with the query "wild boar," among the hundreds of downloaded images was a promotional poster from the film Wild Hogs:

<img src="/images/rekognition/wild-boar-movie.jpg"/>

Ensuring that training images are consistent with the desired label is the simplest way to increase model accuracy, but manually reviewing 1,000s of training data images is a tedious process. These slow, repetitive types of tasks are great candidates for AWS Mechanical Turk, but for image labeling, Rekognition is a fast, accurate, and inexpensive alternative.

<img src="/images/rekognition-2.png"/>
<figcaption>AWS Rekognition</figcaption>

The AWS Rekognition API is a microservice designed to handle all the image and video analysis most applications require. From the documentation, Rekognition's image capabilities include: "...identify the objects, people, text, scenes, and activities, as well as detect any inappropriate content. Amazon Rekognition also provides highly accurate facial analysis and facial recognition." This begs the question: why build your own models at all? While Rekognition is a 90% solution for most image and video analysis, specific use cases require custom models.

<img src="/images/article-deer.jpg"/>

Wildlife identification is one example currently outside of Rekognition's scope. Testing Rekognition with images of Whitetail Deer returns "impala", a somewhat aesthetically similar animal, but from a different taxonomy family and indigenous to another continent. In addition to the elk, mule deer, and caribou sharing the family Cervidae in North America, there are also dozens of Whitetail Deer sub-species spread throughout the Americas. So while Rekognition is broadly accurate, wildlife classification is a specific application which requires a model with deeper discernment.  

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

Next we build a list containing those labels that Rekognition is associating with your desired label. To do this, find 3 to 5 images in your bucket that exemplify the subject label, ideally as single instances excluding other objects. In this example, I used 3 test images for a whitetail deer. Copy these image paths to the `test_images` list.

```python
  test_images = ['image-1-path, image-2-path, image-3-path']
```
##### Step 3 - Run Label Verification with Rekognition:

Unless you would like to modify the Confidence or MaxLabels parameters, the code below with handle the rest. The script requires user confirmation before deleting the files (note: I have yet to find a file mistakenly on deck for deletion).

```python
# Generate a test_labels list by loading photos consecutively and capturing Rekognition's response.

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

# creat a set of unique image labels from our test images

test_labels = set(animal_list)

# create an object type botocore.paginate.PageIterator from images in s3 bucket:

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
# append response labels above confidence > 85% to labels_list:

                labels_list = []
                for label in rek_response['Labels']:
                    if label['Confidence'] > 85:
                        labels_list.append(label['Name'])
```
In the last segment below, I also wanted to remove photos containing people. With Rekognition, this task was as simple as adding a the `('Person' in test_labels)` clause to the conditional statement.

```python
# compare labels_list to test_labels and remove images lacking evidence of our desired subject:

                labels_list = set(labels_list)
                if (not labels_list.intersection(test_labels)) or ('Person' in test_labels):
                    s3_client.delete_object(Bucket = bucket_name, Key = keyString)
                    bad_pics += 1
            except:
                print('Bad image:', keyString)

print('{} images processed'.format(len(keyString_list)))
print('Deleted {} images.'.format(bad_pics))
```
