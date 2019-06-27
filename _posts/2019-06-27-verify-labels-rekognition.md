---
title: "Label Verification with AWS Rekognition"
date: 2019-06-27
tags: [AWS Rekognition, machine learning, boto3]
header:
  image: "/images/rekognition/photo-pile.jpg"
excerpt: "Machine Learning, Data Science"
---

When training a neural network with images downloaded from the web, my experience has been that at least 15% of those photos are not not germane to your label of interest. Verifying image labels with AWS Rekognition is a simple, fast, and inexpensive way to increase model accuracy.

Setup: This post assumes you have an AWS account with your images in an S3 bucket and Boto 3 installed.

##### Step 1 - Instantiate S3 and Rekognition Boto3 Clients:

 ```Python
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

Unless you would like to modify the Confidence or MaxLabels parameters, the code below with handle the rest. The script requires user confirmation before deleting the files, but I have yet to find a file mistakenly on deck for deletion.

In the example below, I also wanted to remove photos containing people. With Rekognition, this task was as simple as adding a `or ('Person' in test_labels)` clause to the conditional statement.

```Python
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
