---
title: "Optimizing Duplicates Processing"
date: 2019-06-21
tags: [data science, image processing, python]
excerpt: "Time Complexity, Machine Learning, Algorithms"
---

<img src="/images/optimizer/eagle2.jpg" alt="drawing" height="200"/>

* This algorithm only required processing 2,139 image pairs compared to 357,435 pairs with a brute force approach on a test sample of 846 images.

* As less than 1% of the original combinations require further evaluation, the top-level processing efficiency gain is 167x.

1,000s of images are required to train a robust neural network. However, redundant images do not improve the model and contribute to overfitting. Further, a single null image file may cause the build to abort with errors after several epochs of fitting. Thus, ensuring images are unique, uncorrupted, and adhere to expected file formats is imperative.

Python's CV2 and OS modules can readily handle null image files; the real challenge lies in avoid duplicate images. To confirm uniqueness not only for newly downloaded images, but also to avoid adding duplicates to our existing image directory, a pairwise comparison of each image is required. The number of comparisons required can be computed using graph theory, treating images as nodes and combinations as edges.

The equation for calculating the number of edges K for a complete, undirected graph with n nodes: K = n(n - 1)/2.

$$\cos (2\theta)$$

Thus, to ensure 1,000 images are unique we must make 499,500 comparisons. Starting with O(N^2) complexity, the processing load is further increased when we compare each image pair pixel by pixel, each having 3 RGB color channels. Since these pixelated pairwise comparisons are (to my knowledge) elemental to confirming image uniqueness, the best way to achieve computational efficiency is to focus on reducing the number of images to compare. While graph theory holds in a "brute force" approach, images have more attributes than nodes - which we can exploit for our reductionist aims.

The `Image_Optimizer` function's core advantage is avoiding computationally intense image comparisons by focusing on the more superficial attribute of size. Taking a list of (file_name, file_size) tuples as the argument, it first finds the number of unique sizes. These sizes are used as keys for two purposes: to create a `counts` dictionary to count the number of unique size occurrences and for a `filtered_dict` dictionary to which we will append filenames.

The `filtered_dict` of {image_dimensions : file_names list} represents a grouped list of images with identical dimensions. This dictionary is returned so that the `Check_Duplicates` function only need compare images of the same size to each other.

```python
  import numpy as np
  import random
  import string
  from collections import Counter, defaultdict
  import cv2
  from PIL import Image
  import os
  import itertools as it
  import shutil
```

```python
# Group images by size to increase computational efficiency:

def Image_Optimizer(name_size):

    short_list = []
    files_investigate = {}
    counts = dict()

    # find unique image size values & store in a dictionary:

    size_list = [n[1] for m,n in enumerate(name_size)]  

    poss_dupes = set(size_list)     

    for m, n in enumerate(name_size):
        if n[1] in poss_dupes:           
            counts[n[1]] = counts.get(n[1], 0) + 1

    # remove items with values = 1 (no need to process images of unique sizes) to use as a checksum:

    short_list = [c for c in counts if counts[c] > 1]  

    # create a dictionary where keys = file size and values = lists of file names.
    # this reduces processing by ensuring images are only compared to others within groups of identical sizes

    for ns in name_size:
        if ns[1] in files_investigate:
             files_investigate[ns[1]].append(ns[0])
        else:
            files_investigate[ns[1]] = [ns[0]]

    # dictionary filters `files_investigate` for values > 1:

    filtered_dict = defaultdict(list)

    for k, v in files_investigate.items():
        if len(v) > 1:
            filtered_dict[k].append(v)

    if len(short_list) == len(filtered_dict):
        print('>> Created {} groups of images to compare\n'.format(len(short_list)))
        return(filtered_dict)
    else:
        print("Error detected.")
```
```python
# check for redundant images in <path> directory:

def Check_Duplicates(path):

    duplicates_list, corrupted_list, size_list, name_size = [], [], [], []
    comp_dict = {}

    # create list of all files in directory & check for errors:

    img_list = [i for i in os.listdir(path)]

    for i, j in enumerate(img_list):
        read = cv2.imread(path + j)

        # create list of tuples in filename, size format:

        try:
            temp = read.shape
            size_list = (j, temp)
            name_size.append(size_list)

        except:
            if j == '.ipynb_checkpoints':
                shutil.rmtree(path + j)     # remove jupyter labs checkpoint file if present
            else:
                print('Bad image found:', j)

    print('Original image list size:', len(name_size))

    # optimize processing by organizing images into groups of equal size:

    prepped_images = Image_Optimizer(name_size)   

    # conduct pairwise comparison of images sharing a key:

    for k, v in prepped_images.items():
        v = sum(v, [])                        # flatten values list
        img_combos = it.combinations(v, 2)    # create list of pairwise combinations from values

        print('Image combinations being processed:', len(list(img_combos)))

        for i, j in enumerate(img_combos):
            try:
                original = cv2.imread(path + j[0])
                duplicate = cv2.imread(path + j[1])

                if original.shape == duplicate.shape:    # double check that image dimensions equal

                # compute image differences and split by channel:

                    difference = cv2.subtract(original, duplicate)
                    b, g, r = cv2.split(difference)

                    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                        print('Images are completely Equal:', j)
                        duplicates_list.append(j)    # append duplicate filenames to list

            except:
                print('Bad file(s) detected:', fname)
                corrupted_list.append(fname)

    print('Duplicates: ', len(duplicates_list))
    print('Corrupted: ', len(corrupted_list))

    if len(duplicates_list) > 0:

        d_list = [i[0] for i in duplicates_list]

        for dl in duplicates_list:
            if dl[0] in comp_dict:
                comp_dict[dl[0]].append(dl[1])
            else:
                comp_dict[dl[0]] = [dl[1]]

        # compile a list of values from comp_dict to delete:

        dump_list = []

        for cd, v in comp_dict.items():
            dump_list.extend(v)

        print('Dump List:',len(dump_list))

        # remove duplicate values and convert to ordered data structure

        dump_list = list(set(dump_list))    
        print(dump_list)

        # user confirmation to delete files:

        user_answer = input('Remove duplicate files? [y] or [n]')

        if user_answer == 'y':
            ctr = 0
            for d in dump_list:
                os.remove(path + d)
                ctr += 1
            print('Deleted {} files.'.format(ctr))
```
