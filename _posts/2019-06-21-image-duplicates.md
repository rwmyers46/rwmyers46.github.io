---
title: "Optimizing Duplicates Processing"
date: 2019-06-21
tags: [data science, image processing, python]
excerpt: "Time Complexity, Machine Learning, Algorithms"
mathjax: "true"
head: <script type="text/javascript" src="path-to-MathJax/MathJax.js"></script>

---

<img src="/images/optimizer/eagle2.jpg" alt="drawing" height="200"/>

* Neural networks trained with duplicate images exhibit decreased accuracy
* A brute force approach to eliminating image duplicates has exponential time complexity
* This algorithm exhibits top-level processor efficiency gain of 167x

Thousands of images are required to train robust neural networks. If redundant images are present in your training set, model accuracy will suffer. Even with a repository of unique images, a single null file may cause the build to abort with errors - after waiting through several fitting epochs. Thus, ensuring images are unique, uncorrupted, and adhere to expected file formats is imperative.

Python's CV2 and OS modules can readily handle null image files; the real challenge lies in avoiding duplicate images. Even if all images within a freshly downloaded set are unique, we must ensure they do not duplicate existing images in your training directory. This can only be confirmed with a pairwise comparison of each image.

The number of comparisons required can be computed using graph theory for a complete, undirected graph and treating images as (n) nodes with combinations of (K) edges:

$$K_n = \frac{n*(n - 1)}{2}$$

This equation dictates that in order to ensure 1,000 images are unique, we must process nearly half a million image pairs. Starting with $$O(N^2)$$ complexity, the processing load further increases because images must be compared pixel-by-pixel, across each pixel's 3 RGB color channels.

To visualize this, meet my dog Jimi. In order to programmatically determine whether the images on the right and left are the same, every pixel must match across each color channel:

<img src="/images/optimizer/jimi_v2.jpg"/>

*Sidenote: For the standard image encoding color depth of 24-bits, each pixel can assume one of $$2^{24} = 16,777,216$$ colors. In lieu of comparing every pixel, we could exploit this vast number of colors to create additional processing efficiency. Assuming the color spectrum is evenly utilized, the probability that 1 randomly selected pixel from the same location of two 100 x 100 images would possess identical colors if they were not identical is ~ 1 in 6.7 billion. However, the color spectrum is not evenly utilized, so this method would first require computing the color distribution of a specific image set which is beyond the scope of this post.*

So assuming that a pixelated pairwise comparison is elemental to confirming image uniqueness, I concluded the only way to increase computational efficiency was to reduce the number of image pairs.

The `Image_Optimizer` operates by finding each image's dimensional size and grouping images with equivalent sizes so that pairwise comparisons are only required within each group.

Taking a list of (file_name, file_size) tuples as the argument, it first finds the number of unique sizes. These sizes are used as keys for two purposes: to create a `counts` dictionary to count the number of unique size occurrences and for a `filtered_dict` dictionary to which we will append filenames.

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
The `filtered_dict` of {image_dimensions : file_names list} represents a grouped list of images with identical dimensions. This dictionary is returned so that the `Check_Duplicates` function only need compare images of the same size to each other.

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

Result:

* On a test sample of 846 images, image processing requirements were reduced from 357,435 pairs for the brute force approach to 2,139 image pairs.
