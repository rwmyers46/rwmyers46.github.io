---
layout: archive
permalink: /projects/
title: "Projects"
author_profile: true
---
<h3>Species Identification with Convolutional Neural Networks</h3>

![image-left](/images/deer.jpg){: .align-left}

<p>Applied convolutional neural networks with Keras / TensorFlow to compute regional species classification using imagery from wildlife cameras and Microsoft Cognitive Services API. Data stored in S3, cleaned with AWS Rekognition, processed on EC2 GPU, and visualized with Tableau.</p>

<a href="https://github.com/rwmyers46/CNN-Species-Identification" class="btn btn--info">View on Github</a>

<hr>

<h3>Net Promoter Scores, Version 2.0</h3>

![image-left](/images/nps-guage-2.jpg){: .align-left}

Improved Net Promoter Scores fidelity for online reviews using NLP (Spacy, Gensim, CountVectorizer, TF-IDF), topic modeling (LDA/NMF/CorEx), sentiment analysis (VADER), and measured accuracy with Naive Bayes and Logistic Regression with data stored in MongoDB on AWS.

<a href="https://github.com/rwmyers46/Net-Promoter-Score-2.0" class="btn btn--info">View on Github</a>

<hr>

<h3>You Are What You Eat! Predicting Diets from Instacart Orders</h3>

![image-left](/images/paleo-image-2.jpg){: .align-left}


Engineered features from Kaggle Instacart dataset to classify users practicing a paleolithic diet.  Built classification model that leveraged Logistic Regression and weighted the F1 results. Data stored in PostgreSQL (AWS).

<a href="https://github.com/rwmyers46/Instacart-Diet-Classification" class="btn btn--info">Coming Soon!</a>

<hr>

<h3>Rural Land Valuation</h3>

![image-left](/images/cow-2.jpg){: .align-left}

Developed a Multivariate Regression model for farm & ranch land valuation using data scraped from online property listings with Python / BeautifulSoup and features engineered with Natural Language Processing and Google Cloud Platform's Maps API. Project includes use of Lasso, Ridge, ElasticNet, XGBoost, and Multilayer Perceptron regressors and parameter optimization with LassoCV, RidgeCV, Yellowbrick, and GridCV.

<a href="https://github.com/rwmyers46/Rural-Land-Valuation" class="btn btn--info">View on Github</a>

<hr>

<h3>Visualizing Venture Markets with NLP</h3>

![image-left](/images/app_screenshot2.png){: .align-left}

This project built an app to explore how Natural Language Processing (NLP) could be leveraged to discover insights that complement traditional market mapping. NLP & Vectorization were applied to data sourced from Crunchbase in the Artificial Intelligence / Machine Learning sector to establish and plot relative market proximity. Language preparation & processing utilized spaCy, NLTK, CountVectorizer, LDA, t-distributed Stochastic Neighbor Embedding (T-SNE), and TfidfVectorizer. The visualization was built using Bokeh running on Flask and deployed with Heroku.

<a href="https://github.com/rwmyers46/Venture-Market-Proximity" class="btn btn--info">View on Github</a>

<a href="https://ai-ventures.herokuapp.com/" class="btn btn--success">Try on Heroku</a>
