---
layout: archive
permalink: /projects/
title: "Projects"
author_profile: true
---

<h3>Visualizing Markets with Natural Language Processing</h3>

![image-left](/images/app_screenshot2.png){: .align-left}

Built a Flask app to visually explore the AI & Machine Learning startup competitive market landscape using Natural Language Processing (NLP). Complementing traditional market mapping, this tool applies NLP and Vectorization to Crunchbase data in the Artificial Intelligence + Machine Learning sectors to compute relative market proximity from company descriptions. Language preparation & processing utilized spaCy, NLTK, CountVectorizer, LDA, t-distributed Stochastic Neighbor Embedding (T-SNE), and TfidfVectorizer. Visualization was built using Bokeh running on Flask and deployed with Heroku.

<a href="https://github.com/rwmyers46/Venture-Market-Proximity" class="btn btn--info">View on Github</a>
<a href="https://ai-ventures.herokuapp.com/" class="btn btn--success">Try on Heroku</a>

<hr>

<h3>Species Identification with Convolutional Neural Networks</h3>

![image-left](/images/deer.jpg){: .align-left}

<p>Applied convolutional neural networks with Keras / TensorFlow to compute regional species classification using imagery from Texas wildlife cameras and Microsoft Cognitive Services API. Data were stored in S3, <a href="https://rwmyers46.github.io/verify-labels-rekognition/">verified with AWS Rekognition</a>, processed on EC2 GPU, and presented with Tableau. The deployed model's functionality demonstrated text notifications to mobile devices using AWS SNS when species were positively identified.  </p>

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
