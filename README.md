# Siamese Retriever
Image retrieval is the process of searching for and retrieving images from a database or repository based on a query which is usually an image itself. It remains interesting problem in the domains of Information Retrieval and Computer Vision. This is the official repository for our Content-Based Image Retrieval system which uses a Siamese-ConvNet architecture backbone written in PyTorch.

# Pre-Trained Model
The computational resources consist of 1 Intel ® Core i7-10700, 16GB of RAM, 1 GPU NVIDIA GeForce GTX 1650 Ti with 4GB of memory. The model was trained on our architecture for our dataset [IF20K](https://drive.google.com/drive/folders/1GGyYYRRznMQ9XllWJyXFNtoDh4vK99kU?usp=share_link) and can be found [here](https://drive.google.com/file/d/1ZpAX8WalKw44wuNPF53sxGwQxgTuySAw/view?usp=share_link). 

# How To Run
Download the dataset and the pre-trained model.
```
cd app
streamlit run app.py
```

# Academic Integrity
If you are currently enrolled in this course, please refer to IIIT-Delhi's Policy on Academic Integrity before referring to any of the repository contents. This repository contains the work we did as undergrads at IIIT-Delhi in CSE-508 Information Retrieval course. We do not encourage plagiarism of any kind.
