# Hand-Gesture-Recognition-ASL
**Master degree's Computer-Vision course project:** Hand gesture recognition, translation from (ASL) american-sign-language to text. <br>
**By:** [Yehuda Yadid](https://www.linkedin.com/in/yehuda-yadid/) and [Eliran Shem Tov](https://www.linkedin.com/in/eliranshemtov/)
*****

## Abstract
In this project we:
1. Loaded a dataset of 78,000 ASL (American Sign Language) images.
2. Pre-processed the dataset and evaluated the different classes distribution.
3. Applied data augmentation techniques to widen the dataset's variety and variance.
4. Created a convolutional neural network (CNN).
5. Trained the CNN to accurately classify letters from the ASL (American Sign Language), on a given image.
6. Validated the model's training results on a test set.
7. Applied the same model on a simple application that takes a video-stream from the endpoint's camera as input, identifies the presence of a hand in the frame, and attempts to translate the ASL sign.
*****
## Dataset
* We first based our model training and test phases on the following dataset, taken from Kaggle (*27,455 images*): 
[Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist?select=sign_mnist_test) <br>
Since this dataset is very synthetic, it's usage led to poor results when testing with an external test set. 

* Therefore, we chose to re-train the model against a larger and less  synthetic dataset (*78,000 images*):
[ASL Alphabet Image dataset](https://www.kaggle.com/grassknoted/asl-alphabet) 
*****
## Results:
* XXX
* XXX
* XXX
*****
## How to run
There are few options to run this project's artifacts:
* ### Colab:
    * Load ```./notebook/main.ipynb``` Jupyter-Notebook to [Google-Colab](https://colab.research.google.com/)
    * No need to explain, though you'll be able to explore the notebook only, without the "video" app.

* ### Locally
    1. Git clone this repository.
    2. Within the repository's directory, run: ```python -m venv ./venv``` 
    3. Run: ```pip install requirements.txt```
    4. Open the project in VSCode or other Jupyter-Notebook explorer. OR install Jupyter locally by following [this guide](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/install.html)
    5. Explore the notebook at ./notebook/main.ipynb
    6. To run the app that classifies hand signs on frames from a video stream, run: ```./venv/bin/python3 video_stream_prediction.py```

* ### In Docker
    1. Git clone this repository.
    2. Within the repository's directory, run: ```docker build --tag translate-asl .```
    3. run: ```docker run translate-asl```
    4. To explore the notebook, open your browser and go to http://localhost:8888
    5. To run the app that classifies hand signs on frames from a video stream:
        -  run: ```docker exec -it translate-asl bash``` 
        - when inside the container, run: ```python video_stream_prediction.py```


****
© [Yehuda Yadid](https://www.linkedin.com/in/yehuda-yadid/) and [Eliran Shem Tov](https://www.linkedin.com/in/eliranshemtov/) | MTA | Aug-21 <br>
This is an educational course project, not for production.
