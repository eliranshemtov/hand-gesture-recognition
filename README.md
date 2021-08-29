# Hand-Gesture-Recognition-ASL
**Master degree's Computer-Vision course project:** Hand gesture recognition, translation from (ASL) american-sign-language to text. <br>
**By:** [Yehuda Yadid](https://www.linkedin.com/in/yehuda-yadid/) and [Eliran Shem Tov](https://www.linkedin.com/in/eliranshemtov/)
*****

## Abstract
In this project we made 3 major attempts to train a neural network to successfully classify american sign language (ASL) images. <br>
1. ### [ASL MNIST](https://www.kaggle.com/datamunge/sign-language-mnist) <br>
   First we used the ASL MNIST dataset, to train a CNN model and achieved "amazing" results of almost 100% accuracy.
   Unfortunately, this dataset is highly synthetic and is not close enough to real-world conditions. Therefore, we've made the next attempt.
   The notebook that details this process could be [found here](https://github.com/eliranshemtov/hand-gesture-recognition/blob/main/notebook/asl-mnist.ipynb)

2. ### [ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet) <br> 
   Then, we tried to improve the results and train a model with a better dataset that would be closer to normal non-lab environments. We used the ASL Alphabet    dataset which looks much more realistic and wide in range of angles.
   Although this attempt yielded much better results, we still aimed to improve the model's fitting, so we turned to the third and final attempt.
   The notebook that details this process could be [found here](https://github.com/eliranshemtov/hand-gesture-recognition/blob/main/notebook/asl-alphabet.ipynb)

3. ### Tailormade wide-range videos <br>
   Lastly we created a (slightly smaller) dataset, that was concluded by a set of videos of hand gestures from a wide set of angels. Each video was broken-down to a    set frames which the model trained on.
   Due to time constraints we limited ourselves to 3 ASL chars: 'A', 'B', 'W'
   The notebook that details this process could be [found here](https://github.com/eliranshemtov/hand-gesture-recognition/blob/main/notebook/main.ipynb)

*****
## Demo:
<img src="https://github.com/eliranshemtov/hand-gesture-recognition/blob/main/resources/demos/demo-predictions.gif" alt="video-demo" width="500"/>


## How to run
#### NOTE: The trained models could be found [here in this repository's releases section](https://github.com/eliranshemtov/hand-gesture-recognition/releases)

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
