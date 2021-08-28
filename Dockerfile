FROM jupyter/scipy-notebook

RUN pip install \
    tensorflow \
    numpy \
    opencv-python \
    mediapipe

VOLUME /notebook
WORKDIR /notebook
EXPOSE 8888
CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --NotebookApp.allow_origin='*'