FROM ermaker/keras

RUN pip install \
    jupyter \
    matplotlib \
    seaborn \
    pandas \ 
    keras \
    scikit-learn 

VOLUME /notebook
WORKDIR /notebook
EXPOSE 8888
CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --NotebookApp.allow_origin='*'