ARG BASE_IMAGE="pytorch/pytorch"
FROM $BASE_IMAGE
WORKDIR /usr/src/app/
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt snakeviz
COPY *.py ./
COPY env env/
EXPOSE 8080
ENTRYPOINT ["/bin/bash", "-c", "python -m cProfile -o train.prof main.py && echo 'Starting snakeviz' && snakeviz -s -H 0.0.0.0 train.prof"]
