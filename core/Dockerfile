ARG BASE_IMAGE="pytorch/pytorch"
FROM $BASE_IMAGE
WORKDIR /app/
ENV TZ=Europe/London

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

COPY core ./core

COPY config.json ./
ENTRYPOINT ["python", "main.py"]
