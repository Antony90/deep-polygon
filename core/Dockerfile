ARG BASE_IMAGE="pytorch/pytorch"
FROM $BASE_IMAGE

WORKDIR /app/
ENV TZ=Europe/London

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt
RUN python -m pip install py-spy

COPY core .
COPY config.json ./

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]