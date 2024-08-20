docker run -it --rm --env-file .env --mount type=bind,src=./models,target=/usr/src/app/models --gpus all offline-train "$@"
