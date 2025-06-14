FROM rl_backend

RUN python -m pip install scalene

ENTRYPOINT ["python", "-m", "scalene", "--html", "main.py"]
