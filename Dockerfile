FROM python:3.6

WORKDIR /model

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY Models/ConditionalGAN .

#RUN tensorboard --logdir ./logs

ENTRYPOINT ["python", "./testing.py"]
