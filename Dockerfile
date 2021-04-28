FROM python:3.6

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT [ "python3" ]

CMD ["Deploy/app.py"]
