FROM python:3.13-slim

RUN pip install -U pip
COPY ./requirements.txt /
RUN pip install -r ./requirements.txt --upgrade

COPY . /

CMD ["python", "predict.py"]
