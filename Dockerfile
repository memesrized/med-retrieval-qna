FROM python:3.8-slim

RUN pip install --upgrade pip

ADD ./medretqna/ /
# cpu version is ~200mb and gpu is ~800, but you can't install cpu via requirements afaik
RUN pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /
EXPOSE 8000

CMD uvicorn main:app --reload --host 0.0.0.0
