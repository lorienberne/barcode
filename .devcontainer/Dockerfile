FROM python:3.8-slim-buster


RUN pip install -U albumentations 
RUN pip install -U opencv-python
RUN pip install -U torch torchvision
RUN pip install -U pyzbar
RUN pip install -U pillow
RUN pip install -U python-barcode
RUN pip install -U ipykernel


RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*
