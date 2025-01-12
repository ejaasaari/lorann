FROM python:3.12

ARG PIP_NO_CACHE_DIR=True

WORKDIR /app

RUN apt-get update && apt-get install -y g++
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install lorann

CMD python -c "import lorann; print('LoRANN has been installed')"
