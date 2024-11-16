FROM python:3.10-slim
RUN apt-get update
RUN apt-get install -y g++ libopenblas-dev build-essential
RUN pip install pybind11

WORKDIR /python
COPY ./swish/ /python/swish
COPY ./Makefile /python/Makefile
RUN make swish

COPY ./setup.py /python/setup.py
COPY ./pyproject.toml /python/pyproject.toml
RUN pip install --no-cache setuptools build
RUN python3 -m build
RUN pip install /python/dist/Swish-*.whl

COPY ./tests.py /python/tests.py
RUN pip install --no-cache-dir tensorflow
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

CMD ["python", "/python/tests.py"]