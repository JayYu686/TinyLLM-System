FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/tinyllm

COPY requirements/torch-cpu.txt requirements/torch-cpu.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements/torch-cpu.txt

COPY LICENSE README.md pyproject.toml ./
COPY src/ src/
RUN python -m pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 tinyllm
USER tinyllm

ENTRYPOINT ["tinyllm"]
CMD ["--help"]
