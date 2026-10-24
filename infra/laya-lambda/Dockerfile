# syntax=docker/dockerfile:1
# Lambda has no GPU. This builds laya.cpp for the CPU backend and packages only
# its multilingual checkpoint (not the English checkpoint beside it locally).
FROM public.ecr.aws/amazonlinux/amazonlinux:2023 AS build

RUN dnf install -y \
    cmake \
    gcc-c++ \
    libicu-devel \
    ninja-build \
    python3-pip \
    && python3 -m pip install --no-cache-dir 'cmake>=3.24,<4' \
    && dnf clean all

RUN curl -fsSL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz -o /tmp/json.tar.xz \
    && python3 -c "import tarfile; tarfile.open('/tmp/json.tar.xz').extractall('/tmp')" \
    && cmake -S /tmp/json -B /tmp/json-build -DJSON_BuildTests=OFF \
    && cmake --install /tmp/json-build \
    && rm -rf /tmp/json /tmp/json-build /tmp/json.tar.xz

COPY local/laya.cpp/source/ /src/

RUN cmake -S /src -B /build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=OFF \
      -DLAYA_CUDA=OFF \
      -DLAYA_VULKAN=OFF \
    && cmake --build /build --parallel

FROM public.ecr.aws/lambda/python:3.12

RUN dnf install -y libicu && dnf clean all

COPY --from=build /build/bin/laya-cli /opt/laya/laya-cli
COPY --from=build /src/models/laya/multilingual /opt/laya/models/laya/multilingual
COPY infra/laya-lambda/handler.py ${LAMBDA_TASK_ROOT}/handler.py

ENV LAYA_BINARY=/opt/laya/laya-cli \
    LAYA_MODEL_DIR=/opt/laya/models/laya \
    LAYA_VARIANT=multilingual \
    LAYA_PORT=8080

CMD ["handler.lambda_handler"]
