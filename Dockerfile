FROM ubuntu:latest

COPY . ./app
WORKDIR /app

RUN apt update; apt install -y sudo wget lsb-release software-properties-common gnupg curl ca-certificates

RUN wget -qO- https://apt.llvm.org/llvm.sh | sudo bash -s -- 21
RUN apt install -y libmlir-21-dev mlir-21-tools 

RUN curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
RUN apt install -y nodejs
RUN npm install

RUN sudo add-apt-repository -y ppa:deadsnakes/ppa; sudo apt install -y python3-pip python3.11-venv

RUN python3.11 -m venv mlir_venv

RUN mlir_venv/bin/pip install --upgrade pip
RUN mlir_venv/bin/pip install --pre torch-mlir torchvision \
      --extra-index-url https://download.pytorch.org/whl/nightly/cpu \
        -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels

RUN mlir_venv/bin/pip install fastapi uvicorn pytest httpx

EXPOSE 3000

CMD ["npm", "run", "start:all"]
