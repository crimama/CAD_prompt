# RUN apt-get update && apt-get install -y \
#     build-essential \
#     cmake \
#     git \
#     wget \
#     curl \
#     ca-certificates \
#     libjpeg-dev \
#     libpng-dev \
#     libtiff-dev \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# Python 3.8 설치
# RUN apt-get update && \
#     apt-get install -y software-properties-common && \
#     add-apt-repository ppa:deadsnakes/ppa && \
#     apt-get update && \
#     apt-get install -y python3.8 python3.8-dev python3.8-distutils && \
#     ln -s /usr/bin/python3.8 /usr/bin/python3

# pip 설치
# RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8

# numpy 및 기타 종속성 패키지 설치
# RUN pip install --upgrade pip && \
#     pip install numpy==1.22.3 \
#                 mkl \
#                 mkl-include \
#                 typing-extensions

# # CUDA 툴킷 및 PyTorch 설치
# RUN pip install torch==1.12.1+cu102 \
#                 torchvision==0.13.1+cu102 \
#                 torchaudio==0.12.1+cu102 \
#                 -f https://download.pytorch.org/whl/cu102/torch_stable.html

# 나머지 conda 패키지와 유사한 필수 패키지 설치 (dependencies 목록 기반)
pip install \
    absl-py==1.2.0 \
    asttokens==2.1.0 \
    backcall==0.2.0 \
    blessings==1.7 \
    cachetools==5.2.0 \
    charset-normalizer==2.1.0 \
    click==8.1.3 \
    cycler==0.11.0 \
    decorator==5.1.1 \
    easydict==1.10 \
    efficientnet-pytorch==0.7.1 \
    einops==0.6.1 \
    executing==1.2.0 \
    faiss-gpu==1.7.2 \
    fonttools==4.33.3 \
    google-auth==2.9.1 \
    google-auth-oauthlib==0.4.6 \
    gpustat==0.6.0 \
    grpcio==1.47.0 \
    idna==3.3 \
    imageio==2.19.3 \
    imgaug==0.4.0 \
    importlib-metadata==4.12.0 \
    ipython==8.6.0 \
    jedi==0.18.1 \
    joblib==1.1.0 \
    kiwisolver==1.4.3 \
    kornia==0.6.8 \
    markdown==3.4.1 \
    matplotlib==3.5.2 \
    matplotlib-inline==0.1.6 \
    networkx==2.8.4 \
    nvidia-ml-py3==7.352.0 \
    oauthlib==3.2.0 \
    opencv-python==4.6.0.66 \
    packaging==21.3 \
    pandas==1.4.2 \
    parso==0.8.3 \
    pexpect==4.8.0 \
    pickleshare==0.7.5 \
    pip==23.2.1 \
    plotly==5.11.0 \
    prompt-toolkit==3.0.32 \
    protobuf==4.23.4 \
    psutil==5.9.1 \
    ptyprocess==0.7.0 \
    pure-eval==0.2.2 \
    pyasn1==0.4.8 \
    pyasn1-modules==0.2.8 \
    pygments==2.13.0 \
    pyparsing==3.0.9 \
    python-dateutil==2.8.2 \
    pytz==2022.1 \
    pywavelets==1.3.0 \
    pyyaml==6.0 \
    requests==2.28.1 \
    requests-oauthlib==1.3.1 \
    rsa==4.9 \
    scikit-image==0.19.3 \
    scikit-learn==1.1.1 \
    scipy==1.8.1 \
    setuptools==63.2.0 \
    shapely==1.8.2 \
    stack-data==0.6.1 \
    tabulate==0.9.0 \
    tenacity==8.1.0 \
    tensorboard==2.9.1 \
    tensorboard-data-server==0.6.1 \
    tensorboard-plugin-wit==1.8.1 \
    tensorboardx==2.6.1 \
    threadpoolctl==3.1.0 \
    tifffile==2022.5.4 \
    timm==0.6.7 \
    tqdm==4.64.0 \
    traitlets==5.5.0 \
    urllib3==1.26.10 \
    wcwidth==0.2.5 \
    werkzeug==2.1.2 \
    zipp==3.8.1