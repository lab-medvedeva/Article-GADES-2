# Image of the GADES 2.0 benchmark: GADES, the baselines and the drivers of scripts/Benchmarking.
# Build and run instructions, including the NVIDIA Container Toolkit, are in the README.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git wget ca-certificates \
        software-properties-common dirmngr gnupg \
        libopenblas-dev liblapacke-dev libarmadillo-dev libhdf5-dev \
        libboost-dev libfftw3-dev libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc > /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
    && add-apt-repository -y "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        r-base r-base-dev python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

ARG ARRAYFIRE_VERSION=v3.10.0
RUN git clone --depth 1 --branch ${ARRAYFIRE_VERSION} --recursive https://github.com/arrayfire/arrayfire.git /opt/arrayfire-src \
    && cmake -S /opt/arrayfire-src -B /opt/arrayfire-src/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/arrayfire \
        -DAF_BUILD_CUDA=ON -DAF_BUILD_CPU=ON -DAF_BUILD_OPENCL=OFF \
        -DAF_BUILD_EXAMPLES=OFF -DAF_BUILD_FORGE=OFF -DBUILD_TESTING=OFF \
    && cmake --build /opt/arrayfire-src/build \
    && cmake --install /opt/arrayfire-src/build \
    && rm -rf /opt/arrayfire-src
ENV LD_LIBRARY_PATH=/opt/arrayfire/lib:${LD_LIBRARY_PATH}

ARG GADES_COMMIT=main
ARG GADES_CUDA_ARCH=sm_86
RUN git clone https://github.com/lab-medvedeva/GADES-main.git /opt/GADES \
    && git -C /opt/GADES checkout ${GADES_COMMIT} \
    && cmake -S /opt/GADES -B /opt/GADES/build \
        -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/opt/GADES/build -DGADES_CUDA_ARCH=${GADES_CUDA_ARCH} \
    && cmake --build /opt/GADES/build
ENV GADES_ROOT=/opt/GADES

RUN apt-get update && apt-get install -y --no-install-recommends \
        libuv1-dev libfreetype6-dev libpng-dev libjpeg-dev libtiff5-dev libfontconfig1-dev \
        libharfbuzz-dev libfribidi-dev libcairo2-dev libxt-dev \
    && rm -rf /var/lib/apt/lists/*
COPY install.R requirements.txt /srv/
ARG CRAN_MIRROR=https://cloud.r-project.org
ENV CRAN_MIRROR=${CRAN_MIRROR}
RUN Rscript -e 'options(timeout = 600, Ncpus = parallel::detectCores(), repos = c(CRAN = Sys.getenv("CRAN_MIRROR")))' \
        -e 'source("/srv/install.R")' \
        -e 'missing <- setdiff(packages, rownames(installed.packages()))' \
        -e 'if (length(missing) > 0) stop(paste("not installed:", paste(missing, collapse = ", ")))'
RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /srv/requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com "cuml-cu12==26.2.*"
ENV PATH=/opt/venv/bin:${PATH}

COPY . /workspace/Article-GADES-2
WORKDIR /workspace/Article-GADES-2/scripts/Benchmarking
RUN g++ -O3 -std=c++17 test_armadillo.cpp -o test_armadillo -larmadillo \
    && g++ -O3 -std=c++17 -I/opt/arrayfire/include test_arrayfire.cpp -o test_arrayfire -L/opt/arrayfire/lib -laf
