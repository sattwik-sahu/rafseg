# RAGSeg

Retrieval Augmented Few-Shot Segmentation for Offroad

*by [Karthik Nambiar](https://github.com/karthiknambiar29), [Agamdeep Singh](https://github.com/jnash10), [Sattwik Sahu](https://github.com/sattwik-sahu)*

**PI:** [Prof. Sujit P B](https://github.com/pbsujit)

## Abstract

_To be updated_

## Usage

### Installation

#### Clone Repo and Create Environment

```bash
git clone https://github.com/sattwik-sahu/rafseg.git
cd rafseg
python -m venv .venv --prompt=rafseg
. .venv/bin/activate
python -m pip install -U pip
```

#### Install Poetry Dependency Manager

```bash
python -m pip install poetry
python -m poetry shell
```

#### Install Packages

```bash
poetry install
```

### Usage

To run the pipeline for evaluation on your images and corresponding segmentation masks

1. Activate the virtual environment
    ```bash
    poetry shell
    ```
2. Run the vlm module
    ```bash
    poetry run python -m vlm \
    path/to/images_dir \
    path/to/masks/dir \
    --dev \
    --vector-store-path path/to/vector_store.pkl
    ```
---

Made with ❤️ in IISER Bhopal.
