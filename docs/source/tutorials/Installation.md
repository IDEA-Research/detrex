# Installation
LiBai provides an editable installation way for you to develop your own project based on LiBai's framework.

## Build LiBai from Source

- Clone this repo:

```bash
git clone https://github.com/Oneflow-Inc/libai.git
cd libai
```
- Create a conda virtual environment and activate it:

```bash
conda create -n libai python=3.7 -y
conda activate libai
```

- Install the stable release of OneFlow with `CUDA` support. See [OneFlow installation guide](https://github.com/Oneflow-Inc/oneflow#install-with-pip-package).
- Install `pybind11`:

```bash
pip install pybind11
```

- For an editable installation of LiBai:

```bash
pip install -e .
```


