# English-Vietnamese Translation using Transformer

## 1. Installation

You should have CUDA installed with version >= 11.1.

```bash
conda create -n trans python=3.9
conda activate trans
bash install.sh
```

## 2. Live demo

...

## 3. Training from scratch

### 3.1. Dataset

Download the [dataset from TED](https://drive.google.com/uc?id=1Fuo_ALIFKlUvOPbK5rUA5OfAS2wKn_95) and put it in folder `./data`.

Expected structure:

```
./data/
    |-- train.en
    |-- train.vi
    |-- tst2012.en
    |-- tst2012.vi
    |-- tst2013.en
    |-- tst2013.vi
```

### 3.2. Training command

```bash
python ...
```

