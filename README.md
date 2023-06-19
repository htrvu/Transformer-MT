# English-Vietnamese Translation using Transformer

## 1. Installation
<span id='install'></span>

You should have CUDA installed with version >= 11.1.

```bash
conda create -n trans python=3.9
conda activate trans
bash install.sh
```

## 2. Demo Application
<span id='demo'></span>

Download resources from this [link]() and put it in folder `./runs`.

Expected structure:

```
./runs/
    |-- folder_names/
        |-- config.yaml
        |-- best.pt
        |-- src_field.pt
        |-- trg_field.pt
        |-- ...
```

Run:

```bash
streamlit run app.py ./runs/folder_names
```

## 3. Training from scratch

### 3.1. Dataset
<span id='dataset'></span>

Download the [dataset from TED](https://drive.google.com/file/d/1y9udEJSwe9eqPSSSt79GImD3Ai-o9nV4/view?usp=sharing) and put it in folder `./data`.

Expected structure:

```
./data/
    |-- train.en
    |-- train.vi
    |-- val.en
    |-- val.vi
    |-- test.en
    |-- test.vi
```

### 3.2. Training command
<span id='train'></span>

```bash
python ...
```

## 4. Evaluation
<span id='eval'></span>

...
