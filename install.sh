pip install --no-cache-dir torch==1.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install torchtext==0.11.0 --no-deps
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz --no-deps
pip install -r requirements.txt
python -m pip install -e .