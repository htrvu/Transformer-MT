import streamlit as st
import torch
from base.predictor import Predictor
from third_party.google_trans import GoogleTranslator
from transformer import Transformer
from utils import *
import dill
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("runs_path", type=str, help="Path to training result folder (e.g. runs/...)")
args = parser.parse_args()


model_output_text = None
gg_output_text = None

max_len = 200
beam_size = 1

def preprocess(input):
    input += '.'

    dict = {
        "'": " &apos;",
        # '"': " &quot; ",
    }

    for key, value in dict.items():
        input = input.replace(key, value)

    return input

@st.cache_resource
def load_model():
    '''
        Load Transformer model from checkpoint
    '''
    global max_len, beam_size

    config_path = os.path.join(args.runs_path, 'config.yaml')
    ckpt_path = os.path.join(args.runs_path, 'best.pt')
    src_field_path = os.path.join(args.runs_path, 'src_field.pt')
    trg_field_path = os.path.join(args.runs_path, 'trg_field.pt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_dict = load_config(config_path)
    max_len = config_dict['DATA']['MAX_LEN']
    beam_size = config_dict['PREDICTOR']['BEAM_SIZE']

    src_field = torch.load(src_field_path, pickle_module=dill)
    trg_field = torch.load(trg_field_path, pickle_module=dill)
    src_vocab_size = len(src_field.vocab)
    trg_vocab_size = len(trg_field.vocab)

    model = Transformer(config_path=config_path,
                        src_vocab_size=src_vocab_size,
                        trg_vocab_size=trg_vocab_size)
    
    model.load_state_dict(torch.load(ckpt_path))

    predictor = Predictor(model, src_field, trg_field, device=device)

    return predictor

@st.cache_resource
def load_gg_trans():
    translator = GoogleTranslator()
    return translator


def setup_page():
    st.set_page_config(
        page_title="Transformer for Machine Translation - CNTN20 - Vu & Thien",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def show_header():
    # Header
    st.markdown('''
    <h2 align="center">
        Statistical Learning - Final Project - CNTN20
        </br>
        Transformer for Machine Translation (English to Vietnamese)
    </h2>

    <h5 align="center">
        Hoàng Trọng Vũ - 20120025
        </br>
        Trần Hữu Thiên - 20120584
    </h5>
    ''', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.image('./images/app/banner.png', use_column_width=True)


def run_ui():
    global model_output_text, gg_output_text, max_len, beam_size

    # Input and output columns
    tmp, left_column, right_column, tmpp = st.columns([1, 2, 2, 1])
    with left_column:
        st.write("**Input sentence (in English, less than 200 words):**")
        input_text = st.text_area("", 
                                  height=100, 
                                  max_chars=max_len,
                                  label_visibility="collapsed")
        
        # Button
        if st.button('Translate'):
            if input_text.strip() == '':
                st.warning('Please input a sentence!')
                model_output_text = None
                gg_output_text = None
                return
            else:
                input_text = input_text.strip()
                model_output_text = load_model()(preprocess(input_text), max_len=max_len, beam_size=beam_size)
                # Avoid fail to connect to Google Translate
                while True:
                    try:
                        gg_output_text = load_gg_trans().translate(input_text, lang_src='en', lang_tgt='vi').lower()
                        break
                    except:
                        pass

    with right_column:
        st.write("**Output from model:**")
        if model_output_text is not None:
            st.write(model_output_text)

        st.write("**Output from Google Translate:**")
        if gg_output_text is not None:
            st.write(gg_output_text)

        


if __name__ == '__main__':
    setup_page()
    show_header()
    load_model()
    load_gg_trans()
    run_ui()