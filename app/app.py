import streamlit as st
from utils.model import (
    load_config,
    load_tokenizer,
    load_model,
    generate_output
)
from utils.utils import (
    load_local_css,
    load_remote_css,
    load_local_text,
    load_examples,
)
from utils.templates import DISABLED_TEXTAREA
from utils.preprocessing import cleaning

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME_OR_PATH = 'm3hrdadfi/bert2bert-fa-news-headline'
MODEL_DESC = load_local_text(BASE_DIR + '/assets/description.txt')
CODE_DESC = load_local_text(BASE_DIR + '/assets/code_snippet.txt')


def main(page_title, page_icon, layout='centered', initial_sidebar_state='expanded'):
    # Config the streamlit app
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state
    )

    # Load styles/assets
    load_remote_css("https://cdn.jsdelivr.net/gh/rastikerdar/vazir-font/dist/font-face.css")
    load_local_css(BASE_DIR + '/assets/style.css')
    load_local_css(BASE_DIR + '/assets/rtl.css')

    # Sidebar
    # st.sidebar.image(logo, width=120)
    st.sidebar.markdown('<h1 class="text-center">News Headline Generation</h1>', unsafe_allow_html=True)
    st.sidebar.markdown(MODEL_DESC, unsafe_allow_html=True)

    # Load examples
    ex_names, ex_map = load_examples(BASE_DIR + '/assets/example.json')

    # Main
    example = st.selectbox('Choose an example', ex_names)
    height = min((len(ex_map[example][0].split()) + 1) * 2, 200)
    sequence = st.text_area('Text', ex_map[example][0], key='sequence', height=height)

    if len(sequence) == 0:
        st.write('Enter some text to see generated headline.')
        return

    with st.spinner('Loading the components ...'):
        config = load_config(MODEL_NAME_OR_PATH)
        tokenizer = load_tokenizer(MODEL_NAME_OR_PATH)
        model = load_model(MODEL_NAME_OR_PATH, config=config)
        # model = load_model(MODEL_NAME_OR_PATH)

    with st.spinner('Generating ...'):
        headline = generate_output(model, tokenizer, cleaning(sequence))
        headline = headline[0] if len(headline) > 0 else headline
        st.markdown(DISABLED_TEXTAREA.format(**{
            'attributes': 'class="generated-headline rtl"',
            'value': f'<span class="r-label">عنوان اصلی</span> {example}'
        }), unsafe_allow_html=True)
        st.markdown(DISABLED_TEXTAREA.format(**{
            'attributes': 'class="generated-headline rtl"',
            'value': f'<span class="r-label">عنوان تولید شده</span> {headline}'
        }), unsafe_allow_html=True)


if __name__ == '__main__':
    main(
        page_title='News Headline Generation',
        page_icon='Φ',
        layout='wide',
        initial_sidebar_state='expanded',
    )
