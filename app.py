"""Entrypoint for the ViHate demo, locally or as a Hugging Face Space.

    uv run app.py           # plain launch
    uv run gradio app.py    # hot-reloading dev server

`gradio app.py` finds `demo` by matching the literal `with gr.Blocks(...) as demo`
below, so keep that line as is. Set MODEL_ID to a Hub repo id or local path to serve a
different checkpoint.
"""

import gradio as gr

from model import Classifier
from ui import TITLE, build_layout

classifier = Classifier()

with gr.Blocks(title=TITLE) as demo:
    build_layout(classifier)

if __name__ == "__main__":
    demo.launch(max_file_size="10mb")
