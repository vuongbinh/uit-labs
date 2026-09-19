import gradio as gr

from model import Classifier
from ui import TITLE, build_layout

classifier = Classifier()

with gr.Blocks(title=TITLE) as demo:
    build_layout(classifier)

if __name__ == "__main__":
    demo.launch(max_file_size="10mb")
