"""Entrypoint for the ViHate demo: `uv run app.py` locally, or a Hugging Face Space.

`demo` is exposed at module level so `uv run gradio app.py` (hot reload) works too.
Set MODEL_ID to a Hub repo id or local path to serve a different checkpoint.
"""

from model import Classifier
from ui import build_demo

demo = build_demo(Classifier())

if __name__ == "__main__":
    demo.launch(max_file_size="10mb")
