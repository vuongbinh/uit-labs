"""Gradio entrypoint for the ViHSD hate-speech demo.

    uv run gradio app.py                                   # hot-reloading dev server
    uv run python app.py                                   # plain launch
    DEMO_MODEL=you/vihsd-visobert uv run gradio app.py     # serve a Hub bundle

`demo` is exposed at module level because that is what `gradio app.py` looks for
(its --demo-name defaults to `demo`). The canonical entrypoint remains
`uv run vihate demo --model <bundle>`, which has richer flags; this file exists for
the conventional Gradio workflow and for hosts that expect an app.py.

DEMO_MODEL defaults to outputs/demo_bundle -- the directory Part 7B of
notebooks/vihate_project_run_fnal.ipynb writes.
"""

import os

from vihate.serve_demo import HateSpeechPredictor, build_app

MODEL = os.environ.get("DEMO_MODEL", "outputs/demo_bundle")

predictor = HateSpeechPredictor(MODEL, device=os.environ.get("DEMO_DEVICE"))
demo = build_app(predictor)

if __name__ == "__main__":
    # allowed_paths: Gradio only serves files under the cwd or the system temp dir,
    # so the batch tab's CSV export needs the output directory granted explicitly.
    demo.launch(
        server_name=os.environ.get("DEMO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("DEMO_PORT", "7860")),
        share=os.environ.get("DEMO_SHARE") == "1",
        show_error=True,
        allowed_paths=[str(predictor.output_dir)],
    )
