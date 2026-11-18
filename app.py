"""Gradio demo: upload a PCB photo, get a defect heatmap and verdict.

    pip install -r requirements-demo.txt
    python app.py --model patchcore --ckpt outputs/patchcore.pt

The decision threshold defaults to the best-F1 value in outputs/results.json
(run scripts/evaluate.py first), or pass --threshold explicitly.
"""
import argparse
from pathlib import Path

import gradio as gr

from pcbad.factory import make_autoencoder, make_patchcore, to_numpy_predictor
from pcbad.inference import Calibration, describe, detect, load_calibration


def build_demo(predict_np, size: int, calib: Calibration) -> "gr.Blocks":
    def analyze(image, threshold):
        if image is None:
            raise gr.Error("Please upload a PCB image first.")
        try:
            d = detect(image, predict_np, size, float(threshold), calib.vmin, calib.vmax)
        except (ValueError, TypeError) as e:
            raise gr.Error(f"Could not process the image: {e}")
        return d.overlay, describe(d)

    with gr.Blocks(title="PCB Anomaly Detection") as demo:
        gr.Markdown(
            "# PCB anomaly detection\n"
            "Unsupervised inspection: the model has only ever seen **defect-free** boards. "
            "Red areas in the heatmap look different from anything it learned as normal."
        )
        with gr.Row():
            inp = gr.Image(type="numpy", label="PCB image")
            out_img = gr.Image(label="Anomaly heatmap")
        thr = gr.Slider(0.0, calib.vmax * 2, value=calib.threshold, step=calib.threshold / 100,
                        label="Decision threshold (higher = fewer alarms)")
        verdict = gr.Textbox(label="Verdict", interactive=False)
        gr.Button("Inspect", variant="primary").click(analyze, [inp, thr], [out_img, verdict])
        gr.Markdown("*Limits: trained on one board type; lighting/camera changes shift scores, "
                    "so re-calibrate the threshold for a new setup.*")
    return demo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["autoencoder", "patchcore"], default="patchcore")
    p.add_argument("--ckpt", default=None, help="default: outputs/<model>.pt")
    p.add_argument("--image-size", type=int, default=256, help="used for PatchCore")
    p.add_argument("--results", default="outputs/results.json")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--share", action="store_true")
    a = p.parse_args()

    ckpt = a.ckpt or f"outputs/{a.model}.pt"
    if not Path(ckpt).exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}. Train a model first (see README).")
    factory = make_autoencoder if a.model == "autoencoder" else make_patchcore
    predict, size = factory(ckpt, a.device, a.image_size)

    if a.threshold is not None:
        calib = Calibration(a.threshold, 0.0, 2 * a.threshold)
    elif Path(a.results).exists():
        calib = load_calibration(a.results, a.model)
    else:
        raise SystemExit("No threshold: run scripts/evaluate.py first or pass --threshold.")

    build_demo(to_numpy_predictor(predict), size, calib).launch(share=a.share)


if __name__ == "__main__":
    main()
