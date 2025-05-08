import os
import time
import warnings
from typing import Optional
from uuid import uuid4

import modal

SGL_LOG_LEVEL = "error"  # try "debug" or "info" if you have issues

MINUTES = 60  # seconds

# We use the [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
# model by Alibaba.

MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_REVISION = "a7a06a1cc11b4514ce9edcde0e3ca1d16e5ff2fc"
TOKENIZER_PATH = "Qwen/Qwen2-VL-7B-Instruct"
MODEL_CHAT_TEMPLATE = "qwen2-vl"

# Modal runs Python functions on containers in the cloud.
# The environment those functions run in is defined by the container's `Image`.
# The block of code below defines our example's `Image`.
cuda_version = "12.6.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10.2")
    .pip_install(  # add sglang and some Python dependencies
        "torch"
    )
)

app = modal.App("clex-app")

@app.cls(
    gpu="A100",
    timeout=20 * MINUTES,
    scaledown_window=20 * MINUTES,
    image=image,
)
@modal.concurrent(max_inputs=100)
class Model:

    @modal.enter()  # what should a container do after it starts but before it gets input?
    def start_runtime(self):
        from utils import load_model_for_testing

        self.model = load_model_for_testing()
        self.model.eval()
        self.model.to("cuda")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, request: dict) -> str:

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        # Run inference
        with torch.no_grad():

            """
            TODO: take sequence from request and pass it to the model
            - padding, onehot encode, tensorize
            """

            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Extract data from batch
                sequences, targets = batch[0], batch[1]

                # Extract sample metadata
                sample_info = {
                    "indices": batch[2],
                    "chroms": batch[3],
                    "strands": batch[4],
                    "start_samples": batch[5],
                    "end_samples": batch[6],
                    "start_losses": batch[7],
                    "end_losses": batch[8],
                }

                sequences = sequences.to(device)
                targets = targets.to(device)

                # Forward pass - using autocast for mixed precision
                with torch.cuda.amp.autocast():
                    # Convert to half precision for FlashAttention compatibility
                    outputs = model(sequences)

                    # Handle squeeze if necessary (from the model architecture)
                    if len(outputs.shape) == 4:  # If shape is [batch, 1, tracks, bins]
                        outputs = outputs.squeeze(1)

                    # Unbin and untransform model outputs
                    outputs = untransform_then_unbin(outputs, resolution=model_resolution)
        print(
            f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )
        
        #TODO return results in json format specified in readme

        return state["answer"]

    @modal.exit()  # what should a container do before it shuts down?
    def shutdown_runtime(self):
        self.runtime.shutdown()


def load_model(base_path: str, skip_data=False):
    import torch
    import torch.nn as nn

    model_path = "model_best.pth"
    cfg_path = "train_config.json"

    from clex.train import Config

    from clex.models.yorzoi import Borzoi
    from clex.models.utils import TargetLengthCrop
    from clex.config import BorzoiConfig

    cfg = Config.read_from_json(cfg_path)

    config = BorzoiConfig(
        dim=cfg.borzoi_cfg["dim"],
        depth=cfg.borzoi_cfg["depth"],
        heads=cfg.borzoi_cfg["heads"],
        return_center_bins_only=cfg.borzoi_cfg["return_center_bins_only"],
        attn_dim_key=cfg.borzoi_cfg["attn_dim_key"],
        attn_dim_value=cfg.borzoi_cfg["attn_dim_value"],
        dropout_rate=cfg.borzoi_cfg["dropout_rate"],
        attn_dropout=cfg.borzoi_cfg["attn_dropout"],
        pos_dropout=cfg.borzoi_cfg["pos_dropout"],
        enable_mouse_head=cfg.borzoi_cfg["enable_mouse_head"],
        flashed=cfg.borzoi_cfg["flashed"],
        separable0=cfg.borzoi_cfg["separable0"],
        separable1=cfg.borzoi_cfg["separable1"],
        head=cfg.borzoi_cfg["head"],
        final_joined_convs=cfg.borzoi_cfg["final_joined_convs"],
        upsampling_unet0=cfg.borzoi_cfg["upsampling_unet0"],
        horizontal_conv0=cfg.borzoi_cfg["horizontal_conv0"],
    )

    model = nn.Sequential(TargetLengthCrop(4992), Borzoi(config))

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

        return model
    return model



@app.local_entrypoint()
def main(
    image_url: Optional[str] = None, question: Optional[str] = None, twice: bool = True
):
    import json
    import urllib.request

    model = Model()

    payload = json.dumps(
        {
            "image_url": image_url,
            "question": question,
        },
    )

    req = urllib.request.Request(
        model.generate.web_url,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req) as response:
        assert response.getcode() == 200, response.getcode()
        print(json.loads(response.read().decode()))

    if twice:
        # second response is faster, because the Function is already running
        with urllib.request.urlopen(req) as response:
            assert response.getcode() == 200, response.getcode()
            print(json.loads(response.read().decode()))

warnings.filterwarnings(  # filter warning from the terminal image library
    "ignore",
    message="It seems this process is not running within a terminal. Hence, some features will behave differently or be disabled.",
    category=UserWarning,
)
