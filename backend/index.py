import sys
import modal

cuda_version = "12.6.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.9")
    .pip_install( 
        "yorzoi==0.0.3",
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp39-cp39-linux_x86_64.whl"
    )
)

app = modal.App("yorzoi-app")

MINUTES = 1

@app.cls(
    gpu="A100",
    timeout=20 * MINUTES,
    scaledown_window=20 * MINUTES,
    image=image,
)
@modal.concurrent(max_inputs=100)
class Model:

    @modal.enter()
    def start_runtime(self):
        from yorzoi.model.borzoi import Borzoi
        self.model = Borzoi.from_pretrained("tom-ellis-lab/yorzoi")
        self.model.to("cuda:0")
        self.model.eval()

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, request: dict) -> str:
        import time
        from uuid import uuid4

        start = time.monotonic_ns()
        request_id = uuid4()
        print(f"Generating response to request {request_id}")

        import random
        import torch
        from yorzoi.dataset import GenomicDataset
        from yorzoi.utils import untransform_then_unbin

        # Check if all sequences are valid (length 4992)
        for sequence in request["sequences"]:
            if len(sequence) != 4992:
                raise ValueError(f"Sequence length must be 4992, got {len(sequence)}")
    
        # Make sure sequences are all uppercase
        request["sequences"] = [sequence.upper() for sequence in request["sequences"]]

        # Make sure sequences are valid DNA sequences
        for sequence in request["sequences"]:
            if not all(c in "ACGT" for c in sequence):
                raise ValueError(f"Sequence contains invalid characters: {sequence}")

        sequences = torch.stack([torch.tensor(GenomicDataset.one_hot_encode(sequence), dtype=torch.float32) for sequence in request["sequences"]])

        print(f"\nPredicting RNA-seq coverage for {sequences.shape[0]} sequences\n")

        sequences = sequences.to("cuda:0")

        with torch.autocast(device_type="cuda"):
            predictions_squashed_scale = self.model(sequences)
            # We remove the binning and squashing transformation to get predictions at 1bp resolution in the original scale.
            predictions: torch.Tensor = untransform_then_unbin(predictions_squashed_scale, resolution=10)

        print(f"Made predictions for {predictions.shape[0]} sequences, {predictions.shape[1]} tracks each (sense: {predictions.shape[1]//2}, antisense: {predictions.shape[1]//2}), \nacross {predictions.shape[2]} bins (resolution=10 => 3000 bp). The first half of the tracks are \nthe sense strand, the second half are the antisense strand. The first track \nof the first half belongs to the same RNA-seq experiment as the first track \nof the second half and so on.\n")

        print(
            f"request {request_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )

        return [predictions[i].tolist() for i in range(predictions.shape[0])]

    @modal.exit()  # what should a container do before it shuts down?
    def shutdown_runtime(self):
        self.runtime.shutdown()