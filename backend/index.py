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
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp39-cp39-linux_x86_64.whl",
        "fastapi[standard]"
    )
)

app = modal.App("yorzoi-app")

MINUTES = 1

@app.function(
    gpu="A100",
    timeout=20 * MINUTES,
    scaledown_window=20 * MINUTES,
    image=image,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    """ASGI FastAPI application with CORS enabled."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, conlist
    from typing import List
    import torch
    import time
    from uuid import uuid4
    import random
    from yorzoi.dataset import GenomicDataset
    from yorzoi.model.borzoi import Borzoi
    from yorzoi.utils import untransform_then_unbin

    # Load model once per container
    model = Borzoi.from_pretrained("tom-ellis-lab/yorzoi")
    model.to("cuda:0")
    model.eval()

    app = FastAPI(title="Yorzoi API with CORS")

    # --- CORS CONFIGURATION ---
    allowed_origins = [
        "https://yorzoi.eu",
        "https://www.yorzoi.eu",
        "http://localhost:5173",  # local dev
        "http://127.0.0.1:5173",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class GenerateRequest(BaseModel):
        sequences: List[str]

        def pad_and_validate(self) -> List[tuple[str, str, int]]:
            """Pad sequences shorter than 4992 bp with 'N' and validate characters/length."""
            processed = []
            for seq in self.sequences:
                import re
                original_seq = re.sub(r"\s+", "", seq.upper())
                original_len = len(original_seq)

                left_pad = 0
                # Pad symmetrically with 'N's if sequence is too short
                if original_len < 4992:
                    delta = 4992 - original_len
                    left_pad = delta // 2
                    right_pad = delta - left_pad  # handles odd delta
                    padded_seq = ("N" * left_pad) + original_seq + ("N" * right_pad)
                elif original_len > 4992:
                    raise ValueError(
                        f"Sequence length must be ≤ 4992, got {original_len}"
                    )
                else: # original_len == 4992
                    padded_seq = original_seq

                # Validate allowed characters (A, C, G, T, N)
                if not set(padded_seq).issubset({"A", "C", "G", "T", "N"}):
                    raise ValueError(f"Invalid characters in sequence: {seq}")

                processed.append((padded_seq, original_seq, left_pad))
            return processed

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        """Predict RNA-seq coverage for the provided DNA sequences."""
        start = time.monotonic_ns()
        req_id = uuid4()
        print(f"Generating response to request {req_id}")

        # Validation & preprocessing
        processed_seqs = req.pad_and_validate()
        padded_seqs = [s[0] for s in processed_seqs]

        # One-hot encode
        sequences_tensor = torch.stack([
            torch.tensor(GenomicDataset.one_hot_encode(seq), dtype=torch.float32)
            for seq in padded_seqs
        ]).to("cuda:0")

        with torch.autocast(device_type="cuda"):
            preds_squashed = model(sequences_tensor)
            preds = untransform_then_unbin(preds_squashed, resolution=10)

        print(
            f"request {req_id} completed in {round((time.monotonic_ns() - start) / 1e9, 2)} seconds"
        )

        # Map original (unpadded) sequences to their predictions
        predictions_dict = {}
        for i, (padded_seq, original_seq, left_pad) in enumerate(processed_seqs):
            original_len = len(original_seq)
            pred_3000 = preds[i]
            num_tracks = pred_3000.shape[0]

            # Create a zero tensor for the full 4992bp range
            full_pred_4992 = torch.zeros(
                num_tracks, 4992, device=pred_3000.device, dtype=pred_3000.dtype
            )
            # Place the 3000bp prediction in the center
            full_pred_4992[:, 996:3996] = pred_3000
            
            # Crop to the original sequence length from the correct start position
            cropped_pred = full_pred_4992[:, left_pad : left_pad + original_len]

            predictions_dict[original_seq] = cropped_pred.tolist()

        return predictions_dict

    return app