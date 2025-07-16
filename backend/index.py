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
        "fastapi[standard]", 
        "orjson"
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
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import ORJSONResponse
    from pydantic import BaseModel, conlist
    from typing import List
    import torch
    import time
    from uuid import uuid4

    from yorzoi.dataset import GenomicDataset
    from yorzoi.model.borzoi import Borzoi
    from yorzoi.utils import untransform_then_unbin

    # Load model once per container
    model = Borzoi.from_pretrained("tom-ellis-lab/yorzoi")
    model.to("cuda:0")
    model.eval()

    app = FastAPI(title="Yorzoi API", default_response_class=ORJSONResponse)

   

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

    app.add_middleware(GZipMiddleware, minimum_size=1024)

    class GenerateRequest(BaseModel):
        sequences: List[str]

        def process_sequences(self) -> List[tuple[str, str, int, bool]]:
            """
            Validates sequences, pads short ones, and flags long ones.
            Returns a list of tuples: (original_seq, processed_seq, left_pad, is_long)
            """
            processed = []
            for seq in self.sequences:
                import re
                original_seq = re.sub(r"\s+", "", seq.upper())
                original_len = len(original_seq)

                # Validate allowed characters (A, C, G, T, N)
                if not set(original_seq).issubset({"A", "C", "G", "T", "N"}):
                    raise ValueError(f"Invalid characters in sequence: {seq}")

                if original_len > 4992:
                    processed.append((original_seq, original_seq, 0, True))
                else:
                    left_pad = 0
                    delta = 4992 - original_len
                    left_pad = delta // 2
                    right_pad = delta - left_pad
                    padded_seq = ("N" * left_pad) + original_seq + ("N" * right_pad)
                    processed.append((original_seq, padded_seq, left_pad, False))
            return processed

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        """Predict RNA-seq coverage for the provided DNA sequences."""
        start_time = time.monotonic()
        req_id = uuid4()
        print(f"Generating response to request {req_id}")

        processed_seqs = req.process_sequences()
        predictions_dict = {}

        for original_seq, seq_to_process, left_pad, is_long in processed_seqs:
            if not is_long:
                # Use existing logic for short sequences
                sequence_tensor = torch.tensor(
                    GenomicDataset.one_hot_encode(seq_to_process), dtype=torch.float32
                ).unsqueeze(0).to("cuda:0")
                
                with torch.autocast(device_type="cuda"):
                    preds_squashed = model(sequence_tensor)
                    preds = untransform_then_unbin(preds_squashed, resolution=10)

                pred_3000 = preds[0]
                num_tracks = pred_3000.shape[0]
                original_len = len(original_seq)

                full_pred_4992 = torch.zeros(
                    num_tracks, 4992, device=pred_3000.device, dtype=pred_3000.dtype
                )
                full_pred_4992[:, 996:3996] = pred_3000
                
                cropped_pred = full_pred_4992[:, left_pad : left_pad + original_len]
                predictions_dict[original_seq] = cropped_pred.tolist()
            else:
                # Implement sliding window for long sequences
                window_size = 4992
                step_size = 1000
                prediction_length = 3000
                prediction_offset = (window_size - prediction_length) // 2 # 996

                original_len = len(original_seq)
                
                # Create chunks
                chunks = []
                for i in range(0, original_len - window_size + 1, step_size):
                    chunks.append(original_seq[i : i + window_size])
                
                if (original_len - window_size) % step_size != 0:
                    chunks.append(original_seq[-window_size:])

                if not chunks:
                    # This case handles sequences that are shorter than window_size but were flagged as long.
                    # This shouldn't happen with the current logic but is a safeguard.
                    chunks.append(original_seq)

                sequences_tensor = torch.stack([
                    torch.tensor(GenomicDataset.one_hot_encode(chunk), dtype=torch.float32)
                    for chunk in chunks
                ]).to("cuda:0")

                with torch.autocast(device_type="cuda"):
                    preds_squashed = model(sequences_tensor)
                    preds = untransform_then_unbin(preds_squashed, resolution=10)
                
                num_tracks = preds.shape[1]

                prediction_aggregator = torch.zeros(
                    num_tracks, original_len, device=preds.device, dtype=preds.dtype
                )
                overlap_counter = torch.zeros(
                    1, original_len, device=preds.device, dtype=preds.dtype
                )

                for i, pred_chunk in enumerate(preds):
                    is_last_chunk = i == len(chunks) - 1
                    is_tail_chunk = (original_len - window_size) % step_size != 0

                    if is_last_chunk and is_tail_chunk:
                        start_pos_in_original = original_len - window_size
                    else:
                        start_pos_in_original = i * step_size
                        
                    prediction_start = start_pos_in_original + prediction_offset
                    prediction_end = prediction_start + prediction_length

                    prediction_aggregator[:, prediction_start:prediction_end] += pred_chunk
                    overlap_counter[:, prediction_start:prediction_end] += 1
                
                overlap_counter[overlap_counter == 0] = 1
                
                final_prediction = prediction_aggregator / overlap_counter
                predictions_dict[original_seq] = final_prediction.tolist()

        print(
            f"request {req_id} completed in {round(time.monotonic() - start_time, 2)} seconds"
        )
        return predictions_dict

    return app