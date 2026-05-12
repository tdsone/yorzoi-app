import sys
import modal

cuda_version = "12.6.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .pip_install(
        "yorzoi==0.2.1",
        "fastapi[standard]",
        "orjson"
    )
)

app = modal.App("yorzoi-app")

MINUTES = 60

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
        if not processed_seqs:
            return {}

        window_size = 4992
        step_size = 1000
        prediction_length = 3000
        prediction_offset = (window_size - prediction_length) // 2  # 996

        # Build one batch from all short sequences (padded to 4992) and all
        # chunks of all long sequences — they share shape [4, 4992] so they
        # can go through the model in a single forward pass.
        batch_items: list = []
        seq_meta: list = []
        for original_seq, seq_to_process, left_pad, is_long in processed_seqs:
            if not is_long:
                start = len(batch_items)
                batch_items.append(GenomicDataset.one_hot_encode(seq_to_process))
                seq_meta.append({
                    "kind": "short",
                    "original_seq": original_seq,
                    "left_pad": left_pad,
                    "batch_start": start,
                })
            else:
                original_len = len(original_seq)
                chunks = [
                    original_seq[i : i + window_size]
                    for i in range(0, original_len - window_size + 1, step_size)
                ]
                if (original_len - window_size) % step_size != 0:
                    chunks.append(original_seq[-window_size:])
                if not chunks:
                    chunks.append(original_seq)

                start = len(batch_items)
                for chunk in chunks:
                    batch_items.append(GenomicDataset.one_hot_encode(chunk))
                seq_meta.append({
                    "kind": "long",
                    "original_seq": original_seq,
                    "original_len": original_len,
                    "batch_start": start,
                    "num_chunks": len(chunks),
                })

        batch_tensor = torch.stack([
            torch.as_tensor(item, dtype=torch.float32) for item in batch_items
        ]).to("cuda:0")

        with torch.autocast(device_type="cuda"):
            preds_squashed = model(batch_tensor)
            preds = untransform_then_unbin(preds_squashed, resolution=10)

        predictions_dict = {}
        for meta in seq_meta:
            if meta["kind"] == "short":
                pred_3000 = preds[meta["batch_start"]]
                num_tracks = pred_3000.shape[0]
                original_seq = meta["original_seq"]
                original_len = len(original_seq)
                left_pad = meta["left_pad"]

                full_pred_4992 = torch.zeros(
                    num_tracks, window_size,
                    device=pred_3000.device, dtype=pred_3000.dtype,
                )
                full_pred_4992[:, prediction_offset : prediction_offset + prediction_length] = pred_3000

                cropped_pred = full_pred_4992[:, left_pad : left_pad + original_len]
                predictions_dict[original_seq] = cropped_pred.tolist()
            else:
                original_seq = meta["original_seq"]
                original_len = meta["original_len"]
                num_chunks = meta["num_chunks"]
                seq_preds = preds[meta["batch_start"] : meta["batch_start"] + num_chunks]
                num_tracks = seq_preds.shape[1]

                prediction_aggregator = torch.zeros(
                    num_tracks, original_len,
                    device=seq_preds.device, dtype=seq_preds.dtype,
                )
                overlap_counter = torch.zeros(
                    1, original_len,
                    device=seq_preds.device, dtype=seq_preds.dtype,
                )

                is_tail_chunk = (original_len - window_size) % step_size != 0
                for i in range(num_chunks):
                    is_last_chunk = i == num_chunks - 1
                    if is_last_chunk and is_tail_chunk:
                        start_pos_in_original = original_len - window_size
                    else:
                        start_pos_in_original = i * step_size

                    prediction_start = start_pos_in_original + prediction_offset
                    prediction_end = prediction_start + prediction_length

                    prediction_aggregator[:, prediction_start:prediction_end] += seq_preds[i]
                    overlap_counter[:, prediction_start:prediction_end] += 1

                overlap_counter[overlap_counter == 0] = 1
                final_prediction = prediction_aggregator / overlap_counter
                predictions_dict[original_seq] = final_prediction.tolist()

        print(
            f"request {req_id} completed in {round(time.monotonic() - start_time, 2)} seconds"
        )
        return predictions_dict

    return app