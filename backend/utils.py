import math
from pathlib import Path
import torch
import numpy as np
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import os
import json


# Define evaluation function
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    track_name_order: List[str],
    device: str = "cuda:0",
    output_dir: str = "evaluation_results",
    create_visualizations: bool = True,
    correlation_threshold: float = 0.0,
    model_resolution=1,
):
    """
    Evaluate the model by computing Pearson correlation between true and predicted values.

    Args:
        model: The trained PyTorch model
        test_loader: DataLoader containing test data
        device: Device to run evaluation on
        output_dir: Directory to save evaluation results
        create_visualizations: Whether to create visualizations
        correlation_threshold: Minimum correlation to visualize (if create_visualizations is True)

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    from clex.utils import untransform_then_unbin

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    model.to(device)

    # Initialize lists to store results
    all_true_values = []
    all_pred_values = []
    all_sample_info = []

    # Run inference
    with torch.no_grad():
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

            # Move to CPU and convert to float32 for analysis
            all_true_values.append(targets.float().cpu())
            all_pred_values.append(outputs.float().cpu())
            all_sample_info.append(sample_info)

    # Concatenate batches
    true_values = torch.cat(all_true_values, dim=0)
    pred_values = torch.cat(all_pred_values, dim=0)

    # Compute correlations
    num_samples = true_values.shape[0]
    num_tracks = true_values.shape[1]

    print("num tracks", num_tracks)

    # Initialize storage for correlations
    track_correlations = []
    sample_correlations = []
    valid_correlations_count = np.zeros(num_tracks)

    # Initialize storage for relative error
    sample_mrae = []
    track_mrae = []

    # Compute correlation for each track across all samples
    for track_idx in range(num_tracks):
        track_results_corr = []
        track_results_mrae = []

        for sample_idx in range(num_samples):
            # Get track values for this sample
            true_track = true_values[sample_idx, track_idx].numpy()
            pred_track = pred_values[sample_idx, track_idx].numpy()

            # Check if track contains valid values (not -10, which indicates missing data)
            if not np.all(true_track == -1 * model_resolution):
                # Mask for valid values
                valid_mask = true_track != -1 * model_resolution

                if (
                    np.sum(valid_mask) > 1
                ):  # Only compute correlation if we have enough valid points
                    true_valid = true_track[valid_mask]
                    pred_valid = pred_track[valid_mask]

                    # Calculate Pearson correlation
                    if (
                        np.std(true_valid) > 0 and np.std(pred_valid) > 0
                    ):  # Check for variance
                        corr, _ = pearsonr(true_valid, pred_valid)
                        track_results_corr.append(corr)

                        # Store correlation for this sample and track
                        sample_correlations.append(
                            {
                                "sample_idx": sample_idx,
                                "track_idx": track_idx,
                                "track_name": track_name_order[track_idx],
                                "correlation": corr,
                                "batch_sample_idx": sample_idx % test_loader.batch_size,
                                "batch_idx": sample_idx // test_loader.batch_size,
                            }
                        )

                        valid_correlations_count[track_idx] += 1

                    # Calculate mean relative absolute error
                    with np.errstate(divide="ignore", invalid="ignore"):
                        denom = np.where(
                            true_valid == 0, np.nan, true_valid
                        )  # mask zeros
                        rel_err = np.abs((pred_valid - true_valid) / denom)

                    mrae = np.nanmean(rel_err)
                    track_results_mrae.append(mrae)

                    sample_mrae.append(
                        {
                            "sample_idx": sample_idx,
                            "track_idx": track_idx,
                            "track_name": track_name_order[track_idx],
                            "mrae": mrae,
                        }
                    )

        # Append mean correlation for this track
        if track_results_corr:
            print("number of nan correlations:", np.sum(np.isnan(track_results_corr)))
            mean_corr = np.nanmean(track_results_corr)
            track_correlations.append(
                {
                    "track_idx": track_idx,
                    "track_name": track_name_order[track_idx],
                    "mean_correlation": mean_corr,
                    "median_correlation": np.median(track_results_corr),
                    "std_correlation": np.std(track_results_corr),
                    "num_valid_samples": len(track_results_corr),
                }
            )
        else:
            track_correlations.append(
                {
                    "track_idx": track_idx,
                    "track_name": track_name_order[track_idx],
                    "mean_correlation": float("nan"),
                    "median_correlation": float("nan"),
                    "std_correlation": float("nan"),
                    "num_valid_samples": 0,
                }
            )

        mean_mrae = np.nanmean(track_results_mrae)
        track_mrae.append(
            {
                "track_idx": track_idx,
                "track_name": track_name_order[track_idx],
                "mean_mrae": mean_mrae,
                "median_mrae": np.nanmedian(track_results_mrae),
                "std_mrae": np.std(track_results_mrae),
                "num_valid_samples": len(track_results_mrae),
            }
        )

    # Create a DataFrame for track correlations
    track_corr_df = pd.DataFrame(track_correlations)
    track_mrae_df = pd.DataFrame(track_mrae)

    # Convert NumPy types to native Python types for JSON serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    # Output metrics
    metrics = {
        "track_correlations": track_corr_df.to_dict("records"),
        "overall_mean_correlation": float(track_corr_df["mean_correlation"].mean()),
        "valid_tracks_count": int((valid_correlations_count > 0).sum()),
    }

    # Save metrics to JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    # Save track correlations to CSV
    track_corr_df.to_csv(
        os.path.join(output_dir, "track_correlations.csv"), index=False
    )

    # Save track mrae to CSV
    track_mrae_df.to_csv(os.path.join(output_dir, "track_mrae.csv"), index=False)

    # Display results in notebook
    display_results(track_corr_df, metrics, track_order=track_name_order)

    # Create and save visualizations
    if create_visualizations:
        # Plot track correlations
        plt.figure(figsize=(12, 6))
        plt.bar(range(num_tracks), track_corr_df["mean_correlation"])
        plt.xticks(range(num_tracks), [f"Track {i}" for i in range(num_tracks)])
        plt.xlabel("Track")
        plt.ylabel("Mean Pearson Correlation")
        plt.title("Mean Pearson Correlation by Track")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "track_correlations.png"))
        plt.show()

        # Convert sample correlations to DataFrame for easier analysis
        sample_corr_df = pd.DataFrame(sample_correlations)

        # Create visualization directory for individual samples
        viz_dir = os.path.join(output_dir, "sample_visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Find high-correlation samples for visualization
        if not sample_corr_df.empty:
            high_corr_samples = sample_corr_df[
                sample_corr_df["correlation"] > correlation_threshold
            ]

            # Group by sample and get mean correlation
            sample_mean_corr = (
                high_corr_samples.groupby("sample_idx")["correlation"]
                .mean()
                .reset_index()
            )

            # Take top 5 samples with highest mean correlation for notebook display
            top_samples = sample_mean_corr.sort_values(
                "correlation", ascending=False
            ).head(5)

            # Visualize top samples
            for _, row in top_samples.iterrows():
                sample_idx = int(row["sample_idx"])

                # Get data for this sample
                true = true_values[sample_idx].numpy()
                pred = pred_values[sample_idx].numpy()

                # Plot each track
                plt.figure(figsize=(15, 10))

                for track_idx in range(num_tracks):
                    true_track = true[track_idx]
                    pred_track = pred[track_idx]

                    # Skip tracks with no valid data
                    if np.all(true_track == -10):
                        continue

                    # Mask for valid values
                    valid_mask = true_track != -10

                    plt.subplot(math.ceil(num_tracks / 3), 3, track_idx + 1)
                    plt.plot(
                        np.arange(len(true_track))[valid_mask],
                        true_track[valid_mask],
                        "b-",
                        label="True",
                    )
                    plt.plot(
                        np.arange(len(pred_track))[valid_mask],
                        pred_track[valid_mask],
                        "r-",
                        label="Predicted",
                    )

                    # Compute correlation for this track in this sample
                    if (
                        np.sum(valid_mask) > 5
                        and np.std(true_track[valid_mask]) > 0
                        and np.std(pred_track[valid_mask]) > 0
                    ):
                        corr, _ = pearsonr(
                            true_track[valid_mask], pred_track[valid_mask]
                        )
                        plt.title(f"Track {track_idx} (Corr: {corr:.3f})")
                    else:
                        plt.title(f"Track {track_idx}")

                    if track_idx == 0:
                        plt.legend()

                plt.suptitle(
                    f"Sample {sample_idx} (Mean Corr: {row['correlation']:.3f})"
                )
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"sample_{sample_idx}.png"))
                plt.show()

            # Create predictions vs. true value scatter plots for the best sample
            if len(top_samples) > 0:
                best_sample_idx = int(top_samples.iloc[0]["sample_idx"])
                true = true_values[best_sample_idx].numpy()
                pred = pred_values[best_sample_idx].numpy()

                plt.figure(figsize=(15, 10))
                for track_idx in range(num_tracks):
                    true_track = true[track_idx]
                    pred_track = pred[track_idx]

                    # Skip tracks with no valid data
                    if np.all(true_track == -10):
                        continue

                    # Mask for valid values
                    valid_mask = true_track != -10

                    if np.sum(valid_mask) > 5:
                        plt.subplot(math.ceil(num_tracks / 3), 3, track_idx + 1)
                        plt.scatter(
                            true_track[valid_mask], pred_track[valid_mask], alpha=0.5
                        )

                        # Add diagonal line
                        min_val = min(
                            true_track[valid_mask].min(), pred_track[valid_mask].min()
                        )
                        max_val = max(
                            true_track[valid_mask].max(), pred_track[valid_mask].max()
                        )
                        plt.plot([min_val, max_val], [min_val, max_val], "k--")

                        # Compute correlation
                        if (
                            np.std(true_track[valid_mask]) > 0
                            and np.std(pred_track[valid_mask]) > 0
                        ):
                            corr, _ = pearsonr(
                                true_track[valid_mask], pred_track[valid_mask]
                            )
                            plt.title(f"Track {track_idx} (Corr: {corr:.3f})")
                        else:
                            plt.title(f"Track {track_idx}")

                        plt.xlabel("True Values")
                        plt.ylabel("Predicted Values")

                plt.suptitle("True vs Predicted Values (Best Sample)")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "true_vs_pred_best_sample.png"))
                plt.show()

    return metrics, track_corr_df, true_values, pred_values

def load_model(base_path: str, skip_data=False):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import pandas as pd

    # Import your model classes and dataset
    from clex.dataset import GenomicDataset

    model_path = f"{base_path}/model_best.pth"
    cfg_path = f"{base_path}/train_config.json"

    from clex.train import Config

    # Determine model type from path
    if "baseline" in model_path:
        from clex.models.baseline import DNAConvNet

        model = DNAConvNet()
    elif "clex" in model_path or "borzoi" in model_path:
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
    else:
        raise ValueError(f"Could not determine model type from path: {model_path}")

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    if not skip_data:
        # Load dataset
        samples = pd.read_pickle(cfg.path_to_samples)

        # num_tracks = samples.iloc[0]["track_values"].shape[0]

        # if cfg.fwd_track_only:
        #     samples["track_values"] = samples["track_values"].apply(
        #         lambda x: x[: (num_tracks // 2)]
        #     )

        #     num_tracks = num_tracks // 2

        #     print(
        #         "Training on ",
        #         samples.iloc[0]["track_values"].shape,
        #         "tracks (fwd only).",
        #     )

        # Filter for test set
        test_samples = samples[samples["fold"] == "test"]
        test_dataset = GenomicDataset(test_samples, resolution=1, split_name="test")

        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=10,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        with open(f"{base_path}/track_annotation.json", "r") as f:
            row_annotation = json.load(f)
            track_order = row_annotation["+"] + row_annotation["-"]

        return model, test_loader, track_order
    return model, None, None
