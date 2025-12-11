# utils/compute_time_eval.py

import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Headless-friendly backend for servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TIMER_FILE_SUFFIX = "_timer_info.txt"


# ------------------------- internal helpers -------------------------

def _read_text(file_path: str) -> str:
    """Read text file robustly (tolerate weird encodings)."""
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            with open(file_path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            continue
    # Last resort: ignore errors
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _safe_float(s: Optional[str]) -> float:
    if s is None:
        return np.nan
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return np.nan


def _safe_int(s: Optional[str]) -> float:
    if s is None:
        return np.nan
    try:
        return int(str(s).strip())
    except Exception:
        return np.nan


def _re1(pattern: str, text: str, flags=re.IGNORECASE) -> Optional[str]:
    m = re.search(pattern, text, flags=flags)
    return m.group(1) if m else None


def _re2(pattern: str, text: str, flags=re.IGNORECASE) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(pattern, text, flags=flags)
    return (m.group(1), m.group(2)) if m else (None, None)


# ------------------------- core parsing -------------------------

def parse_timer_file(file_path: str, *, min_step_seconds: float = 1.0, debug: bool = False) -> Dict[str, float]:
    """
    Parse a single *_timer_info.txt file.
    - Never raises on missing fields; fills NaN instead.
    - Extracts totals, nodes, JSON sizes, and step times (> min_step_seconds).
    """
    content = _read_text(file_path)

    if debug:
        print(f"\n----- DEBUG: {os.path.basename(file_path)} -----")
        print(repr(content))

    metrics: Dict[str, float] = {}
    metrics["file_name"] = os.path.basename(file_path)

    # Dimensions (optional)
    w, h = _re2(r"Image Dimensions:\s*(\d+)\s*[xX]\s*(\d+)", content)
    metrics["width"] = _safe_int(w)
    metrics["height"] = _safe_int(h)

    # Total Time (regex then manual fallback)
    tt = _re1(r"Total Time:\s*([\d.]+)\s*seconds", content)
    if tt is None:
        for line in content.splitlines():
            if "Total Time:" in line:
                m = re.search(r"Total Time:\s*([\d.]+)", line, flags=re.IGNORECASE)
                if m:
                    tt = m.group(1)
                break
    metrics["total_time"] = _safe_float(tt)

    # Nodes & JSON sizes (optional)
    metrics["nodes_before"] = _safe_int(_re1(r"Total graph nodes \(before pruning\):\s*(\d+)", content))
    metrics["nodes_after"] = _safe_int(_re1(r"Total graph nodes \(after pruning\):\s*(\d+)", content))
    metrics["json_size_before_kb"] = _safe_float(_re1(r"JSON File Size \(before pruning\):\s*([\d.,]+)\s*KB", content))
    metrics["json_size_after_kb"]  = _safe_float(_re1(r"JSON File Size \(after pruning\):\s*([\d.,]+)\s*KB", content))

    # Step times: keep > threshold (optional)
    step_matches = re.findall(r"([A-Za-z &]+? check):\s*([\d.]+)\s*seconds", content, flags=re.IGNORECASE)
    for step_name, secs in step_matches:
        val = _safe_float(secs)
        if not np.isnan(val) and val > min_step_seconds:
            col = step_name.strip()  # e.g., "Detecting doors check"
            metrics[col] = val

    return metrics


# ------------------------- plots -------------------------

def _plot_image_area_vs_total_time(df: pd.DataFrame, out_dir: str) -> None:
    data = df.dropna(subset=["total_time", "width", "height"]).copy()
    if data.empty:
        return
    data["image_area"] = data["width"] * data["height"]

    plt.figure(figsize=(10, 6))
    plt.scatter(data["image_area"], data["total_time"], alpha=0.7)
    plt.xlabel("Image Area (pixels)", fontsize=13)
    plt.ylabel("Total Time (seconds)", fontsize=13)
    plt.title("Image Area vs Total Time", fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "image_area_vs_total_time.png"), dpi=300)
    plt.close()


def _plot_nodes_before_after(df: pd.DataFrame, out_dir: str) -> None:
    data = df.dropna(subset=["nodes_before", "nodes_after"]).copy()
    if data.empty:
        return

    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    for idx, (_, row) in enumerate(data.iterrows()):
        c = colors[idx % len(colors)]
        plt.scatter([idx], [row["nodes_before"]], c=[c], s=90, marker="o")
        plt.scatter([idx], [row["nodes_after"]],  c=[c], s=90, marker="D")
    plt.xlabel("File Index", fontsize=13)
    plt.ylabel("Number of Nodes", fontsize=13)
    plt.title("Nodes Before and After Pruning", fontsize=15)
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", label="Before Pruning",
                       markerfacecolor="black", markersize=9),
            plt.Line2D([0], [0], marker="D", color="w", label="After Pruning",
                       markerfacecolor="black", markersize=9),
        ],
        fontsize=11, loc="upper right",
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nodes_before_after_pruning.png"), dpi=300)
    plt.close()


def _plot_json_size_vs_nodes(df: pd.DataFrame, out_dir: str) -> None:
    data = df.dropna(subset=["nodes_before", "nodes_after",
                             "json_size_before_kb", "json_size_after_kb"]).copy()
    if data.empty:
        return

    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    for idx, (_, row) in enumerate(data.iterrows()):
        c = colors[idx % len(colors)]
        plt.scatter(row["nodes_before"], row["json_size_before_kb"], c=[c], s=90, marker="o")
        plt.scatter(row["nodes_after"],  row["json_size_after_kb"],  c=[c], s=90, marker="D")
    plt.xlabel("Number of Nodes", fontsize=13)
    plt.ylabel("JSON File Size (KB)", fontsize=13)
    plt.title("JSON File Size vs Number of Nodes", fontsize=15)
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", label="Before Pruning",
                       markerfacecolor="black", markersize=9),
            plt.Line2D([0], [0], marker="D", color="w", label="After Pruning",
                       markerfacecolor="black", markersize=9),
        ],
        fontsize=11, loc="upper right",
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "json_size_vs_nodes.png"), dpi=300)
    plt.close()


def _plot_average_step_times(df: pd.DataFrame, out_dir: str) -> None:
    step_cols = [c for c in df.columns if c.lower().endswith("check")]
    if not step_cols:
        return
    step_avg = df[step_cols].mean(numeric_only=True).dropna().sort_values()
    if step_avg.empty:
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(step_avg.index, step_avg.values, alpha=0.85)
    plt.xlabel("Steps", fontsize=13)
    plt.ylabel("Average Time (seconds)", fontsize=13)
    plt.title("Average Time for Significant Steps", fontsize=15)
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h + 0.05, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "average_step_times.png"), dpi=300)
    plt.close()


def generate_plots(df: pd.DataFrame, output_dir: str) -> None:
    """Generate plots when required columns are available."""
    os.makedirs(output_dir, exist_ok=True)

    if df.empty:
        print("No data to plot.")
        return

    # Use only rows with a total_time for plots driven by time
    df_plot = df.dropna(subset=["total_time"])
    if df_plot.empty:
        print("No complete timer files with 'Total Time' found. Skipping time-based plots.")
    else:
        _plot_image_area_vs_total_time(df_plot, output_dir)
        _plot_average_step_times(df_plot, output_dir)

    # Plots that need nodes/json
    _plot_nodes_before_after(df, output_dir)
    _plot_json_size_vs_nodes(df, output_dir)


# ------------------------- public API -------------------------

def analyze_timer_files(
    time_dir_txt: str,
    time_dir_plots: str,
    *,
    prefix: Optional[str] = None,
    min_step_seconds: float = 1.0,
    debug: bool = False,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Parse all *_timer_info.txt files under `time_dir_txt`, optionally filtering by filename prefix.
    Generates plots into `time_dir_plots`, returns a DataFrame with all parsed metrics.
    - Robust to partial/incomplete files.
    - Will not raise on parse failures; files that fully fail are skipped.
    """
    if not os.path.isdir(time_dir_txt):
        raise FileNotFoundError(f"time_dir_txt not found: {time_dir_txt}")

    os.makedirs(time_dir_plots, exist_ok=True)

    # Only timer files; optional prefix filter
    files = [
        os.path.join(time_dir_txt, f)
        for f in os.listdir(time_dir_txt)
        if f.endswith(TIMER_FILE_SUFFIX) and (prefix is None or f.startswith(prefix))
    ]
    files.sort()

    if not files:
        print("No timer files found to analyze.")
        return pd.DataFrame()

    parsed: List[Dict[str, float]] = []
    skipped: List[str] = []

    for fp in files:
        try:
            metrics = parse_timer_file(fp, min_step_seconds=min_step_seconds, debug=debug)
            parsed.append(metrics)
        except Exception as e:
            skipped.append(f"{os.path.basename(fp)} :: {e}")

    if skipped:
        # Write a small report so the user can inspect problematic files later
        report_path = os.path.join(time_dir_plots, "skipped_files.txt")
        with open(report_path, "w", encoding="utf-8") as r:
            r.write("Files skipped due to parse errors:\n")
            for line in skipped:
                r.write(f"- {line}\n")
        print(f"Some files were skipped. See: {report_path}")

    if not parsed:
        print("No files could be parsed.")
        return pd.DataFrame()

    df = pd.DataFrame(parsed)

    # Derivatives/quality-of-life columns
    if "width" in df.columns and "height" in df.columns:
        df["image_area"] = df["width"] * df["height"]

    # Save CSV summary for downstream analysis
    if save_csv:
        csv_path = os.path.join(time_dir_plots, "time_eval_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved summary CSV: {csv_path}")

    # Plots
    generate_plots(df, time_dir_plots)
    print("Plots saved in:", time_dir_plots)
    return df


# ------------------------- CLI -------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Analyze timer logs and generate plots.")
    p.add_argument("--time_dir_txt", required=True, help="Directory containing *_timer_info.txt files.")
    p.add_argument("--time_dir_plots", required=True, help="Output directory for plots and CSV.")
    p.add_argument("--prefix", default=None, help="Only analyze files starting with this prefix (e.g., image basename).")
    p.add_argument("--min_step_seconds", type=float, default=1.0, help="Only record step times > this many seconds.")
    p.add_argument("--debug", action="store_true", help="Print raw file contents while parsing.")
    p.add_argument("--no_csv", action="store_true", help="Do not save the summary CSV.")
    args = p.parse_args()

    analyze_timer_files(
        args.time_dir_txt,
        args.time_dir_plots,
        prefix=args.prefix,
        min_step_seconds=args.min_step_seconds,
        debug=args.debug,
        save_csv=not args.no_csv,
    )
