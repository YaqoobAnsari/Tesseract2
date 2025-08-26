import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def parse_timer_file(file_path):
    """Parse a single timer file to extract relevant metrics."""
    with open(file_path, "r") as file:
        content = file.read()

    # Debugging: Print raw content to verify issues
    print("Raw content of the file:")
    print(repr(content))

    # Initialize metrics dictionary
    metrics = {}
    metrics['file_name'] = os.path.basename(file_path)

    try:
        # Extract metrics using regex
        #metrics['file_size_kb'] = float(re.search(r"File Size:\s*([\d.]+)\sKB", content).group(1))
        dimensions = re.search(r"Image Dimensions:\s*(\d+)\s*[xX]\s*(\d+)", content)
        metrics['width'] = int(dimensions.group(1))
        metrics['height'] = int(dimensions.group(2))

        # Total Time extraction with fallback
        total_time_match = re.search(r"Total Time:\s*([\d.]+)\sseconds", content)
        if total_time_match:
            metrics['total_time'] = float(total_time_match.group(1))
        else:
            # Debugging: If regex fails, print and attempt a manual approach
            print("Could not extract 'Total Time' using regex.")
            for line in content.splitlines():
                if "Total Time:" in line:
                    metrics['total_time'] = float(line.split(":")[1].split()[0])
                    print(f"Extracted 'Total Time' manually: {metrics['total_time']}")
                    break
            else:
                raise ValueError("Total Time not found in the file.")

        metrics['nodes_before'] = int(re.search(r"Total graph nodes \(before pruning\):\s*(\d+)", content).group(1))
        metrics['nodes_after'] = int(re.search(r"Total graph nodes \(after pruning\):\s*(\d+)", content).group(1))
        metrics['json_size_before_kb'] = float(re.search(r"JSON File Size \(before pruning\):\s*([\d.]+)\sKB", content).group(1))
        metrics['json_size_after_kb'] = float(re.search(r"JSON File Size \(after pruning\):\s*([\d.]+)\sKB", content).group(1))

        # Extract step times (excluding trivial steps)
        steps = re.findall(r"(.+ check):\s*([\d.]+)\sseconds", content)
        significant_steps = {step: float(time) for step, time in steps if float(time) > 1.0}
        metrics.update(significant_steps)

    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        raise

    return metrics


def generate_plots(data, output_dir):
    """Generate and save plots for the extracted data."""
    os.makedirs(output_dir, exist_ok=True)

    # Image Area vs Total Time
    data['image_area'] = data['width'] * data['height']
    plt.figure(figsize=(10, 6))
    plt.scatter(data['image_area'], data['total_time'], color='blue', alpha=0.7)
    plt.xlabel('Image Area (pixels)', fontsize=14)
    plt.ylabel('Total Time (seconds)', fontsize=14)
    plt.title('Image Area vs Total Time', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, 'image_area_vs_total_time.png'), dpi=300)
    plt.close()

    # Nodes Before and After Pruning (Color-coded by file, with shape legend)
    plt.figure(figsize=(12, 8))
    unique_colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    for idx, row in data.iterrows():
        color = unique_colors[idx % len(unique_colors)]
        plt.scatter([idx], row['nodes_before'], color=color, s=100, marker='o')
        plt.scatter([idx], row['nodes_after'], color=color, s=100, marker='D')
    plt.xlabel('File Index', fontsize=14)
    plt.ylabel('Number of Nodes', fontsize=14)
    plt.title('Nodes Before and After Pruning', fontsize=16)
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='Before Pruning', markerfacecolor='black', markersize=10),
            plt.Line2D([0], [0], marker='D', color='w', label='After Pruning', markerfacecolor='black', markersize=10),
        ],
        fontsize=12,
        loc='upper right',
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, 'nodes_before_after_pruning.png'), dpi=300)
    plt.close()

    # JSON File Size vs Number of Nodes (Color-coded by file, with shape legend)
    plt.figure(figsize=(12, 8))
    for idx, row in data.iterrows():
        color = unique_colors[idx % len(unique_colors)]
        plt.scatter(row['nodes_before'], row['json_size_before_kb'], color=color, s=100, marker='o')
        plt.scatter(row['nodes_after'], row['json_size_after_kb'], color=color, s=100, marker='D')
    plt.xlabel('Number of Nodes', fontsize=14)
    plt.ylabel('JSON File Size (KB)', fontsize=14)
    plt.title('JSON File Size vs Number of Nodes', fontsize=16)
    plt.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', label='Before Pruning', markerfacecolor='black', markersize=10),
            plt.Line2D([0], [0], marker='D', color='w', label='After Pruning', markerfacecolor='black', markersize=10),
        ],
        fontsize=12,
        loc='upper right',
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, 'json_size_vs_nodes.png'), dpi=300)
    plt.close()

    # Average Time Taken for Significant Steps
    step_columns = [col for col in data.columns if 'check' in col]
    step_averages = data[step_columns].mean().sort_values()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(step_averages.index, step_averages.values, color='teal', alpha=0.8)
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Average Time (seconds)', fontsize=14)
    plt.title('Average Time for Significant Steps', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f'{height:.2f}', ha='center', fontsize=10)

    plt.savefig(os.path.join(output_dir, 'average_step_times.png'), dpi=300)
    plt.close()


def analyze_timer_files(time_dir_txt, time_dir_plots):
    """Mother function to parse files, analyze metrics, and generate plots."""
    timer_files = [os.path.join(time_dir_txt, file) for file in os.listdir(time_dir_txt) if file.endswith('.txt')]
    data = [parse_timer_file(file) for file in timer_files]
    df = pd.DataFrame(data)
    generate_plots(df, time_dir_plots)
    print("Plots saved successfully in:", time_dir_plots)
    return df

 
#df_results = analyze_timer_files(time_dir_txt, time_dir_plots)
