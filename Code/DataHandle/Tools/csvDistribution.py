import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
import os
from pathlib import Path

# Set font and graphic parameters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Solve PyCharm graphic display issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display errors


def calculate_thresholds(dialogue_lengths):
    """Dynamically calculate tertile thresholds"""
    if not dialogue_lengths:
        return None, None

    # Use numpy's percentile function to calculate 33.3% and 66.7% percentiles
    q1 = np.percentile(dialogue_lengths, 33.3)  # First tertile
    q2 = np.percentile(dialogue_lengths, 66.7)  # Second tertile

    print(f"Dynamically calculated tertile thresholds: Q1(33.3%) = {q1:.2f}, Q2(66.7%) = {q2:.2f}")
    return q1, q2


def calculate_dialogue_length(csv_path, is_pos=True):
    """Calculate total word count for each dialogue group"""
    try:
        df = pd.read_csv(csv_path)
        group_field = 'case' if is_pos else 'case_id'
        df['content'] = df['content'].astype(str)

        dialogue_stats = []
        for case_name, group in df.groupby(group_field):
            total_chars = group['content'].str.len().sum()
            dialogue_stats.append({
                'case': case_name,
                'total_chars': total_chars,
                'type': 'Pos' if is_pos else 'Neg',
                'file': os.path.basename(csv_path)
            })
        return dialogue_stats
    except Exception as e:
        print(f"Error processing file {csv_path}: {e}")
        return []


def calculate_median_by_segment(lengths, thresholds=None):
    """Calculate medians for three segments using dynamic thresholds"""
    if thresholds is None:
        # If no thresholds provided, calculate dynamically
        thresholds = calculate_thresholds(lengths)
        if thresholds[0] is None:
            return {}, {}, ([], [], [])

    q1, q2 = thresholds

    short_segment = [x for x in lengths if x <= q1]
    medium_segment = [x for x in lengths if q1 < x <= q2]
    long_segment = [x for x in lengths if x > q2]

    medians = {
        'short_median': np.median(short_segment) if short_segment else 0,
        'medium_median': np.median(medium_segment) if medium_segment else 0,
        'long_median': np.median(long_segment) if long_segment else 0,
        'q1_threshold': q1,
        'q2_threshold': q2
    }

    counts = {
        'short_count': len(short_segment),
        'medium_count': len(medium_segment),
        'long_count': len(long_segment),
        'total_count': len(lengths)
    }

    return medians, counts, (short_segment, medium_segment, long_segment)


def plot_histogram_with_medians(dialogue_lengths, output_dir, thresholds=None):
    """Plot histogram with median markers, using dynamic thresholds"""
    if not dialogue_lengths:
        print("No data available")
        return

    # Convert to numpy array
    lengths = np.array(dialogue_lengths)

    # Calculate dynamic thresholds
    if thresholds is None:
        thresholds = calculate_thresholds(lengths)

    q1, q2 = thresholds

    # Calculate medians for three segments
    medians, counts, segments = calculate_median_by_segment(lengths, thresholds)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set colors
    colors = ['#FF4444', '#44CC44', '#4444FF']  # Red, Green, Blue

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot histograms for three segments
    short_segment, medium_segment, long_segment = segments

    n1, bins1, patches1 = plt.hist(short_segment, bins=15, alpha=0.7, color=colors[0],
                                   edgecolor='black', label='Short Dialogues')
    n2, bins2, patches2 = plt.hist(medium_segment, bins=15, alpha=0.7, color=colors[1],
                                   edgecolor='black', label='Medium Dialogues')
    n3, bins3, patches3 = plt.hist(long_segment, bins=15, alpha=0.7, color=colors[2],
                                   edgecolor='black', label='Long Dialogues')

    # Add median markers on x-axis (points instead of lines)
    plt.scatter(medians['short_median'], 0, color=colors[0], s=100, marker='o',
                zorder=5, label=f'Short Median: {medians["short_median"]:.0f}')
    plt.scatter(medians['medium_median'], 0, color=colors[1], s=100, marker='o',
                zorder=5, label=f'Medium Median: {medians["medium_median"]:.0f}')
    plt.scatter(medians['long_median'], 0, color=colors[2], s=100, marker='o',
                zorder=5, label=f'Long Median: {medians["long_median"]:.0f}')

    plt.xlabel('Dialogue Word Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Dialogue Length Distribution with Dynamic Tertile Segmentation', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'dialogue_length_histogram_dynamic.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Dynamic threshold histogram saved to: {plot_path}")
    plt.close()

    return medians, counts, thresholds


def plot_density_with_fill(dialogue_lengths, output_dir, thresholds=None):
    """Plot kernel density estimation with median markers, using dynamic thresholds"""
    if not dialogue_lengths:
        print("No data available")
        return

    # Convert to numpy array
    lengths = np.array(dialogue_lengths)

    # Calculate dynamic thresholds
    if thresholds is None:
        thresholds = calculate_thresholds(lengths)

    q1, q2 = thresholds

    # Calculate medians for three segments
    medians, counts, segments = calculate_median_by_segment(lengths, thresholds)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set colors
    colors = ['#FF4444', '#44CC44', '#4444FF']  # Red, Green, Blue
    fill_colors = ['#FF4444', '#44CC44', '#4444FF']  # Light red, light green, light blue

    # Create figure
    plt.figure(figsize=(12, 6))

    # Kernel density estimation
    kde = gaussian_kde(lengths)
    x_range = np.linspace(min(lengths), max(lengths), 1000)
    kde_values = kde(x_range)

    # Plot kernel density curve
    plt.plot(x_range, kde_values, 'k-', linewidth=2, label='Kernel Density Estimation')

    # Define x-axis ranges for three segments
    x_short = x_range[x_range <= q1]
    y_short = kde_values[x_range <= q1]

    x_medium = x_range[(x_range > q1) & (x_range <= q2)]
    y_medium = kde_values[(x_range > q1) & (x_range <= q2)]

    x_long = x_range[x_range > q2]
    y_long = kde_values[x_range > q2]

    # Fill areas for three segments
    plt.fill_between(x_short, y_short, color=fill_colors[0], alpha=0.5, label='Short Dialogues')
    plt.fill_between(x_medium, y_medium, color=fill_colors[1], alpha=0.5, label='Medium Dialogues')
    plt.fill_between(x_long, y_long, color=fill_colors[2], alpha=0.5, label='Long Dialogues')

    # Add median markers on x-axis (points instead of lines)
    plt.scatter(medians['short_median'], 0, color=colors[0], s=100, marker='o',
                zorder=5, label=f'Short Median: {medians["short_median"]:.0f}')
    plt.scatter(medians['medium_median'], 0, color=colors[1], s=100, marker='o',
                zorder=5, label=f'Medium Median: {medians["medium_median"]:.0f}')
    plt.scatter(medians['long_median'], 0, color=colors[2], s=100, marker='o',
                zorder=5, label=f'Long Median: {medians["long_median"]:.0f}')

    plt.xlabel('Dialogue Word Count', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Dialogue Length Distribution with Dynamic Tertile Segmentation', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'dialogue_length_density_dynamic.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Dynamic threshold density plot saved to: {plot_path}")
    plt.close()

    return medians, counts, thresholds


def plot_combined_histogram_density(dialogue_lengths, output_dir, thresholds=None):
    """Plot combined histogram and density plot with median markers, using dynamic thresholds"""
    if not dialogue_lengths:
        print("No data available")
        return

    # Convert to numpy array
    lengths = np.array(dialogue_lengths)

    # Calculate dynamic thresholds
    if thresholds is None:
        thresholds = calculate_thresholds(lengths)

    q1, q2 = thresholds

    # Calculate medians for three segments
    medians, counts, segments = calculate_median_by_segment(lengths, thresholds)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set colors
    colors = ['#FF4444', '#44CC44', '#4444FF']  # Red, Green, Blue
    fill_colors = ['#FF4444', '#44CC44', '#4444FF']  # Light red, light green, light blue

    # Create figure and dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot histograms on left y-axis
    short_segment, medium_segment, long_segment = segments

    n1, bins1, patches1 = ax1.hist(short_segment, bins=15, alpha=0.7, color=colors[0],
                                   edgecolor='black', label='Short Dialogues')
    n2, bins2, patches2 = ax1.hist(medium_segment, bins=15, alpha=0.7, color=colors[1],
                                   edgecolor='black', label='Medium Dialogues')
    n3, bins3, patches3 = ax1.hist(long_segment, bins=15, alpha=0.7, color=colors[2],
                                   edgecolor='black', label='Long Dialogues')

    # Add median markers on x-axis (points instead of lines)
    ax1.scatter(medians['short_median'], 0, color=colors[0], s=100, marker='o',
                zorder=5, label=f'Short Median: {medians["short_median"]:.0f}')
    ax1.scatter(medians['medium_median'], 0, color=colors[1], s=100, marker='o',
                zorder=5, label=f'Medium Median: {medians["medium_median"]:.0f}')
    ax1.scatter(medians['long_median'], 0, color=colors[2], s=100, marker='o',
                zorder=5, label=f'Long Median: {medians["long_median"]:.0f}')

    ax1.set_xlabel('Dialogue Word Count', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create right y-axis for density plot
    ax2 = ax1.twinx()

    # Kernel density estimation
    kde = gaussian_kde(lengths)
    x_range = np.linspace(min(lengths), max(lengths), 1000)
    kde_values = kde(x_range)

    ax2.plot(x_range, kde_values, 'k-', linewidth=2, label='Kernel Density Estimation')

    ax2.set_ylabel('Probability Density', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    plt.title('Dialogue Length Distribution: Combined Histogram and Density Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'dialogue_length_combined_dynamic.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Dynamic threshold combined plot saved to: {plot_path}")
    plt.close()

    return medians, counts, thresholds


def main():
    base_path = r"D:\Study\Paper\EI\Data\4Gambling"

    # Define file paths
    file_paths = {
        'Pos': {
            'DeepSeek': f"{base_path}/Pos/csv/DeepSeek/网络赌博诈骗.csv",
            'DouBao': f"{base_path}/Pos/csv/DouBao/网络赌博诈骗.csv",
            'Kimi': f"{base_path}/Pos/csv/Kimi/网络赌博诈骗.csv"
        },
        'Neg': {
            'DeepSeek': f"{base_path}/Neg/csv/DeepSeek/Lottery_dialogues.csv",
            'DouBao': f"{base_path}/Neg/csv/DouBao/Lottery_dialogues.csv",
            'Kimi': f"{base_path}/Neg/csv/Kimi/Lottery_dialogues.csv"
        }
    }

    # Output directory
    output_dir = f"{base_path}/analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    # Collect all dialogue lengths
    all_dialogue_lengths = []
    dialogue_details = []

    # Process Pos data
    for model, path in file_paths['Pos'].items():
        if os.path.exists(path):
            print(f"Processing Pos data - {model}: {path}")
            stats = calculate_dialogue_length(path, is_pos=True)
            for stat in stats:
                all_dialogue_lengths.append(stat['total_chars'])
                dialogue_details.append(stat)
        else:
            print(f"File does not exist: {path}")

    # Process Neg data
    for model, path in file_paths['Neg'].items():
        if os.path.exists(path):
            print(f"Processing Neg data - {model}: {path}")
            stats = calculate_dialogue_length(path, is_pos=False)
            for stat in stats:
                all_dialogue_lengths.append(stat['total_chars'])
                dialogue_details.append(stat)
        else:
            print(f"File does not exist: {path}")

    if not all_dialogue_lengths:
        print("No valid data found")
        return

    # Save detailed statistical results
    details_df = pd.DataFrame(dialogue_details)
    stats_path = os.path.join(output_dir, 'dialogue_statistics_dynamic.csv')
    details_df.to_csv(stats_path, index=False, encoding='utf-8-sig')

    # Calculate dynamic tertile thresholds
    thresholds = calculate_thresholds(all_dialogue_lengths)

    if thresholds[0] is None:
        print("Unable to calculate thresholds, data may be empty")
        return

    q1, q2 = thresholds

    # Calculate medians for three segments
    medians, counts, segments = calculate_median_by_segment(all_dialogue_lengths, thresholds)

    print(f"\n=== Statistical Summary ===")
    print(f"Total dialogues: {len(all_dialogue_lengths)}")
    print(f"Word count range: {min(all_dialogue_lengths)} - {max(all_dialogue_lengths)}")
    print(f"Overall average word count: {np.mean(all_dialogue_lengths):.2f}")
    print(f"Overall median: {np.median(all_dialogue_lengths):.2f}")

    print(f"\n=== Dynamic Tertile Segmentation Results ===")
    print(f"First tertile (Q1, 33.3%): {q1:.2f} words")
    print(f"Second tertile (Q2, 66.7%): {q2:.2f} words")
    print(f"Short dialogues (≤{q1:.0f} words, {counts['short_count']} dialogues, {counts['short_count'] / counts['total_count'] * 100:.1f}%): Median = {medians['short_median']:.2f} words")
    print(f"Medium dialogues ({q1:.0f}-{q2:.0f} words, {counts['medium_count']} dialogues, {counts['medium_count'] / counts['total_count'] * 100:.1f}%): Median = {medians['medium_median']:.2f} words")
    print(f"Long dialogues (>{q2:.0f} words, {counts['long_count']} dialogues, {counts['long_count'] / counts['total_count'] * 100:.1f}%): Median = {medians['long_median']:.2f} words")

    # Validate segmentation proportions
    expected_short = counts['total_count'] * 0.333
    expected_medium = counts['total_count'] * 0.333
    expected_long = counts['total_count'] * 0.334

    print(f"\n=== Segmentation Proportion Validation ===")
    print(f"Expected short dialogues: {expected_short:.1f}, Actual: {counts['short_count']}, Deviation: {abs(counts['short_count'] - expected_short):.1f}")
    print(f"Expected medium dialogues: {expected_medium:.1f}, Actual: {counts['medium_count']}, Deviation: {abs(counts['medium_count'] - expected_medium):.1f}")
    print(f"Expected long dialogues: {expected_long:.1f}, Actual: {counts['long_count']}, Deviation: {abs(counts['long_count'] - expected_long):.1f}")

    # Generate three distribution plots (using dynamic thresholds)
    print(f"\nGenerating dynamic threshold histogram...")
    hist_medians, hist_counts, hist_thresholds = plot_histogram_with_medians(all_dialogue_lengths, output_dir, thresholds)

    print(f"Generating dynamic threshold density plot...")
    density_medians, density_counts, density_thresholds = plot_density_with_fill(all_dialogue_lengths, output_dir, thresholds)

    print(f"Generating dynamic threshold combined plot...")
    combined_medians, combined_counts, combined_thresholds = plot_combined_histogram_density(all_dialogue_lengths, output_dir, thresholds)

    print(f"\nOutput files:")
    print(f"Detailed statistics: {stats_path}")
    print(f"Dynamic threshold histogram: {output_dir}/dialogue_length_histogram_dynamic.png")
    print(f"Dynamic threshold density plot: {output_dir}/dialogue_length_density_dynamic.png")
    print(f"Dynamic threshold combined plot: {output_dir}/dialogue_length_combined_dynamic.png")

    # Save threshold information
    threshold_info = {
        'q1_33_percent': q1,
        'q2_67_percent': q2,
        'total_dialogues': len(all_dialogue_lengths),
        'calculation_method': 'Dynamic tertile method based on 33.3% and 66.7% percentiles'
    }

    threshold_df = pd.DataFrame([threshold_info])
    threshold_path = os.path.join(output_dir, 'dynamic_thresholds.csv')
    threshold_df.to_csv(threshold_path, index=False, encoding='utf-8-sig')
    print(f"Threshold information: {threshold_path}")


if __name__ == "__main__":
    main()