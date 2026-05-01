import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime

def plot_metrics_from_files(file_paths, output_dir):
    """
    read the specified .parquet files, extract the core metrics, and plot the trends of distance to goal and flock dispersion over time for comparison.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use('seaborn-v0_8-whitegrid')
    
    # create two subplots: one for distance to goal, one for flock dispersion
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    # loop through each file, read the metrics, and plot them
    for file_path in file_paths:
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        # extract a label from the file name for the legend
        label = os.path.splitext(os.path.basename(file_path))[0].replace("data_", "")

        # because the data may have multiple rows per frame (one for each agent), we need to drop duplicates to get one record per frame for the metrics
        # drop_duplicates() ensures we only have one record per frame for the metrics, and sort by frame to ensure the trend lines are correct
        frame_metrics = df[['Frame', 'dist_to_goal', 'dispersion']].drop_duplicates().sort_values('Frame').reset_index(drop=True)

        # plot chart 1: distance to goal
        ax1.plot(frame_metrics['Frame'], frame_metrics['dist_to_goal'], label=label, alpha=0.9)
        
        # plot chart 2: flock dispersion
        ax2.plot(frame_metrics['Frame'], frame_metrics['dispersion'], label=label, alpha=0.9)

    # --- save chart 1 ---
    ax1.set_title('Trend of Distance to Goal over Time', fontsize=16)
    ax1.set_xlabel('Simulation Frame', fontsize=12)
    ax1.set_ylabel('Distance', fontsize=12)
    ax1.legend(title="Simulation Run")
    ax1.grid(True)

    # --- save chart 2 ---
    ax2.set_title('Trend of Flock Dispersion over Time', fontsize=16)
    ax2.set_xlabel('Simulation Frame', fontsize=12)
    ax2.set_ylabel('Max Distance to COM', fontsize=12)
    ax2.legend(title="Simulation Run")
    ax2.grid(True)

    # save the figures with a timestamp to avoid overwriting, and print the paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig1_path = os.path.join(output_dir, f'analysis_dist_to_goal_{timestamp}.png')
    fig2_path = os.path.join(output_dir, f'analysis_dispersion_{timestamp}.png')
    
    fig1.savefig(fig1_path)
    plt.close(fig1)
    
    fig2.savefig(fig2_path)
    plt.close(fig2)
    
    print(f"✅ Analysis charts saved:\n- {fig1_path}\n- {fig2_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze and compare shepherd simulation data from .parquet files.')
    parser.add_argument('files', nargs='+', help='One or more .parquet file paths to compare.')
    parser.add_argument('--output', default=config.ANALYSIS_LOG_DIR, help='Directory to save the output charts.')
    
    # In the main function, import config so the script can run independently
    try:
        import config
        output_dir = config.ANALYSIS_LOG_DIR
    except (ImportError, AttributeError):
        output_dir = 'logs/analysis' # Fallback if config is not available

    args = parser.parse_args()
    
    # if the user specifies an output directory via command line, override the default from config
    if args.output != output_dir:
        output_dir = args.output
        
    plot_metrics_from_files(args.files, output_dir)
