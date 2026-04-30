import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import os
import config

COLOR_MAP = {'Normal': 'white', 'A': 'red', 'B': 'orange'}
TRACE_COLOR_MAP = {'Normal': 'lightgray', 'A': 'red', 'B': 'orange'}
LEGEND_LABELS = {'Normal': 'Normal (Flock)', 'A': 'Type A (Unresponsive)', 'B': 'Type B (Dispersing)'}

class Visualizer:
    def __init__(self, env):
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_xlim(0, config.SPACE_SIZE)
        self.ax.set_ylim(0, config.SPACE_SIZE)
        self.ax.set_title("Swarm Shepherding Simulation")

        # draw the goal area (sheepfold)
        goal_circle = patches.Circle((env.goal_pos[0], env.goal_pos[1]), radius=config.GOAL_RADIUS, color='lightgreen', alpha=0.3)
        self.ax.add_patch(goal_circle)
        # draw the goal center (golden star)
        self.ax.plot(env.goal_pos[0], env.goal_pos[1], 'y*', markersize=18, label='Goal')
        
        # initialize scatter objects for dynamic update
        self.scat_sheep = self.ax.scatter([], [], edgecolors='gray', s=40)
        self.scat_dog = self.ax.scatter([], [], marker='^', c='#333333', s=80)

        # custom legend elements for different agent types
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=LEGEND_LABELS['Normal'], markerfacecolor=COLOR_MAP['Normal'], markeredgecolor='gray', markersize=8),
            Line2D([0], [0], marker='o', color='w', label=LEGEND_LABELS['A'], markerfacecolor=COLOR_MAP['A'], markeredgecolor='gray', markersize=8),
            Line2D([0], [0], marker='o', color='w', label=LEGEND_LABELS['B'], markerfacecolor=COLOR_MAP['B'], markeredgecolor='gray', markersize=8),
            Line2D([0], [0], marker='^', color='w', label='Dog', markerfacecolor='#333333', markersize=10)
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        plt.ion() # enable interactive mode, allow dynamic refresh

    def render(self, env):
        # update the data of the scatter objects instead of redrawing
        self.scat_sheep.set_offsets(env.sheep_pos)
        self.scat_sheep.set_facecolors(np.vectorize(COLOR_MAP.get)(env.sheep_status))
        self.scat_dog.set_offsets([env.dog_pos])

        plt.pause(0.01) # pause for a very short time to refresh the screen

    def close(self):
        plt.ioff()
        plt.show()

    def save_trajectory(self, env, timestamp):
        if not os.path.exists(config.VISUAL_LOG_DIR):
            os.makedirs(config.VISUAL_LOG_DIR)
            
        df = pd.DataFrame(env.history_data, columns=['Frame', 'Agent_Type', 'Agent_ID', 'X', 'Y', 'Status'])
        
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(0, config.SPACE_SIZE)
        ax.set_ylim(0, config.SPACE_SIZE)
        ax.set_title("Shepherding Trajectory")

        # draw the goal area
        goal_circle = patches.Circle((env.goal_pos[0], env.goal_pos[1]), radius=config.GOAL_RADIUS, color='lightgreen', alpha=0.3)
        ax.add_patch(goal_circle)
        ax.plot(env.goal_pos[0], env.goal_pos[1], 'y*', markersize=18, label='Goal')

        # plot sheep traces
        sheep_df = df[df['Agent_Type'] == 'Sheep']
        for aid in sheep_df['Agent_ID'].unique():
            trace = sheep_df[sheep_df['Agent_ID'] == aid]
            status = trace['Status'].iloc[0]
            color = TRACE_COLOR_MAP.get(status, 'lightgray')
            
            # add label only for the first agent of each specific type to prevent duplicate legend entries
            label = ""
            if aid == sheep_df[sheep_df['Status'] == status]['Agent_ID'].min():
                label = f"{LEGEND_LABELS[status]} Trace"
                
            lw = 1.0 if status != 'Normal' else 0.5
            ax.plot(trace['X'], trace['Y'], color=color, linewidth=lw, label=label)

        # plot dog trace
        dog_df = df[df['Agent_Type'] == 'Dog']
        ax.plot(dog_df['X'], dog_df['Y'], color='#333333', linewidth=1.5, label='Dog Trace')
        
        # mark start points
        ax.plot(env.dog_start_pos[0], env.dog_start_pos[1], 'ro', markersize=6, label='Dog Start')
        ax.plot(env.sheep_start_center[0], env.sheep_start_center[1], 'ko', markersize=6, label='Sheep Center')

        ax.legend(loc='upper right')
        
        export_path = os.path.join(config.VISUAL_LOG_DIR, f"trajectory_{timestamp}.png")
        fig.savefig(export_path)
        plt.close(fig)

    def save_animation_mp4(self, env, timestamp):
        if not os.path.exists(config.VIDEO_LOG_DIR):
            os.makedirs(config.VIDEO_LOG_DIR)
            
        # Capacity control: clean up old videos before saving a new one
        try:
            mp4_files = [os.path.join(config.VIDEO_LOG_DIR, f) for f in os.listdir(config.VIDEO_LOG_DIR) if f.endswith('.mp4')]
            if len(mp4_files) >= config.MAX_LOG_FILES:
                mp4_files.sort(key=os.path.getmtime)
                num_to_delete = len(mp4_files) - config.MAX_LOG_FILES + 1
                for f in mp4_files[:num_to_delete]:
                    os.remove(f)
        except OSError as e:
            print(f"Warning: Could not perform video log rotation: {e}")

        df = pd.DataFrame(env.history_data, columns=['Frame', 'Agent_Type', 'Agent_ID', 'X', 'Y', 'Status'])
        
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(0, config.SPACE_SIZE)
        ax.set_ylim(0, config.SPACE_SIZE)
        ax.set_title("Shepherding Animation (MP4)")

        goal_circle = patches.Circle((env.goal_pos[0], env.goal_pos[1]), radius=config.GOAL_RADIUS, color='lightgreen', alpha=0.3)
        ax.add_patch(goal_circle)
        ax.plot(env.goal_pos[0], env.goal_pos[1], 'y*', markersize=18, label='Goal')

        scat_sheep = ax.scatter([], [], edgecolors='gray', s=40)
        scat_dog = ax.scatter([], [], marker='^', c='#333333', s=80)
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=LEGEND_LABELS['Normal'], markerfacecolor=COLOR_MAP['Normal'], markeredgecolor='gray', markersize=8),
            Line2D([0], [0], marker='o', color='w', label=LEGEND_LABELS['A'], markerfacecolor=COLOR_MAP['A'], markeredgecolor='gray', markersize=8),
            Line2D([0], [0], marker='o', color='w', label=LEGEND_LABELS['B'], markerfacecolor=COLOR_MAP['B'], markeredgecolor='gray', markersize=8),
            Line2D([0], [0], marker='^', color='w', label='Dog', markerfacecolor='#333333', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Initialize frame text watermark at the top-left corner
        frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top', fontsize=10, fontweight='bold', color='black')

        frames = sorted(df['Frame'].unique())

        def update(frame):
            frame_data = df[df['Frame'] == frame]
            sheep_data = frame_data[frame_data['Agent_Type'] == 'Sheep']
            dog_data = frame_data[frame_data['Agent_Type'] == 'Dog']
            scat_sheep.set_offsets(sheep_data[['X', 'Y']].values)
            statuses = sheep_data['Status'].values
            scat_sheep.set_facecolors(np.vectorize(COLOR_MAP.get)(statuses))
            scat_dog.set_offsets(dog_data[['X', 'Y']].values)

            # Recalculate metrics for watermark
            sheep_pos = sheep_data[['X', 'Y']].values
            if sheep_pos.shape[0] > 0:
                com = np.mean(sheep_pos, axis=0)
                dist_to_goal = np.linalg.norm(env.goal_pos - com)
                flock_dispersion = np.max(np.linalg.norm(sheep_pos - com, axis=1))
                watermark_text = (
                    f"Frame: {frame}\n"
                    f"Dist to Goal: {dist_to_goal:.2f}\n"
                    f"Flock Dispersion: {flock_dispersion:.2f}"
                )
            else:
                watermark_text = f"Frame: {frame}"
            frame_text.set_text(watermark_text)

            return scat_sheep, scat_dog, frame_text

        ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
        export_path = os.path.join(config.VIDEO_LOG_DIR, f"animation_{timestamp}.mp4")
        
        try:
            ani.save(export_path, writer='ffmpeg', fps=30)
        except Exception as e:
            print(f"\n[Error] Could not export MP4 video: {e}")
            print("[Hint] Please ensure 'ffmpeg' is installed and added to your system PATH.")
        finally:
            plt.close(fig)