import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import pandas as pd
import os
import config

class Visualizer:
    def __init__(self, env):
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_xlim(0, config.SPACE_SIZE)
        self.ax.set_ylim(0, config.SPACE_SIZE)
        self.ax.set_title("Swarm Shepherding Simulation")

        # draw the goal area (sheepfold)
        goal_circle = patches.Circle((env.goal_pos[0], env.goal_pos[1]), radius=20, color='lightgreen', alpha=0.3)
        self.ax.add_patch(goal_circle)
        # draw the goal center (golden star)
        self.ax.plot(env.goal_pos[0], env.goal_pos[1], 'y*', markersize=18, label='Goal')
        
        # initialize scatter objects for dynamic update
        self.scat_sheep = self.ax.scatter([], [], c='white', edgecolors='gray', s=40, label='Sheep')
        self.scat_dog = self.ax.scatter([], [], marker='^', c='#333333', s=80, label='Dog')

        self.ax.legend(loc='upper right')
        plt.ion() # enable interactive mode, allow dynamic refresh

    def render(self, env):
        # update the data of the scatter objects instead of redrawing
        self.scat_sheep.set_offsets(env.sheep_pos)
        self.scat_dog.set_offsets([env.dog_pos])

        plt.pause(0.01) # pause for a very short time to refresh the screen

    def close(self):
        plt.ioff()
        plt.show()

    def save_trajectory(self, env, timestamp):
        if not os.path.exists(config.VISUAL_LOG_DIR):
            os.makedirs(config.VISUAL_LOG_DIR)
            
        df = pd.DataFrame(env.history_data, columns=['Frame', 'Agent_Type', 'Agent_ID', 'X', 'Y'])
        
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(0, config.SPACE_SIZE)
        ax.set_ylim(0, config.SPACE_SIZE)
        ax.set_title("Shepherding Trajectory")

        # draw the goal area
        goal_circle = patches.Circle((env.goal_pos[0], env.goal_pos[1]), radius=20, color='lightgreen', alpha=0.3)
        ax.add_patch(goal_circle)
        ax.plot(env.goal_pos[0], env.goal_pos[1], 'y*', markersize=18, label='Goal')

        # plot sheep traces
        sheep_df = df[df['Agent_Type'] == 'Sheep']
        for aid in sheep_df['Agent_ID'].unique():
            trace = sheep_df[sheep_df['Agent_ID'] == aid]
            label = 'Sheep Trace' if aid == 0 else ""
            ax.plot(trace['X'], trace['Y'], color='lightgray', linewidth=0.5, label=label)

        # plot dog trace
        dog_df = df[df['Agent_Type'] == 'Dog']
        ax.plot(dog_df['X'], dog_df['Y'], color='red', linewidth=1.5, label='Dog Trace')
        
        # mark start points
        ax.plot(env.dog_start_pos[0], env.dog_start_pos[1], 'ro', markersize=6, label='Dog Start')
        ax.plot(env.sheep_start_center[0], env.sheep_start_center[1], 'ko', markersize=6, label='Sheep Center')

        ax.legend(loc='upper right')
        
        export_path = os.path.join(config.VISUAL_LOG_DIR, f"trajectory_{timestamp}.png")
        fig.savefig(export_path)
        plt.close(fig)

    def save_animation(self, env, timestamp):
        if not os.path.exists(config.VISUAL_LOG_DIR):
            os.makedirs(config.VISUAL_LOG_DIR)
            
        df = pd.DataFrame(env.history_data, columns=['Frame', 'Agent_Type', 'Agent_ID', 'X', 'Y'])
        
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(0, config.SPACE_SIZE)
        ax.set_ylim(0, config.SPACE_SIZE)
        ax.set_title("Shepherding Animation")

        goal_circle = patches.Circle((env.goal_pos[0], env.goal_pos[1]), radius=20, color='lightgreen', alpha=0.3)
        ax.add_patch(goal_circle)
        ax.plot(env.goal_pos[0], env.goal_pos[1], 'y*', markersize=18, label='Goal')

        scat_sheep = ax.scatter([], [], c='white', edgecolors='gray', s=40, label='Sheep')
        scat_dog = ax.scatter([], [], marker='^', c='#333333', s=80, label='Dog')
        ax.legend(loc='upper right')

        frames = sorted(df['Frame'].unique())

        def update(frame):
            frame_data = df[df['Frame'] == frame]
            sheep_data = frame_data[frame_data['Agent_Type'] == 'Sheep']
            dog_data = frame_data[frame_data['Agent_Type'] == 'Dog']
            scat_sheep.set_offsets(sheep_data[['X', 'Y']].values)
            scat_dog.set_offsets(dog_data[['X', 'Y']].values)
            return scat_sheep, scat_dog

        ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
        export_path = os.path.join(config.VISUAL_LOG_DIR, f"animation_{timestamp}.gif")
        ani.save(export_path, writer='pillow', fps=30)
        plt.close(fig)