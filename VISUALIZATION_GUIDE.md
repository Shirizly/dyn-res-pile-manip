# MPC Visualization Guide

This guide explains the visualization features added to the MPC code in `visualize_mpc.py`.

## Features

The following visualization and saving options have been added:

1. **Save Images**: Save the state after every action as individual PNG images
2. **Save Video**: Save the entire action sequence as an MP4 video (sped up by a configurable factor)
3. **Plot Rewards**: Save a graph showing the reward (distance to goal) evolution per action step

## Configuration

All visualization options are configured in `config/mpc/config.yaml` under the `mpc.visualization` section:

```yaml
mpc:
  # ... other MPC parameters ...
  
  # Visualization options
  visualization:
    save_images: True         # Save image after each action
    save_video: True          # Save video of all actions
    video_fps: 5              # Frames per second for the video
    video_speedup: 1.0        # Speedup factor for video (1.0 = normal speed, 2.0 = 2x speed)
    plot_rewards: True        # Plot reward graph over action steps
    output_dir: 'outputs'     # Output directory for saved files
```

### Configuration Parameters

- **`save_images`** (bool): When `True`, saves each observation as a separate PNG image after each MPC step
- **`save_video`** (bool): When `True`, creates an MP4 video of the entire MPC trajectory
- **`video_fps`** (int): Base frames per second for the video (default: 5)
- **`video_speedup`** (float): Multiplier for video playback speed
  - `1.0` = normal speed
  - `2.0` = 2x speed (twice as fast)
  - `0.5` = half speed (slow motion)
  - Effective FPS = `video_fps * video_speedup`
- **`plot_rewards`** (bool): When `True`, generates a line plot of reward evolution over MPC steps
- **`output_dir`** (str): Base directory for all output files (default: 'outputs')

## Output Structure

When any visualization option is enabled, outputs are saved to a timestamped directory:

```
outputs/
└── mpc_run_2024_03_18_14_30_45_123/
    ├── goal.png                    # Target goal image
    ├── images/                     # Individual frame images (if save_images=True)
    │   ├── step_0000.png
    │   ├── step_0001.png
    │   ├── ...
    │   └── step_0020.png
    ├── mpc_trajectory.mp4          # Video of trajectory (if save_video=True)
    ├── reward_evolution.png        # Reward plot (if plot_rewards=True)
    ├── rewards.npy                 # Raw reward values (if plot_rewards=True)
    └── actions.npy                 # Actions taken (if plot_rewards=True)
```

## Usage

1. **Edit the configuration file** (`config/mpc/config.yaml`) to enable desired visualization options

2. **Run the MPC visualization**:
   ```bash
   python visualize_mpc.py
   ```

3. **Check the output**: After the MPC run completes, check the `outputs/` directory for the timestamped output folder

## Implementation Details

### Functions Added

Three new functions were added to `visualize_mpc.py`:

1. **`save_observations_as_images(raw_obs, output_dir, prefix='step')`**
   - Saves each observation frame as a PNG image
   - Images are numbered sequentially (e.g., step_0000.png, step_0001.png, ...)
   - Uses OpenCV to save RGB images at equilibrium after each action

2. **`save_observations_as_video(raw_obs, output_path, fps=5, speedup=1.0)`**
   - Creates an MP4 video from the observation sequence
   - Applies speedup factor to effective FPS
   - Note: This saves only the actual simulation states, not the MPC computation time

3. **`plot_reward_evolution(rewards, output_path, title='Reward Evolution Over MPC Steps')`**
   - Creates a line plot of rewards over MPC steps
   - Shows reward progression and final improvement
   - Prints reward statistics (initial, final, mean, improvement)

### Data Sources

The visualization functions use data returned by `env.step_subgoal_ptcl()`:
- **`raw_obs`**: Array of shape `(n_mpc+1, height, width, 5)` containing RGBA and depth at each step
- **`rewards`**: Array of shape `(n_mpc+1,)` containing the reward (distance to goal) at each step
- **`actions`**: Array of shape `(n_mpc, action_dim)` containing the actions taken

### Notes

- The video **does not** include the time spent computing the MPC optimization between actions - it only shows the actual simulation states after each action reaches equilibrium
- Images are saved in BGR format (OpenCV standard)
- The reward represents the distance to the goal, computed by the MPC's reward function
- All visualizations are optional and can be enabled/disabled independently via the config file

## Tips

- For faster playback, increase `video_speedup` (e.g., 2.0 for 2x speed)
- For smoother video, increase `video_fps` (e.g., 10 or 30)
- Individual images are useful for detailed frame-by-frame analysis
- The reward plot helps visualize the MPC's progress toward the goal over time
- Set all visualization options to `False` if you only want to run the MPC without saving outputs

## Example Output

After running with all visualization options enabled:
```
MPC completed: 20 steps
Initial reward: 0.3245
Final reward: 0.7823
Reward improvement: 0.4578

Saving outputs to: outputs/mpc_run_2024_03_18_14_30_45_123
Saved goal image to outputs/mpc_run_2024_03_18_14_30_45_123/goal.png
Saved 21 images to outputs/mpc_run_2024_03_18_14_30_45_123/images
Saved video to outputs/mpc_run_2024_03_18_14_30_45_123/mpc_trajectory.mp4 with 21 frames at 5 fps
Saved reward plot to outputs/mpc_run_2024_03_18_14_30_45_123/reward_evolution.png
Reward stats - Initial: 0.3245, Final: 0.7823, Mean: 0.5621, Improvement: 0.4578

Visualization complete!
```
