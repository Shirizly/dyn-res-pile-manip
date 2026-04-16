#!/usr/bin/env python
"""
Test the visualization save pipeline (images, video, reward plot).
Runs a few env steps (no MPC) to validate rendering + saving works correctly.
Tests both snapshot-based saving and full-manipulation video_recorder.
"""

import os
import cv2
import numpy as np
from env.flex_env import FlexEnv
from utils import load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms
from visualize_mpc import save_observations_as_images, plot_reward_evolution


def main():
    print("=" * 60)
    print("Visualization Save Pipeline Test")
    print("=" * 60)

    config = load_yaml("config/mpc/config.yaml")
    env = FlexEnv(config)
    env.reset()

    timestamp = get_current_YYYY_MM_DD_hh_mm_ss_ms()
    output_dir = os.path.join('test_outputs', f'save_test_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}\n")

    # Collect frames by stepping the environment
    print("Collecting frames...")
    obs = env.render()
    print(f"  obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"  RGB range: [{obs[:,:,:3].min():.1f}, {obs[:,:,:3].max():.1f}]")

    n_actions = 4
    raw_obs = np.zeros((n_actions + 1, env.screenHeight, env.screenWidth, 5))
    raw_obs[0] = obs
    rewards = np.zeros(n_actions + 1)
    rewards[0] = 1.0

    # Set up video_recorder (list of cv2.VideoWriter) like visualize_mpc does
    video_path = os.path.join(output_dir, 'full_manipulation.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vw = cv2.VideoWriter(video_path, fourcc, 30, (env.screenWidth, env.screenHeight))
    video_recorder = [vw] if vw.isOpened() else None
    print(f"  Video recorder opened: {vw.isOpened()}")

    for i in range(1, n_actions + 1):
        action, _ = env.sample_action(1)
        # Pass video_recorder to step() — this records every intermediate frame
        obs = env.step(action[0, 0], video_recorder=video_recorder)
        if obs is None:
            print(f"  Sim exploded at step {i}")
            raw_obs = raw_obs[:i]
            rewards = rewards[:i]
            break
        raw_obs[i] = obs
        rewards[i] = rewards[i-1] - 0.1 + np.random.uniform(-0.05, 0.05)
        print(f"  Action {i}: obs shape={obs.shape}, RGB max={obs[:,:,:3].max():.1f}")

    # Release video
    if video_recorder:
        video_recorder[0].release()
    video_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
    print(f"\n  Full manipulation video: {video_size:,} bytes")

    print(f"\nCollected {len(raw_obs)} snapshot observations from {n_actions} actions")

    # Test 1: Save snapshot images (one per action)
    print("\n--- Test 1: Save snapshot images ---")
    images_dir = os.path.join(output_dir, 'images')
    save_observations_as_images(raw_obs, images_dir)

    # Test 2: Plot rewards
    print("\n--- Test 2: Plot rewards ---")
    plot_path = os.path.join(output_dir, 'rewards.png')
    plot_reward_evolution(rewards, plot_path)

    # Summary
    print(f"\n{'=' * 60}")
    print("All outputs:")
    for root, dirs, files in os.walk(output_dir):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, output_dir)
            size = os.path.getsize(fpath)
            print(f"  {rel:40s} {size:>10,} bytes")
    print(f"\n>>> Check: {output_dir}")


if __name__ == "__main__":
    main()
