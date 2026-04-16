import cv2
import os
from glob import glob

def images_to_video(
    image_dir,
    output_path="output.mp4",
    fps=0.5,
    pattern="*.png"
):
    # Collect and sort images (assumes numbering in filenames)
    image_paths = sorted(glob(os.path.join(image_dir, pattern)))

    image_paths = sorted(
        glob(os.path.join(image_dir, pattern)),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
    )

    if not image_paths:
        raise ValueError("No images found.")

    # Read first image to get size
    first_frame = cv2.imread(image_paths[0])
    height, width, _ = first_frame.shape

    # Define video writer (mp4 + H264)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely compatible
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for path in image_paths:
        frame = cv2.imread(path)

        if frame is None:
            print(f"Skipping unreadable file: {path}")
            continue

        # Ensure consistent size
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))

        video.write(frame)

    video.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    images_to_video(
        image_dir="/home/shirizly/Code/GranularBaselines/dyn-res-pile-manip/test_outputs/save_test_2026-03-18-15-51-18-729328/images",
        output_path="output.mp4",
        fps=0.5,             # <-- adjust frame rate here
        pattern="*.png"      # or *.jpg
    )