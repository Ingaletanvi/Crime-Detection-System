# scripts/extract_clips.py

import os
import cv2

def extract_clips_from_videos(video_dir, output_dir, clip_len=16):
    os.makedirs(output_dir, exist_ok=True)

    classes = os.listdir(video_dir)
    for class_name in classes:
        class_path = os.path.join(video_dir, class_name)
        label = 'crime' if class_name in ['gun', 'knife'] else 'non_crime'
        save_path = os.path.join(output_dir, label)
        os.makedirs(save_path, exist_ok=True)

        for vid_name in os.listdir(class_path):
            vid_path = os.path.join(class_path, vid_name)
            cap = cv2.VideoCapture(vid_path)

            frames = []
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (112, 112))
                frames.append(frame)

                # Extract clip every `clip_len` frames
                if len(frames) == clip_len:
                    clip_folder = os.path.join(save_path, f"{class_name}_clip_{idx}")
                    os.makedirs(clip_folder, exist_ok=True)
                    for i, f in enumerate(frames):
                        cv2.imwrite(os.path.join(clip_folder, f"frame_{i:03}.jpg"), f)
                    idx += 1
                    frames = []

            cap.release()

video_dir = r"C:\Users\itanv\Desktop\CRIME youtube dataset - Copy\crime_detection_system\CrimeVideos"
output_dir = r"C:\Users\itanv\Desktop\CRIME youtube dataset - Copy\crime_detection_system\data"
extract_clips_from_videos(video_dir, output_dir)
