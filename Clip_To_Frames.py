import cv2
import os


def video_to_frames(video_path, output_folder, step):

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % step == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.png")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ Done! Saved {saved_count} frames to {output_folder}")

#function stream line d to extract frames every 2 frames

def extract_frames_every_n(video_path, output_folder, n):
    video_to_frames(video_path, output_folder, step=n)

video_to_frames(r"F:\Clash Royal\2025-09-07 21-06-43.mp4", r"F:\Clash Royal Troop Detection Data Set\Training Frames v2", step=15)