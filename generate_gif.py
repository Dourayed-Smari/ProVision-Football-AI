import cv2
from PIL import Image
import os

video_path = 'output_videos/c1_out.mp4'
gif_path = 'output_videos/demo.gif'

cap = cv2.VideoCapture(video_path)
frames = []

# Take every 3rd frame to keep GIF size small, for about 5 seconds (150 frames total / 3 = 50 frames)
count = 0
while len(frames) < 60:
    ret, frame = cap.read()
    if not ret:
        break
    
    if count % 3 == 0:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize for GIF (640px wide is plenty for README)
        h, w = frame_rgb.shape[:2]
        new_w = 640
        new_h = int(h * (new_w / w))
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        frames.append(Image.fromarray(frame_resized))
    
    count += 1

cap.release()

if frames:
    # Save as GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=100, # 100ms per frame = 10fps
        loop=0
    )
    print(f"GIF created at {gif_path}")
else:
    print("No frames found")
