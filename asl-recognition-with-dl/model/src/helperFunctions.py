import cv2
from pytube import YouTube

def download_video(dir, url):
    filepath = YouTube(url).streams.first().download(dir)
    return filepath

def extract_and_save_cropped_frames(start_frame_number, end_frame_number, bbox, video_path, save_dir):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    
    frame_count = 0  # Initialize a counter for naming files
    while cap.isOpened():
        ret, frame = cap.read()
        if ret and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame_number:
            x0, y0, x1, y1 = bbox
            height, width = frame.shape[:2]
            cropped_frame = frame[int(y0 * height):int(y1 * height), int(x0 * width):int(x1 * width)]
            # Save cropped frame to disk
            cv2.imwrite(f"{save_dir}/cropped_frame_{frame_count}.png", cropped_frame)
            frame_count += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()