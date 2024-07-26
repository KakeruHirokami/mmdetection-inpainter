import cv2
import os

def getframe(video_path):
    dirname = video_path.split(".")[0]
    os.makedirs(dirname, exist_ok=True) 

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f'[1/3]Convert video to frame images {frame_number}/{total_frames-1}')
        
        # フレームを保存する場合は以下のコードを使用
        framename = str(frame_number).zfill(5)
        cv2.imwrite(f'{dirname}/{framename}.png', frame)
        
        frame_number += 1

    cap.release()
    return dirname

if __name__ == "__main__":
    video_path = '20240525_100522000_iOS.mp4'
    getframe(video_path)