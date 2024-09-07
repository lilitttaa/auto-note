from typing import List
import cv2
import time
import numpy as np


def calculate_mse(frame1, frame2):
    """计算两帧之间的MSE（均方误差）"""
    err = np.sum((frame1.astype("float") - frame2.astype("float")) ** 2)
    err /= float(frame1.shape[0] * frame1.shape[1])
    return err


class FrameInfo:
    def __init__(self, frame: np.ndarray, frame_time: float):
        self.frame = frame
        self.frame_time = frame_time


def generate_key_frame_infos(
    cap: cv2.VideoCapture,
    fps: float,
    interval: int,
    mse_threshold: int,
):
    frame = None
    previous_frame = None
    frame_time = 0
    frame_infos: List[FrameInfo] = []
    while True:
        ret, frame = cap.read()
        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if not ret:
            break
        if int(frame_count) % int(fps * interval) == 0:
            if previous_frame is not None:
                if calculate_mse(previous_frame, frame) > mse_threshold:
                    print("generate key frame: ", frame_time)
                    frame_infos.append(FrameInfo(frame, frame_time))
            previous_frame = frame.copy()
    return frame_infos


def find_precision_frame(
    cap: cv2.VideoCapture,
    key_frame_info: FrameInfo,
    delta_frame_time: float,
    fps: float,
    interval: float,
    mse_threshold: int,
):
    start_frame_time = max(0, key_frame_info.frame_time - delta_frame_time)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_frame_time)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not ret:
            break
        if int(frame_count) % int(fps * interval) == 0:
            if calculate_mse(key_frame_info.frame, frame) <= mse_threshold:
                return FrameInfo(frame, frame_time)
        frame_count += 1
    raise Exception("Can't find precision frame")


def format_millis_with_ms(millis: float):
    total_seconds = int(millis / 1000.0)
    milliseconds = int((millis - (total_seconds * 1000)) / 10)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return "{:02}:{:02}:{:02},{:01d}".format(hours, minutes, seconds, milliseconds)


def main():
    video_path = r"D:\Project\auto-note\155050182-1-16.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    mse_threshold = 1000
    print("start generate key frame infos...")
    frame_infos = generate_key_frame_infos(cap, fps, 10, mse_threshold)
    print("generate key frame infos done.")
    for frame_info in frame_infos:
        frame_time_str = time.strftime("%H:%M:%S", time.gmtime(frame_info.frame_time))
        print(f"frame_time: {frame_time_str}")
    print("-----------------------------")
    print("start find precision frame...")
    precision_frame_infos: List[FrameInfo] = []
    for frame_info in frame_infos:
        precision_frame_info = find_precision_frame(
            cap,
            frame_info,
            10 * 1000,
            fps,
            0.5,
            mse_threshold,
        )
        precision_frame_infos.append(precision_frame_info)
        frame_time_str = format_millis_with_ms(precision_frame_info.frame_time)
        file_name = time.strftime(
            "%H-%M-%S", time.gmtime(precision_frame_info.frame_time / 1000)
        )
        cv2.imwrite(f"tmp\{file_name}.jpg", precision_frame_info.frame)
        print("save frame: ", frame_time_str)
    print("find precision frame done.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
