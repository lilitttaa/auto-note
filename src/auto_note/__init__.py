from typing import List
import cv2
import time
import numpy as np
import os
import shutil


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
    print("start generate key frame infos...")
    frame = None
    previous_frame = None
    frame_time = 0
    frame_infos: List[FrameInfo] = []
    while True:
        ret, frame = cap.read()
        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        if not ret:
            break
        if int(frame_count) % int(fps * interval) == 0:
            if previous_frame is not None:
                if calculate_mse(previous_frame, frame) > mse_threshold:
                    print("generate key frame: ", frame_time)
                    frame_infos.append(FrameInfo(frame, frame_time))
            else:
                print("generate key frame: ", frame_time)
                frame_infos.append(FrameInfo(frame, frame_time))
            previous_frame = frame.copy()
    print("generate key frame infos done.")
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
    return key_frame_info


def format_millis_with_ms(millis: float):
    total_seconds = int(millis / 1000.0)
    milliseconds = int((millis - (total_seconds * 1000)) / 10)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return "{:02}:{:02}:{:02},{:01d}".format(hours, minutes, seconds, milliseconds)


class SubtitleInfo:
    def __init__(self, time: float, text: str):
        self.time = time
        self.text = text


def read_subtitle_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    subtitle_infos: List[SubtitleInfo] = []
    for i in range(0, len(lines)):
        import re

        line = lines[i].strip()
        time_str = line.split(" ")[0]
        time = time_str.split(":")
        if len(time) == 3:
            seconds = int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
        else:
            seconds = int(time[0]) * 60 + int(time[1])
        miliseconds = seconds * 1000
        text = lines[i].replace(time_str, "").strip()
        subtitle_infos.append(SubtitleInfo(miliseconds, text))
    return subtitle_infos


class SubtitleFrameInfo:
    def __init__(self, frame_info: FrameInfo, subtitle_infs: List[SubtitleInfo]):
        self.frame_info = frame_info
        self.subtitle_infos = subtitle_infs

    def to_markdown(self, enable_img_sub_dir: bool):
        frame_time_str = format_millis_with_ms(self.frame_info.frame_time)
        markdown = f"### {frame_time_str}\n\n"
        file_name = time.strftime(
            "%H-%M-%S", time.gmtime(self.frame_info.frame_time / 1000)
        )
        markdown += (
            f"![{file_name}](tmp/{file_name}.jpg)\n\n"
            if enable_img_sub_dir
            else f"![{file_name}]({file_name}.jpg)\n\n"
        )
        for subtitle_info in self.subtitle_infos:
            markdown += f"{subtitle_info.text}\n\n"
        return markdown


def split_subtitle(subtitle_infos: List[SubtitleInfo], frame_infos: List[FrameInfo]):
    subtitle_frame_infos: List[SubtitleFrameInfo] = []
    for i in range(0, len(frame_infos)):
        frame_info = frame_infos[i]
        attach_subtitle_infos: List[SubtitleInfo] = []
        for j in range(0, len(subtitle_infos)):
            subtitle_info = subtitle_infos[j]
            if subtitle_info.time >= frame_info.frame_time and (
                i + 1 >= len(frame_infos)
                or subtitle_info.time < frame_infos[i + 1].frame_time
            ):
                attach_subtitle_infos.append(subtitle_info)
        subtitle_frame_infos.append(
            SubtitleFrameInfo(frame_info, attach_subtitle_infos)
        )
    return subtitle_frame_infos


def reset_dir(dir: str):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def main():
    note_name = "Note_Games101_3"
    dir_path = r"D:\Project\auto-note"
    mse_threshold = 1000
    interval = 10
    precious_interval = 0.5
    low_res_video_path = r"D:\Project\auto-note\games101_3.mp4"
    high_res_video_path = r"D:\Project\auto-note\games101_3.mp4"
    enable_img_sub_dir = True

    output_dir_path = os.path.join(dir_path, note_name)
    reset_dir(output_dir_path)
    if enable_img_sub_dir:
        reset_dir(os.path.join(output_dir_path, "tmp"))

    cap = cv2.VideoCapture(low_res_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_infos = generate_key_frame_infos(cap, fps, interval, mse_threshold)

    print("start find precision frame...")
    precision_frame_infos: List[FrameInfo] = []
    for frame_info in frame_infos:
        precision_frame_info = find_precision_frame(
            cap,
            frame_info,
            interval * 1000,
            fps,
            precious_interval,
            mse_threshold,
        )
        precision_frame_infos.append(precision_frame_info)
    print("find precision frame done.")
    cap.release()
    cv2.destroyAllWindows()

    print("start save frame...")
    cap = cv2.VideoCapture(high_res_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    for precision_frame_info in precision_frame_infos:
        cap.set(cv2.CAP_PROP_POS_MSEC, precision_frame_info.frame_time)
        ret, frame = cap.read()
        if not ret:
            break
        frame_time_str = format_millis_with_ms(precision_frame_info.frame_time)
        file_name = time.strftime(
            "%H-%M-%S", time.gmtime(precision_frame_info.frame_time / 1000)
        )
        file_path = (
            os.path.join(note_name, "tmp", f"{file_name}.jpg")
            if enable_img_sub_dir
            else os.path.join(note_name, f"{file_name}.jpg")
        )
        cv2.imwrite(file_path, frame)
        print("save frame: ", frame_time_str)
    cap.release()
    cv2.destroyAllWindows()
    print("save frame done.")

    print("start generate markdown...")
    subtitle_infos = read_subtitle_file("script.txt")
    subtitle_frame_infos = split_subtitle(subtitle_infos, precision_frame_infos)
    with open(
        os.path.join(output_dir_path, f"{note_name}.md"), "w", encoding="utf-8"
    ) as f:
        f.write("---\n")
        f.write(f"title: {note_name}\n")
        f.write("---\n\n")
        for subtitle_frame_info in subtitle_frame_infos:
            f.write(subtitle_frame_info.to_markdown(enable_img_sub_dir))
            f.write("\n")
    print("generate markdown done.")


if __name__ == "__main__":
    main()
