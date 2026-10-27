#!/usr/bin/env python3
"""camera.py -- Web カメラを開いて、画面への配信 (MJPEG) と記録フォルダへの録画をする。

nanoVNA_OptiTrack_get_dataset/camera_recorder.py と同じく OpenCV (DirectShow) で開くが、
**コマごとに perf_counter の時刻を打つ**ところが違う (あちらは壁時計と一定間隔の書き出しで
動画の時刻を合わせていた)。録画したコマの時刻は camera_frames.csv に残るので、
motion.csv / frames.csv と同じ時計で突き合わせられる。

録画は MotionSessionLog が sink を付け外しする (BER の記録と同じ期間だけ撮る)。
"""
from __future__ import annotations

import threading
import time

try:
    import cv2
except ImportError:                       # OpenCV が無くても BER とモーションは使えるように
    cv2 = None

STREAM_WIDTH = 640                         # 画面に流すときの横幅 (録画は元の大きさ)
JPEG_QUALITY = 75


class Camera:
    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.index = 0
        self.cap = None
        self.thread: threading.Thread | None = None
        self.stop_ev = threading.Event()
        self.frame = None                  # 最新のコマ (BGR)
        self.t = None                      # その受け取り時刻 [perf_counter]
        self.seq = 0
        self.width = self.height = 0
        self.fps = 0.0
        self.error = ""
        self.sink = None                   # (frame, t, seq) -> None。録画中だけ付く
        self._jpeg = (None, -1)            # 画面用に縮めた JPEG と、その seq
        self._times: list[float] = []

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def open(self, index: int = 0) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) がありません。uv sync で opencv-python が入ります")
        self.close()
        cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(index))
        if not cap.isOpened():
            raise RuntimeError(f"カメラ {index} を開けません (接続と、ほかのアプリが使っていないかを確かめる)")
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # 古いコマを溜めない
        except Exception:                                # noqa: BLE001
            pass
        ok, f = cap.read()
        if not ok or f is None:
            cap.release()
            raise RuntimeError(f"カメラ {index} からコマを読めません")
        self.cap, self.index = cap, int(index)
        self.height, self.width = f.shape[:2]
        self.error, self.seq, self._times = "", 0, []
        self.stop_ev.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_ev.is_set():
            ok, f = self.cap.read()
            t = time.perf_counter()
            if not ok or f is None:
                self.stop_ev.wait(0.01)
                continue
            with self.cond:
                self.frame, self.t = f, t
                self.seq += 1
                seq, sink = self.seq, self.sink
                self._times = (self._times + [t])[-60:]
                self.cond.notify_all()
            if sink is not None:
                try:
                    sink(f, t, seq)
                except Exception as e:                   # noqa: BLE001 (撮影は止めない)
                    self.error = f"録画に失敗: {type(e).__name__}: {e}"

    def close(self) -> None:
        self.stop_ev.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        with self.cond:
            self.frame, self.t = None, None
            self.cond.notify_all()

    def jpeg(self, after_seq: int, timeout: float = 1.0):
        """after_seq より新しいコマの JPEG (縮小) と seq。来なければ (None, after_seq)。"""
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq > after_seq or not self.running, timeout):
                return None, after_seq
            if self.frame is None:
                return None, after_seq
            if self._jpeg[1] == self.seq:
                return self._jpeg
            f, seq = self.frame, self.seq
        h, w = f.shape[:2]
        if w > STREAM_WIDTH:
            f = cv2.resize(f, (STREAM_WIDTH, int(h * STREAM_WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        out = (buf.tobytes() if ok else None, seq)
        with self.cond:
            self._jpeg = out
        return out

    def measured_fps(self) -> float:
        with self.lock:
            ts = list(self._times)
        return (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 0.0

    def status(self) -> dict:
        return dict(available=cv2 is not None, running=self.running, index=self.index,
                    width=self.width, height=self.height, fps=round(self.measured_fps(), 1),
                    seq=self.seq, error=self.error)


class VideoLog:
    """記録フォルダへの録画: camera.mp4 と、コマごとの時刻 camera_frames.csv。"""

    def __init__(self, cam: Camera, path_mp4, path_csv, perf0: float):
        fps = cam.measured_fps() or 30.0
        self.fps = float(round(min(max(fps, 5.0), 60.0)))
        self.w = cv2.VideoWriter(str(path_mp4), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                 (cam.width, cam.height))
        if not self.w.isOpened():
            raise RuntimeError(f"動画ファイルを開けません: {path_mp4}")
        self.csv = open(path_csv, "w", encoding="utf-8", newline="", buffering=1 << 16)
        self.csv.write("video_frame,t,camera_seq\n")
        self.perf0, self.n, self.lock, self.cam = perf0, 0, threading.Lock(), cam
        with cam.lock:
            cam.sink = self.write

    def write(self, frame, t, seq) -> None:
        with self.lock:
            if self.w is None:
                return
            self.w.write(frame)
            self.csv.write(f"{self.n},{t - self.perf0:.6f},{seq}\n")
            self.n += 1

    def close(self) -> dict:
        with self.cam.lock:
            if self.cam.sink == self.write:
                self.cam.sink = None
        with self.lock:
            if self.w is not None:
                self.w.release()
                self.w = None
                self.csv.close()
        return dict(frames=self.n, nominal_fps=self.fps)
