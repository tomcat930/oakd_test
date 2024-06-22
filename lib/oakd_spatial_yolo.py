#!/usr/bin/env python3
# coding: utf-8

import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import blobconverter
import cv2
import depthai as dai
import numpy as np

from .util import HostSync, TextHelper

DISPLAY_WINDOW_SIZE_RATE = 2.0
idColors = np.random.random(size=(256, 3)) * 256


class OakdSpatialYolo(object):
    """OAK-Dを使用してYOLO3次元物体認識を行うクラス。"""

    def __init__(
        self,
        config_path: str,
        model_path: str,
        fps: int,
        fov: float = 73.0,
        cam_debug: bool = False,
        robot_coordinate: bool = False,
        show_bird_frame: bool = True,
    ) -> None:
        """クラスの初期化メソッド。

        Args:
            config_path (str): YOLOモデルの設定ファイルのパス。
            model_path (str): YOLOモデルファイルのパス。
            fps (int): カメラのフレームレート。
            fov (float): カメラの視野角 (degree)。defaultはOAK-D LiteのHFOVの73.0[deg]。
            cam_debug (bool, optional): カメラのデバッグ用ウィンドウを表示するかどうか。デフォルトはFalse。
            robot_coordinate (bool, optional): ロボットのヘッド向きを使って物体の位置を変換するかどうか。デフォルトはFalse。
            show_bird_frame (bool, optional): 俯瞰フレームを表示するかどうか。デフォルトはTrue。

        """
        if not Path(config_path).exists():
            raise ValueError("Path {} does not exist!".format(config_path))
        with Path(config_path).open() as f:
            config = json.load(f)
        nnConfig = config.get("nn_config", {})

        self.width = 640
        self.height = 640
        # parse input shape
        if "input_size" in nnConfig:
            self.width, self.height = tuple(
                map(int, nnConfig.get("input_size").split("x"))
            )

        self.jet_custom = cv2.applyColorMap(
            np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET
        )
        self.jet_custom = self.jet_custom[::-1]
        self.jet_custom[0] = [0, 0, 0]
        # extract metadata
        metadata = nnConfig.get("NN_specific_metadata", {})
        self.classes = metadata.get("classes", {})
        self.coordinates = metadata.get("coordinates", {})
        self.anchors = metadata.get("anchors", {})
        self.anchorMasks = metadata.get("anchor_masks", {})
        self.iouThreshold = metadata.get("iou_threshold", {})
        self.confidenceThreshold = metadata.get("confidence_threshold", {})
        # parse labels
        nnMappings = config.get("mappings", {})
        self.labels = nnMappings.get("labels", {})

        self.nn_path = Path(model_path)
        # get model path
        if not self.nn_path.exists():
            print(
                "No blob found at {}. Looking into DepthAI model zoo.".format(
                    self.nn_path
                )
            )
            self.nn_path = Path(
                blobconverter.from_zoo(
                    model_path, shaves=6, zoo_type="depthai", use_cache=True
                )
            )
        self.fps = fps
        self.fov = fov
        self.cam_debug = cam_debug
        self.robot_coordinate = robot_coordinate
        self.show_bird_frame = show_bird_frame
        self.max_z = 15000  # [mm]
        self._stack = contextlib.ExitStack()
        self._pipeline = self._create_pipeline()
        self._device = self._stack.enter_context(dai.Device(self._pipeline))
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        self.qRgb = self._device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.qDet = self._device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        self.qRaw = self._device.getOutputQueue(name="raw", maxSize=4, blocking=False)
        self.qDepth = self._device.getOutputQueue(
            name="depth", maxSize=4, blocking=False
        )
        self.counter = 0
        self.startTime = time.monotonic()
        self.frame_name = 0
        self.dir_name = ""
        self.path = ""
        self.num = 0
        self.text = TextHelper()
        if self.robot_coordinate:
            from akari_client import AkariClient

            self.akari = AkariClient()
            self.joints = self.akari.joints
            self.sync = HostSync(5)
        else:
            self.sync = HostSync(4)
        if self.show_bird_frame:
            self.bird_eye_frame = self.create_bird_frame()
        self.raw_frame = None

    def close(self) -> None:
        """OAK-Dを閉じる。"""
        self._device.close()

    def convert_to_pos_from_akari(self, pos: Any, pitch: float, yaw: float) -> Any:
        """入力されたAKARIのヘッドの向きに応じて、カメラからの三次元位置をAKARI正面からの位置に変換する。

        Args:
            pos (Any): 物体の3次元位置。
            pitch (float): AKARIのヘッドのチルト角度。
            yaw (float): AKARIのヘッドのパン角度。

        Returns:
            Any: 変換後の3次元位置。

        """
        pitch = -1 * pitch
        yaw = -1 * yaw
        cur_pos = np.array([[pos.x], [pos.y], [pos.z]])
        arr_y = np.array(
            [
                [math.cos(yaw), 0, math.sin(yaw)],
                [0, 1, 0],
                [-math.sin(yaw), 0, math.cos(yaw)],
            ]
        )
        arr_p = np.array(
            [
                [1, 0, 0],
                [
                    0,
                    math.cos(pitch),
                    -math.sin(pitch),
                ],
                [0, math.sin(pitch), math.cos(pitch)],
            ]
        )
        ans = arr_y @ arr_p @ cur_pos
        return ans

    def get_labels(self) -> List[Any]:
        """認識ラベルファイルから読み込んだラベルのリストを取得する。

        Returns:
            List[str]: 認識ラベルのリスト。

        """
        return self.labels

    def _create_pipeline(self) -> dai.Pipeline:
        """OAK-Dのパイプラインを作成する。

        Returns:
            dai.Pipeline: OAK-Dのパイプライン。

        """
        # Create pipeline
        pipeline = dai.Pipeline()
        device = dai.Device()

        # Define sources and outputs
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewKeepAspectRatio(False)

        spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")

        # Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        camRgb.setPreviewSize(1920, 1080)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        camRgb.setFps(self.fps)
        try:
            calibData = device.readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.CAM_A)
            if lensPosition:
                camRgb.initialControl.setManualFocus(lensPosition)
        except BaseException:
            raise
        # Use ImageMqnip to resize with letterboxing
        manip = pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(self.width * self.height * 3)
        manip.initialConfig.setResizeThumbnail(self.width, self.height)
        camRgb.preview.link(manip.inputImage)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # setting node configs
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        monoRight.setFps(self.fps)
        monoLeft.setFps(self.fps)

        spatialDetectionNetwork.setBlobPath(self.nn_path)
        spatialDetectionNetwork.setConfidenceThreshold(self.confidenceThreshold)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
        spatialDetectionNetwork.setDepthLowerThreshold(300)
        spatialDetectionNetwork.setDepthUpperThreshold(35000)

        # Yolo specific parameters
        spatialDetectionNetwork.setNumClasses(self.classes)
        spatialDetectionNetwork.setCoordinateSize(self.coordinates)
        spatialDetectionNetwork.setAnchors(self.anchors)
        spatialDetectionNetwork.setAnchorMasks(self.anchorMasks)
        spatialDetectionNetwork.setIouThreshold(self.iouThreshold)

        manip.out.link(spatialDetectionNetwork.input)
        spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)

        xoutNn = pipeline.create(dai.node.XLinkOut)
        xoutNn.setStreamName("nn")
        spatialDetectionNetwork.out.link(xoutNn.input)

        xoutRaw = pipeline.create(dai.node.XLinkOut)
        xoutRaw.setStreamName("raw")
        camRgb.video.link(xoutRaw.input)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
        return pipeline

    def frame_norm(self, frame: np.ndarray, bbox: Tuple[float]) -> List[int]:
        """画像フレーム内のbounding boxの座標をフレームサイズで正規化する。

        Args:
            frame (np.ndarray): 画像フレーム。
            bbox (Tuple[float]): bounding boxの座標 (xmin, ymin, xmax, ymax)。

        Returns:
            List[int]: フレームサイズで正規化された整数座標のリスト。

        """
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def get_frame(self) -> Union[np.ndarray, List[Any]]:
        """フレーム画像と検出結果を取得する。

        Returns:
            Union[np.ndarray, List[Any]]: フレーム画像と検出結果のリストのタプル。

        """
        frame = None
        detections = []
        ret = False
        try:
            ret = self.qRgb.has()
            if ret:
                rgb_mes = self.qRgb.get()
                self.sync.add_msg("rgb", rgb_mes)
                if self.robot_coordinate:
                    self.sync.add_msg(
                        "head_pos",
                        self.joints.get_joint_positions(),
                        str(rgb_mes.getSequenceNum()),
                    )
        except BaseException:
            raise
        ret = False
        try:
            ret = self.qDepth.has()
            if ret:
                self.sync.add_msg("depth", self.qDepth.get())
        except BaseException:
            raise
        ret = False
        try:
            ret = self.qRaw.has()
            if ret:
                self.sync.add_msg("raw", self.qRaw.get())
        except BaseException:
            raise
        ret = False
        try:
            ret = self.qDet.has()
            if ret:
                self.sync.add_msg("detections", self.qDet.get())
                self.counter += 1
        except BaseException:
            raise
        msgs = self.sync.get_msgs()
        if msgs is not None:
            detections = msgs["detections"].detections
            frame = msgs["rgb"].getCvFrame()
            depthFrame = msgs["depth"].getFrame()
            self.raw_frame = msgs["raw"].getCvFrame()
            depthFrameColor = cv2.normalize(
                depthFrame, None, 256, 0, cv2.NORM_INF, cv2.CV_8UC3
            )
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, self.jet_custom)
            if self.cam_debug:
                cv2.imshow("rgb", cv2.resize(frame, (self.width, self.height)))
                cv2.imshow(
                    "depth",
                    cv2.resize(depthFrameColor, (self.width, int(self.width * 3 / 4))),
                )
            height = int(frame.shape[1] * 9 / 16)
            width = frame.shape[1]
            brank_height = width - height
            frame = frame[
                int(brank_height / 2) : int(frame.shape[0] - brank_height / 2),
                0:width,
            ]
            for detection in detections:
                # Fix ymin and ymax to cropped frame pos
                detection.ymin = (width / height) * detection.ymin - (
                    brank_height / 2 / height
                )
                detection.ymax = (width / height) * detection.ymax - (
                    brank_height / 2 / height
                )
            if self.robot_coordinate:
                self.pos = msgs["head_pos"]
                for detection in detections:
                    converted_pos = self.convert_to_pos_from_akari(
                        detection.spatialCoordinates, self.pos["tilt"], self.pos["pan"]
                    )
                    detection.spatialCoordinates.x = converted_pos[0][0]
                    detection.spatialCoordinates.y = converted_pos[1][0]
                    detection.spatialCoordinates.z = converted_pos[2][0]
        return frame, detections

    def get_raw_frame(self) -> np.ndarray:
        """カメラで撮影した生の画像フレームを取得する。

        Returns:
            np.ndarray: 生画像フレーム。

        """
        return self.raw_frame

    def get_labeled_frame(
        self,
        frame: np.ndarray,
        detections: List[Any],
        id: Optional[int] = None,
        disp_info: bool = False,
    ) -> np.ndarray:
        """認識結果をフレーム画像に描画する。

        Args:
            frame (np.ndarray): 画像フレーム。
            detections (List[Any]): 検出結果のリスト。
            id (Optional[int], optional): 描画するオブジェクトのID。指定すると、そのIDのみを描画した画像フレームを返す。指定しない場合は全てのオブジェクトを描画する。
            disp_info (bool, optional): クラス名とconfidenceをフレーム内に表示するかどうか。デフォルトはFalse。

        Returns:
            np.ndarray: 描画された画像フレーム。

        """
        for detection in detections:
            if id is not None and detections.id != id:
                continue
            # Denormalize bounding box
            bbox = self.frame_norm(
                frame,
                (
                    detection.xmin,
                    detection.ymin,
                    detection.xmax,
                    detection.ymax,
                ),
            )
            x1 = bbox[0]
            x2 = bbox[2]
            y1 = bbox[1]
            y2 = bbox[3]
            try:
                label = self.labels[detection.label]
            except BaseException:
                label = detection.label
                print
            self.text.rectangle(frame, (x1, y1), (x2, y2), idColors[detection.label])
            if disp_info:
                self.text.put_text(frame, str(label), (x1 + 10, y1 + 20))
                self.text.put_text(
                    frame,
                    "{:.0f}%".format(detection.confidence * 100),
                    (x1 + 10, y1 + 50),
                )
                if detection.spatialCoordinates.z != 0:
                    self.text.put_text(
                        frame,
                        "X: {:.2f} m".format(detection.spatialCoordinates.x / 1000),
                        (x1 + 10, y1 + 80),
                    )
                    self.text.put_text(
                        frame,
                        "Y: {:.2f} m".format(detection.spatialCoordinates.y / 1000),
                        (x1 + 10, y1 + 110),
                    )
                    self.text.put_text(
                        frame,
                        "Z: {:.2f} m".format(detection.spatialCoordinates.z / 1000),
                        (x1 + 10, y1 + 140),
                    )
        return frame

    def display_frame(
        self, name: str, frame: np.ndarray, detections: List[Any]
    ) -> None:
        """画像フレームと認識結果を描画する。

        Args:
            name (str): ウィンドウ名。
            frame (np.ndarray): 画像フレーム。
            detections (List[Any]): 認識結果のリスト。

        """
        if frame is not None:
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * DISPLAY_WINDOW_SIZE_RATE),
                    int(frame.shape[0] * DISPLAY_WINDOW_SIZE_RATE),
                ),
            )
            if detections is not None:
                frame = self.get_labeled_frame(
                    frame=frame, detections=detections, disp_info=True
                )
            cv2.putText(
                frame,
                "NN fps: {:.2f}".format(
                    self.counter / (time.monotonic() - self.startTime)
                ),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.3,
                (255, 255, 255),
            )
            # Show the frame
            cv2.imshow(name, frame)
            if self.show_bird_frame:
                self.draw_bird_frame(detections)

    def create_bird_frame(self) -> np.ndarray:
        """
        俯瞰フレームを生成する。

        Returns:
            np.ndarray: 俯瞰フレーム。

        """
        fov = self.fov
        frame = np.zeros((300, 300, 3), np.uint8)
        cv2.rectangle(
            frame, (0, 283), (frame.shape[1], frame.shape[0]), (70, 70, 70), -1
        )

        alpha = (180 - fov) / 2
        center = int(frame.shape[1] / 2)
        max_p = frame.shape[0] - int(math.tan(math.radians(alpha)) * center)
        fov_cnt = np.array(
            [
                (0, frame.shape[0]),
                (frame.shape[1], frame.shape[0]),
                (frame.shape[1], max_p),
                (center, frame.shape[0]),
                (0, max_p),
                (0, frame.shape[0]),
            ]
        )
        cv2.fillPoly(frame, [fov_cnt], color=(70, 70, 70))
        return frame

    def update_bird_frame_distance(self, distance: int) -> None:
        """俯瞰フレームの距離方向の表示最大値を変更する。
        Args:
            distance (int): 最大距離[mm]。
        """
        self.max_z = distance

    def pos_to_point_x(self, frame_width: int, pos_x: float) -> int:
        """
        3次元位置をbird frame上のx座標に変換する

        Args:
            frame_width (int): bird frameの幅
            pos_x (float): 3次元位置のx

        Returns:
            int: bird frame上のx座標
        """
        return int(pos_x / self.max_z * frame_width + frame_width / 2)

    def pos_to_point_y(self, frame_height: int, pos_z: float) -> int:
        """
        3次元位置をbird frame上のy座標に変換する

        Args:
            frame_height (int): bird frameの高さ
            pos_z (float): 3次元位置のz

        Returns:
            int: bird frame上のy座標
        """
        return frame_height - int(pos_z / self.max_z * frame_height) - 20

    def draw_bird_frame(self, detections: List[Any], show_labels: bool = False) -> None:
        """
        俯瞰フレームに検出結果を描画する。

        Args:
            detections (List[Any]): 認識結果のリスト。
            show_labels (bool, optional): ラベルを表示するかどうか。デフォルトはFalse。

        """
        birds = self.bird_eye_frame.copy()
        for i in range(0, len(detections)):
            point_y = self.pos_to_point_y(
                birds.shape[0], detections[i].spatialCoordinates.z
            )
            point_x = self.pos_to_point_x(
                birds.shape[1], detections[i].spatialCoordinates.x
            )
            if detections[i].label is not None:
                if show_labels:
                    cv2.putText(
                        birds,
                        self.labels[detections[i].label],
                        (point_x - 30, point_y + 5),
                        cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        idColors[detections[i].label],
                    )
                cv2.circle(
                    birds,
                    (point_x, point_y),
                    2,
                    idColors[detections[i].label],
                    thickness=5,
                    lineType=8,
                    shift=0,
                )
        cv2.imshow("birds", birds)