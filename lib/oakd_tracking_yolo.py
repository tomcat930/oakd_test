#!/usr/bin/env python

import contextlib
import copy
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import blobconverter
import cv2
import depthai as dai
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .util import HostSync, TextHelper

DISPLAY_WINDOW_SIZE_RATE = 2.0
idColors = np.random.random(size=(256, 3)) * 256


class OakdTrackingYolo(object):
    """OAK-Dを使用してYOLO3次元物体トラッキングを行うクラス。"""

    def __init__(
        self,
        config_path: str,
        model_path: str,
        fps: int,
        fov: float = 73.0,
        cam_debug: bool = False,
        robot_coordinate: bool = False,
        track_targets: Optional[List[Union[int, str]]] = None,
        show_bird_frame: bool = True,
        show_spatial_frame: bool = False,
        show_orbit: bool = False,
        log_path: Optional[str] = None,
    ) -> None:
        """クラスの初期化メソッド。

        Args:
            config_path (str): YOLOモデルの設定ファイルのパス。
            model_path (str): YOLOモデルファイルのパス。
            fps (int): カメラのフレームレート。
            fov (float): カメラの視野角 (degree)。defaultはOAK-D LiteのHFOVの73.0[deg]
            cam_debug (bool, optional): カメラのデバッグ用ウィンドウを表示するかどうか。デフォルトはFalse。
            robot_coordinate (bool, optional): ロボットのヘッド向きを使って物体の位置を変換するかどうか。デフォルトはFalse。
            track_targets (Optional[List[Union[int, str]]], optional): トラッキング対象のラベルリスト。デフォルトはNone。
            show_bird_frame (bool, optional): 俯瞰フレームを表示するかどうか。デフォルトはTrue。
            show_spatial_frame (bool, optional): 3次元フレームを表示するかどうか。デフォルトはFalse。
            show_orbit (bool, optional): 3次元軌道を表示するかどうか。デフォルトはFalse。
            log_path (Optional[str], optional): 物体の軌道履歴を保存するパス。show_orbitがTrueの時のみ有効。

        """
        if not Path(config_path).exists():
            raise ValueError("Path {} does not exist!".format(config_path))
        with Path(config_path).open() as f:
            config = json.load(f)
        nnConfig = config.get("nn_config", {})

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
        self.track_targets = track_targets
        self.robot_coordinate = robot_coordinate
        self.show_bird_frame = show_bird_frame
        self.show_spatial_frame = show_spatial_frame
        self.show_orbit = show_orbit
        self.max_z = 15000  # [mm]
        if self.show_orbit:
            self.orbit_data_list = OrbitDataList(labels=self.labels, log_path=log_path)
        self._stack = contextlib.ExitStack()
        self._pipeline = self._create_pipeline()
        self._device = self._stack.enter_context(dai.Device(self._pipeline))
        self.qRgb = self._device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.qDet = self._device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        self.qRaw = self._device.getOutputQueue(name="raw", maxSize=4, blocking=False)
        self.qDepth = self._device.getOutputQueue(
            name="depth", maxSize=4, blocking=False
        )
        self.qTrack = self._device.getOutputQueue(
            "tracklets", maxSize=4, blocking=False
        )
        self.spatial_frame_range = 3.0
        self.counter = 0
        self.start_time = time.monotonic()
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
        self.track = None
        if self.show_bird_frame:
            self.bird_eye_frame = self.create_bird_frame()
        if self.show_spatial_frame:
            self.create_spatial_frame()
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

    def get_labels(self) -> Any:
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
        objectTracker = pipeline.create(dai.node.ObjectTracker)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        trackerOut = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        trackerOut.setStreamName("tracklets")

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

        # トラッキングする物体のID、もしくは物体名を配列で渡す。
        # 指定がない場合はconfigファイル内の全物体をトラッキング対象に指定
        if self.track_targets is None:
            objectTracker.setDetectionLabelsToTrack(list(range(self.classes)))
        else:
            target_list = []
            for target in self.track_targets:
                if isinstance(target, int):
                    target_list.append(target)
                elif isinstance(target, str):
                    if target in self.labels:
                        target_list.append(self.labels.index(target))
            objectTracker.setDetectionLabelsToTrack(target_list)
        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        objectTracker.setTrackerIdAssignmentPolicy(
            dai.TrackerIdAssignmentPolicy.UNIQUE_ID
        )

        manip.out.link(spatialDetectionNetwork.input)
        spatialDetectionNetwork.passthrough.link(xoutRgb.input)
        stereo.depth.link(spatialDetectionNetwork.inputDepth)

        xoutNn = pipeline.create(dai.node.XLinkOut)
        xoutNn.setStreamName("nn")
        spatialDetectionNetwork.out.link(xoutNn.input)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        xoutRaw = pipeline.create(dai.node.XLinkOut)
        xoutRaw.setStreamName("raw")
        camRgb.video.link(xoutRaw.input)
        spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
        spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)
        spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
        spatialDetectionNetwork.out.link(objectTracker.inputDetections)
        objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
        objectTracker.out.link(trackerOut.input)
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

    def get_frame(self) -> Union[np.ndarray, List[Any], Any]:
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
        ret = False
        try:
            ret = self.qTrack.has()
            if ret:
                self.track = self.qTrack.get()
        except BaseException:
            raise
        msgs = self.sync.get_msgs()
        tracklets = None
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
            if self.track is not None:
                tracklets = self.track.tracklets
                for tracklet in tracklets:
                    # Fix roi to cropped frame pos
                    tracklet.roi.y = (width / height) * tracklet.roi.y - (
                        brank_height / 2 / height
                    )
                    tracklet.roi.height = tracklet.roi.height * width / height

            if self.robot_coordinate:
                self.pos = msgs["head_pos"]
                for detection in detections:
                    converted_pos = self.convert_to_pos_from_akari(
                        detection.spatialCoordinates, self.pos["tilt"], self.pos["pan"]
                    )
                    detection.spatialCoordinates.x = converted_pos[0][0]
                    detection.spatialCoordinates.y = converted_pos[1][0]
                    detection.spatialCoordinates.z = converted_pos[2][0]
                for tracklet in tracklets:
                    converted_pos = self.convert_to_pos_from_akari(
                        tracklet.spatialCoordinates, self.pos["tilt"], self.pos["pan"]
                    )
                    tracklet.spatialCoordinates.x = converted_pos[0][0]
                    tracklet.spatialCoordinates.y = converted_pos[1][0]
                    tracklet.spatialCoordinates.z = converted_pos[2][0]
            if self.show_orbit:
                self.orbit_data_list.update_orbit_data(tracklets)
        return frame, detections, tracklets

    def get_raw_frame(self) -> np.ndarray:
        """カメラで撮影した生の画像フレームを取得する。

        Returns:
            np.ndarray: 生画像フレーム。

        """
        return self.raw_frame

    def get_labeled_frame(
        self,
        frame: np.ndarray,
        tracklets: List[Any],
        id: Optional[int] = None,
        disp_info: bool = False,
    ) -> np.ndarray:
        """認識結果をフレーム画像に描画する。

        Args:
            frame (np.ndarray): 画像フレーム。
            tracklets (List[Any]): トラッキング結果のリスト。
            id (Optional[int], optional): 描画するオブジェクトのID。指定すると、そのIDのみを描画した画像フレームを返す。指定しない場合は全てのオブジェクトを描画する。
            disp_info (bool, optional): クラス名とconfidenceをフレーム内に表示するかどうか。デフォルトはFalse。

        Returns:
            np.ndarray: 描画された画像フレーム。

        """
        for tracklet in tracklets:
            if id is not None and tracklet.id != id:
                continue
            if tracklet.status.name == "TRACKED":
                roi = tracklet.roi.denormalize(frame.shape[1], frame.shape[0])
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)
                try:
                    label = self.labels[tracklet.label]
                except Exception:
                    label = tracklet.label
                self.text.rectangle(frame, (x1, y1), (x2, y2), idColors[tracklet.id])
                if disp_info:
                    self.text.put_text(frame, str(label), (x1 + 10, y1 + 20))
                    self.text.put_text(
                        frame,
                        f"ID: {[tracklet.id]}",
                        (x1 + 10, y1 + 45),
                    )
                    self.text.put_text(frame, tracklet.status.name, (x1 + 10, y1 + 70))

                    if tracklet.spatialCoordinates.z != 0:
                        self.text.put_text(
                            frame,
                            "X: {:.2f} m".format(tracklet.spatialCoordinates.x / 1000),
                            (x1 + 10, y1 + 95),
                        )
                        self.text.put_text(
                            frame,
                            "Y: {:.2f} m".format(tracklet.spatialCoordinates.y / 1000),
                            (x1 + 10, y1 + 120),
                        )
                        self.text.put_text(
                            frame,
                            "Z: {:.2f} m".format(tracklet.spatialCoordinates.z / 1000),
                            (x1 + 10, y1 + 145),
                        )
        return frame

    def display_frame(self, name: str, frame: np.ndarray, tracklets: List[Any]) -> None:
        """画像フレームと認識結果を描画する。

        Args:
            name (str): ウィンドウ名。
            frame (np.ndarray): 画像フレーム。
            tracklets (List[Any]): トラッキング結果のリスト。
            birds(bool): 俯瞰フレームを表示するかどうか。デフォルトはTrue。

        """
        if frame is not None:
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] * DISPLAY_WINDOW_SIZE_RATE),
                    int(frame.shape[0] * DISPLAY_WINDOW_SIZE_RATE),
                ),
            )
            if tracklets is not None:
                self.get_labeled_frame(frame=frame, tracklets=tracklets, disp_info=True)
            cv2.putText(
                frame,
                "NN fps: {:.2f}".format(
                    self.counter / (time.monotonic() - self.start_time)
                ),
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.3,
                (255, 255, 255),
            )
            # Show the frame
            cv2.imshow(name, frame)
            if self.show_bird_frame:
                self.draw_bird_frame(tracklets)
            if self.show_spatial_frame:
                self.draw_spatial_frame(tracklets)

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

    def draw_bird_frame(self, tracklets: List[Any], show_labels: bool = False) -> None:
        """
        俯瞰フレームに検出結果を描画する。

        Args:
            tracklets (List[Any]): トラッキング結果のリスト。
            show_labels (bool, optional): ラベルを表示するかどうか。デフォルトはFalse。

        """
        birds = self.bird_eye_frame.copy()
        if tracklets is not None:
            for i in range(0, len(tracklets)):
                if tracklets[i].status.name == "TRACKED":
                    point_y = self.pos_to_point_y(
                        birds.shape[0], tracklets[i].spatialCoordinates.z
                    )
                    point_x = self.pos_to_point_x(
                        birds.shape[1], tracklets[i].spatialCoordinates.x
                    )
                    if show_labels:
                        cv2.putText(
                            birds,
                            self.labels[tracklets[i].label],
                            (point_x - 30, point_y + 5),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.5,
                            (0, 255, 0),
                        )
                    cv2.circle(
                        birds,
                        (point_x, point_y),
                        2,
                        idColors[tracklets[i].id],
                        thickness=5,
                        lineType=8,
                        shift=0,
                    )
                    if self.show_orbit:
                        orbit = self.orbit_data_list.get_orbit_from_id(tracklets[i].id)
                        if orbit is not None:
                            prev_point: Optional[Tuple[int, int]] = None
                            for pos in orbit.pos_log:
                                cur_point = (
                                    self.pos_to_point_x(birds.shape[1], pos.x),
                                    self.pos_to_point_y(birds.shape[0], pos.z),
                                )
                                cv2.circle(
                                    birds,
                                    cur_point,
                                    2,
                                    idColors[tracklets[i].id],
                                    thickness=2,
                                    lineType=8,
                                    shift=0,
                                )
                                if prev_point is not None:
                                    cv2.line(
                                        birds,
                                        prev_point,
                                        cur_point,
                                        idColors[tracklets[i].id],
                                        thickness=1,
                                    )
                                prev_point = (
                                    self.pos_to_point_x(birds.shape[1], pos.x),
                                    self.pos_to_point_y(birds.shape[0], pos.z),
                                )
                            if prev_point is not None:
                                cv2.line(
                                    birds,
                                    prev_point,
                                    (point_x, point_y),
                                    idColors[tracklets[i].id],
                                    2,
                                )

        cv2.imshow("birds", birds)

    def set_spatial_frame_range(self, range: float) -> None:
        """3次元プロットの範囲を設定する。

        Args:
            range (float):3次元プロットの描画距離[m]。

        """
        self.spatial_frame_range = range

    def create_spatial_frame(self) -> None:
        """3次元プロットを作成する。"""
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.show()
        self.ax.view_init(elev=25, azim=-40, roll=0)

    def draw_spatial_frame(self, tracklets: List[Any]) -> None:
        """3次元プロットを描画する。

        Args:
            tracklets (List[Any]): トラッキング結果のリスト。

        """
        # AKARIのヘッドを描画
        plt.cla()
        self.ax.set_xlim(
            [-1 * self.spatial_frame_range / 2, self.spatial_frame_range / 2]
        )
        self.ax.set_ylim([0, self.spatial_frame_range])
        self.ax.set_zlim(
            [-1 * self.spatial_frame_range / 2, self.spatial_frame_range / 2]
        )
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Z")
        self.ax.set_zlabel("Y")
        self.ax.plot(
            [-1 * self.spatial_frame_range / 2, self.spatial_frame_range / 2],
            [0, 0],
            [0, 0],
            color="gray",
            linestyle="--",
        )  # x=0の基準線
        self.ax.plot(
            [-1 * self.spatial_frame_range / 2, -1 * self.spatial_frame_range / 2],
            [0, self.spatial_frame_range],
            [0, 0],
            color="gray",
            linestyle="--",
        )  # x=0の基準線
        self.ax.plot(
            [0, 0],
            [0, self.spatial_frame_range],
            [-1 * self.spatial_frame_range / 2, -1 * self.spatial_frame_range / 2],
            color="gray",
            linestyle="--",
        )  # x=0の基準線
        self.ax.plot(
            [0, 0],
            [0, 0],
            [-1 * self.spatial_frame_range / 2, self.spatial_frame_range / 2],
            color="gray",
            linestyle="--",
        )  # y=0の基準線
        cam_width = self.spatial_frame_range / 10.0
        cam_height = self.spatial_frame_range / 30.0
        cam_depth = self.spatial_frame_range / 15.0
        vertices = np.array(
            [
                [0, 0, 0],
                [cam_width, cam_depth, cam_height],
                [-1 * cam_width, cam_depth, cam_height],
                [-1 * cam_width, cam_depth, -1 * cam_height],
                [cam_width, cam_depth, -1 * cam_height],
            ]
        )
        pan = 0
        tilt = 0
        if self.robot_coordinate:
            pan = self.joints.get_joint_positions()["pan"]
            tilt = self.joints.get_joint_positions()["tilt"]
            R_pan = np.array(
                [
                    [np.cos(pan), -np.sin(pan), 0],
                    [np.sin(pan), np.cos(pan), 0],
                    [0, 0, 1],
                ]
            )
            R_tilt = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(tilt), -np.sin(tilt)],
                    [0, np.sin(tilt), np.cos(tilt)],
                ]
            )
            rotation_matrix = np.dot(R_pan, R_tilt)
            rotated_vertices = np.dot(vertices, rotation_matrix.T)

            faces = [
                [rotated_vertices[0], rotated_vertices[1], rotated_vertices[4]],
                [rotated_vertices[1], rotated_vertices[2], rotated_vertices[4]],
                [rotated_vertices[2], rotated_vertices[3], rotated_vertices[4]],
                [rotated_vertices[3], rotated_vertices[0], rotated_vertices[4]],
                [
                    rotated_vertices[0],
                    rotated_vertices[1],
                    rotated_vertices[2],
                    rotated_vertices[3],
                ],
            ]
            self.ax.add_collection3d(
                Poly3DCollection(
                    faces,
                    facecolors="black",
                    linewidths=1,
                    edgecolors="black",
                    alpha=0.5,
                )
            )
        if tracklets is not None:
            for i in range(0, len(tracklets)):
                if tracklets[i].status.name == "TRACKED":
                    # Y軸とZ軸は表示の観点から反転
                    x = tracklets[i].spatialCoordinates.x / 1000
                    y = tracklets[i].spatialCoordinates.z / 1000
                    z = tracklets[i].spatialCoordinates.y / 1000
                    color = [
                        idColors[tracklets[i].id][2] / 255,
                        idColors[tracklets[i].id][1] / 255,
                        idColors[tracklets[i].id][0] / 255,
                    ]
                    self.ax.scatter(x, y, z, color=color)
        plt.pause(0.001)
        plt.draw()


@dataclass
class PosLog:
    """保存する位置情報"""

    time: int
    x: float
    y: float
    z: float


class LogJson(TypedDict):
    """保存するJSONの型"""

    id: int
    name: str
    time: float
    pos: List[Tuple[float, float, float]]


class OrbitData(object):
    """trackletsの位置情報を保存する形式を定義したクラス"""

    def __init__(self, name: str, id: int, pos_log: PosLog):
        """クラスの初期化メソッド。"""
        self.name: str = name
        self.id: int = id
        self.pos_log: List[PosLog] = []
        self.tmp_pos_log: List[PosLog] = [pos_log]


class OrbitDataList(object):
    """trackletsの移動履歴を保存するためのクラス"""

    def __init__(self, labels: List[str], log_path: Optional[str] = None):
        """クラスの初期化メソッド。

        Args:
            labels (List[str]): trackletsのlabel
            log_path (Optional[str]): logを保存するディレクトリパス
        """
        self.LOGGING_INTEREVAL = 0.5
        # LOGGING_INTEREVALの間にこの回数以上存在しなければ誤認識と判定
        self.AVAILABLE_TIME_THRESHOLD = 3
        self.start_time = time.time()
        self.last_update_time = 0.0
        self.data: List[OrbitData] = []
        self.labels: List[str] = labels
        self.file_name = None
        if log_path is not None:
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            current_time = datetime.now()
            self.file_name = (
                log_path + f"/data_{current_time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            init_json = {}
            init_json["start_time"] = f"{current_time.strftime('%Y/%m/%d %H:%M:%S')}"
            init_json["interval"] = self.LOGGING_INTEREVAL
            init_json["logs"] = []
            with open(self.file_name, mode="wt", encoding="utf-8") as f:
                json.dump(init_json, f, ensure_ascii=False, indent=2)

    def __del__(self):
        """OrbitDataListオブジェクトが削除される際に呼び出されるデストラクタ。"""
        for data in self.data:
            self.save_pos_log(data)

    def get_cur_time(self) -> float:
        """現在の時間を取得する

        Returns:
            float: 現在の時間
        """
        return time.time() - self.start_time

    def get_orbit_from_id(self, id: int) -> OrbitData:
        """IDからOrbitDataを取得する

        Args:
            id (int): trackletのid

        Returns:
            OrbitData: IDに対応するOrbitData

        """
        for data in self.data:
            if data.id == id:
                return data
        return None

    def add_new_data(self, tracklet: Any) -> None:
        """trackletから新しいidのOrbitDataを作成

        Args:
            tracklet (Any): 保存するtracklet

        """
        pos_data = PosLog(
            time=self.get_cur_time(),
            x=tracklet.spatialCoordinates.x,
            y=tracklet.spatialCoordinates.y,
            z=tracklet.spatialCoordinates.z,
        )
        self.data.append(
            OrbitData(
                name=self.labels[tracklet.label], id=tracklet.id, pos_log=pos_data
            )
        )

    def add_track_data(self, tracklet: Any, pos_list: List[PosLog]) -> None:
        """trackletの位置情報をpos_listに追加
        Args:
            tracklet (Any): 追加するtracklet
            pos_list (List[PosLog]): 追加先のlist

        """
        pos_data = PosLog(
            time=self.get_cur_time(),
            x=tracklet.spatialCoordinates.x,
            y=tracklet.spatialCoordinates.y,
            z=tracklet.spatialCoordinates.z,
        )
        pos_list.append(pos_data)

    def update_orbit_data(self, tracklets: List[Any]) -> None:
        """trackletsの位置情報で軌道情報を更新

        Args:
            tracklets (List[Any]): tracklets

        """
        if tracklets is None:
            return
        for tracklet in tracklets:
            if tracklet.status.name == "TRACKED":
                new_data = True
                for data in self.data:
                    if tracklet.id == data.id:
                        self.add_track_data(tracklet, data.tmp_pos_log)
                        new_data = False
                if new_data:
                    self.add_new_data(tracklet)
        self.fix_pos_log()
        self.remove_old_data(tracklets)

    def weighted_average_position(
        self, pos_logs: List[PosLog], target_time: float
    ) -> PosLog:
        """指定された時間での位置を、重み付け平均から推測する。

        Args:
            pos_logs (List[PosLog]): 時間と位置のログのリスト。
            target_time (float): 推測したい時間。

        Returns:
            PosLog: 推測された位置。
        """
        total_weight = 0.0
        weighted_sum_x = 0.0
        weighted_sum_y = 0.0
        weighted_sum_z = 0.0
        for log in pos_logs:
            # 時間差に基づいた重みを計算（時間差が小さいほど重みが大きくなる）
            weight = 1 / (abs(log.time - target_time) + 1e-9)  # ゼロ除算を避けるための微小値
            total_weight += weight
            weighted_sum_x += log.x * weight
            weighted_sum_y += log.y * weight
            weighted_sum_z += log.z * weight
        # 重み付け平均を計算
        avg_x = weighted_sum_x / total_weight
        avg_y = weighted_sum_y / total_weight
        avg_z = weighted_sum_z / total_weight

        return PosLog(time=target_time, x=avg_x, y=avg_y, z=avg_z)

    def fix_pos_log(self) -> None:
        """tmp_pos_logに保存された位置情報をLOGGING_INTEREVALで時間平均してpos_logに保存"""
        cur_time = self.get_cur_time()
        while True:
            next_time = self.last_update_time + self.LOGGING_INTEREVAL * 3 / 2
            if cur_time - next_time < 0:
                break
            for data in self.data:
                tmp_list: List[PosLog] = []
                while True:
                    if len(data.tmp_pos_log) > 0:
                        if data.tmp_pos_log[0].time < next_time:
                            tmp_list.append(data.tmp_pos_log.pop(0))
                        else:
                            break
                    else:
                        break
                if len(tmp_list) >= self.AVAILABLE_TIME_THRESHOLD:
                    data.pos_log.append(
                        self.weighted_average_position(
                            tmp_list, self.last_update_time + self.LOGGING_INTEREVAL
                        )
                    )
            self.last_update_time += self.LOGGING_INTEREVAL

    def remove_old_data(self, tracklets: List[Any]) -> None:
        """trackletsから消えたデータをpos_logから削除して保存
        Args:
            tracklets (List[Any]): tracklets

        """
        new_data = []
        for data in self.data:
            is_tracking = False
            for tracklet in tracklets:
                if tracklet.id == data.id:
                    is_tracking = True
                    break
            if not is_tracking:
                if self.file_name is not None:
                    self.save_pos_log(data)
            else:
                new_data.append(data)
        self.data = new_data

    def save_pos_log(self, data: OrbitData) -> None:
        """OrbitDataのpos_logをファイルに保存

        Args:
            data (OrbitData): 保存するOrbitData

        """
        if len(data.pos_log) == 0:
            return
        new_data: LogJson = {
            "id": data.id,
            "name": data.name,
            "time": data.pos_log[0].time,
            "pos": [
                (
                    round(pos.x / 1000.0, 3),
                    round(pos.y / 1000.0, 3),
                    round(pos.z / 1000.0, 3),
                )
                for pos in data.pos_log
            ],
        }
        json_open = open(self.file_name, "r")
        log_file = json.load(json_open)
        log_file["logs"].append(new_data)
        with open(self.file_name, mode="wt", encoding="utf-8") as f:
            json.dump(log_file, f, ensure_ascii=False, indent=2)


class OrbitPlayer(OakdTrackingYolo):
    """軌道ログを再生するためのクラス"""

    def __init__(
        self,
        log_path: str,
        speed: float = 1.0,
        fov: float = 73.0,
        max_z: float = 15000,
    ) -> None:
        """クラスの初期化メソッド。
        Args:
            log_path (str): ログファイルのパス
            speed (float, optional): 再生速度 (1.0 が等速). デフォルトは 1.0.
            fov (float, optional): 俯瞰マップ上に描画されるOAK-Dの視野角 (度). デフォルトは 73.0 度.
            max_z (float, optional): 俯瞰マップの最大Z座標値. デフォルトは 15000.

        Raises:
            FileNotFoundError: ログファイルが存在しない場合に発生.
        """
        self.log = None
        try:
            json_open = open(log_path, "r")
            self.log = json.load(json_open)
        except FileNotFoundError:
            print(f"Error: The file {log_path} does not exist.")
            return
        self.fov = fov
        self.speed = speed
        self.max_z = max_z
        self.interval = float(self.log["interval"])
        self.end_time = self.get_end_time(self.log["logs"])
        self.bird_eye_frame = self.create_bird_frame()
        self.datetime = datetime.strptime(self.log["start_time"], "%Y/%m/%d %H:%M:%S")

    def get_end_time(self, logs: List[Any]) -> float:
        """
        ログファイル内のデータの終了時刻を取得する。

        Args:
            logs (List[Any]): 軌道の時系列データ

        Returns:
            float: ログデータの終了時刻 [s]
        """
        end_time = 0
        for data in logs:
            cur_end_time = float(data["time"]) + self.interval * len(data["pos"])
            if cur_end_time > end_time:
                end_time = cur_end_time
        return end_time

    def get_cur_index(self, now: float, data: Dict[str, Any]) -> int:
        """現在時刻からインデックス番号を取得する。

        Args:
            now (float): 現在の時刻 (秒単位)
            data (Dict[str, Any]): 軌道の時系列データ

        Returns:
            int: データの現在インデックス。
                -2 は開始時刻前、-1 は時刻が最終インデックスを超えている、
                それ以外の正の整数はインデックスを表す。
        """
        spend_time = now - float(data["time"])
        if spend_time < 0:
            return -2
        if spend_time >= self.interval * len(data["pos"]):
            return -1
        return int(spend_time / self.interval)

    def play_log(self) -> None:
        """ログデータに基づいて、設定された時間間隔で軌跡を描画する。"""
        plotting_list = []
        now = 0.0
        while now <= self.end_time:
            # 記録開始時間に到達したらplotting_listに追加
            for data in self.log["logs"]:
                if data["time"] == now:
                    plotting_list.append(copy.deepcopy(data))
            updated_plotting_list = []
            for plotting_data in plotting_list:
                if self.get_cur_index(now, plotting_data) >= 0:
                    updated_plotting_list.append(plotting_data)
            plotting_list = copy.deepcopy(updated_plotting_list)
            self.draw_bird_frame(now, plotting_list)
            now += self.interval
            self.datetime += timedelta(seconds=self.interval)
            time.sleep(self.interval / self.speed)

    def draw_bird_frame(self, now: float, log_list: List[Any]) -> None:
        """俯瞰フレームにログを描画する。

        Args:
            datas (List[Any]): 描画中の軌道ログのリスト。

        """
        birds = self.bird_eye_frame.copy()
        cv2.putText(
            birds,
            self.datetime.strftime("%Y/%m/%d %H:%M:%S.%f")[:-4],
            (0, birds.shape[1] - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.3,
            (255, 255, 255),
        )
        if log_list is not None:
            for data in log_list:
                cur_index = self.get_cur_index(now, data)
                point_y = self.pos_to_point_y(
                    birds.shape[0], data["pos"][cur_index][2] * 1000
                )
                point_x = self.pos_to_point_x(
                    birds.shape[1], data["pos"][cur_index][0] * 1000
                )
                cv2.circle(
                    birds,
                    (point_x, point_y),
                    2,
                    idColors[data["id"]],
                    thickness=5,
                    lineType=8,
                    shift=0,
                )
                cv2.putText(
                    birds,
                    data["name"],
                    (point_x - 30, point_y - 10),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.5,
                    idColors[data["id"]],
                )
                prev_point = None
                for i in range(0, cur_index):
                    cur_point = (
                        self.pos_to_point_x(birds.shape[1], data["pos"][i][0] * 1000),
                        self.pos_to_point_y(birds.shape[0], data["pos"][i][2] * 1000),
                    )
                    cv2.circle(
                        birds,
                        cur_point,
                        2,
                        idColors[data["id"]],
                        thickness=2,
                        lineType=8,
                        shift=0,
                    )
                    if prev_point is not None:
                        cv2.line(
                            birds,
                            prev_point,
                            cur_point,
                            idColors[data["id"]],
                            thickness=1,
                        )
                    prev_point = (
                        self.pos_to_point_x(birds.shape[1], data["pos"][i][0] * 1000),
                        self.pos_to_point_y(birds.shape[0], data["pos"][i][2] * 1000),
                    )
                if prev_point is not None:
                    cv2.line(
                        birds,
                        prev_point,
                        (point_x, point_y),
                        idColors[data["id"]],
                        2,
                    )
        cv2.imshow("birds", birds)
        cv2.waitKey(1)