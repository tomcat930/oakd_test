#!/usr/bin/env python3
# coding: utf-8

import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import requests
from tqdm import tqdm


def download_file(path: str, link: str) -> None:
    """
    指定されたリンクからファイルをダウンロードする関数。

    Args:
        path (str): ダウンロード先のファイルパス
        link (str): ダウンロード元のリンク

    Raises:
        Exception: ダウンロード中にエラーが発生した場合
    """
    if os.path.exists(path):
        return
    # ディレクトリが存在しない場合は作成
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    try:
        # ファイルをダウンロード
        print(f"{path} doesn't exist. Download from {link}")
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress = tqdm(total=total_size, unit="iB", unit_scale=True)
        with open(path, "wb") as f:
            for data in response.iter_content(chunk_size=block_size):
                if data:
                    f.write(data)
                    progress.update(len(data))
        progress.close()
        print(f"{path} download finished.")
    except Exception:
        print("Download error")


class TextHelper(object):
    """
    フレームに文字列を描画するクラス

    """

    def __init__(self) -> None:
        """クラスのコンストラクタ"""
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def put_text(self, frame: np.ndarray, text: str, coords: Tuple[int, int]) -> None:
        """
        フレームに文字列を描画する。

        Args:
            frame (np.ndarray): 描画対象の画像フレーム。
            text (str): 描画する文字列。
            coords (Tuple[int, int]): 描画開始位置の座標。

        """
        cv2.putText(
            frame, text, coords, self.text_type, 0.8, self.bg_color, 3, self.line_type
        )
        cv2.putText(
            frame, text, coords, self.text_type, 0.8, self.color, 1, self.line_type
        )

    def rectangle(
        self,
        frame: np.ndarray,
        p1: Tuple[float],
        p2: Tuple[float],
        color: Tuple[int, int, int],
    ) -> None:
        """
        フレームに矩形を描画する。

        Args:
            frame (np.ndarray): 描画対象の画像フレーム。
            p1 (Tuple[float]): 矩形の開始座標。
            p2 (Tuple[float]): 矩形の終了座標。
            color: (Tuple[int, int, int]): 矩形描画色

        """
        cv2.rectangle(frame, p1, p2, (0, 0, 0), 4)
        cv2.rectangle(frame, p1, p2, color, 2)


class HostSync(object):
    """各フレームのメッセージを同期するクラス。"""

    def __init__(self, sync_size: int = 4):
        """HostSyncクラスの初期化メソッド。

        Args:
            sync_size (int, optional): 同期するメッセージの数。デフォルトは4。

        """
        self.dict = {}
        self.head_seq = 0
        self.sync_size = sync_size

    def add_msg(self, name: str, msg: Any, seq: Optional[str] = None) -> None:
        """メッセージをDictに追加するメソッド。

        Args:
            name (str): メッセージの名前。
            msg (Any): 追加するメッセージ。
            seq (str, optional): メッセージのシーケンス番号。デフォルトはNone。

        """
        if seq is None:
            seq = str(msg.getSequenceNum())
        if seq not in self.dict:
            self.dict[seq] = {}
        self.dict[seq][name] = msg

    def get_msgs(self) -> Any:
        """同期されたメッセージを取得するメソッド。

        Returns:
            Any: 同期されたメッセージ。

        """
        remove = []
        for name in self.dict:
            remove.append(name)
            if len(self.dict[name]) == self.sync_size:
                ret = self.dict[name]
                for rm in remove:
                    del self.dict[rm]
                return ret
        return None