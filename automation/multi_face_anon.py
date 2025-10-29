#!/usr/bin/env python3
"""Multi-person face de-identification pipeline built on InsightFace.

This script expects a donors directory containing synthetic donor faces.
Each donor face is processed once and re-used to keep per-person identities
stable across a video.  Hair and ears are preserved by restricting the swap to
facial regions only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - fallback if ort missing
    ort = None

from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

SIMILARITY_THRESHOLD = 0.35
MAX_TRACK_GAP = 60  # frames


class IdentityTrack:
    """Represents a face track with a stable donor identity."""

    def __init__(self, track_id: int, donor_face, embedding: np.ndarray, frame_index: int):
        self.track_id = track_id
        self.donor_face = donor_face
        self.embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        self.last_seen = frame_index
        self.hit_count = 1

    def update(self, embedding: np.ndarray, frame_index: int) -> None:
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        self.embedding = (self.embedding * self.hit_count + embedding) / (self.hit_count + 1)
        self.embedding /= np.linalg.norm(self.embedding) + 1e-8
        self.hit_count += 1
        self.last_seen = frame_index


class TrackManager:
    def __init__(self):
        self.tracks: Dict[int, IdentityTrack] = {}
        self.next_track_id = 0

    def cleanup(self, frame_index: int) -> None:
        to_remove = [track_id for track_id, track in self.tracks.items() if frame_index - track.last_seen > MAX_TRACK_GAP]
        for track_id in to_remove:
            del self.tracks[track_id]

    def match(self, embedding: np.ndarray, frame_index: int, donor_face) -> IdentityTrack:
        best_track = None
        best_score = -1.0
        for track in self.tracks.values():
            if frame_index - track.last_seen > MAX_TRACK_GAP:
                continue
            score = float(np.dot(track.embedding, embedding) / (np.linalg.norm(embedding) + 1e-8))
            if score > best_score:
                best_score = score
                best_track = track
        if best_track is None or best_score < SIMILARITY_THRESHOLD:
            track = IdentityTrack(self.next_track_id, donor_face, embedding, frame_index)
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1
            return track
        best_track.update(embedding, frame_index)
        return best_track


def resolve_ctx_id(prefer_gpu: bool = True) -> int:
    if not prefer_gpu:
        return -1
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    if ort is not None:
        try:
            providers = ort.get_available_providers()
            for provider in providers:
                if any(tag in provider for tag in ("CUDA", "Dml", "ROCM", "Tensorrt")):
                    return 0
        except Exception:
            pass
    return -1


def load_donor_faces(app: FaceAnalysis, donor_dir: Path) -> List:
    donor_faces = []
    image_paths = sorted([p for p in donor_dir.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not image_paths:
        raise FileNotFoundError(f"No donor faces found in {donor_dir}")
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        faces = app.get(image)
        if not faces:
            continue
        donor_faces.append((faces[0], image_path.name))
    if not donor_faces:
        raise RuntimeError(f"Failed to load any donor faces from {donor_dir}")
    return [face for face, _ in donor_faces]


def prepare_swapper(model_dir: Path, ctx_id: int) -> INSwapper:
    model_path = model_dir / "inswapper_128.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Please download InsightFace's inswapper_128.onnx and place it in the models directory."
        )
    return INSwapper(str(model_path), providers=None if ctx_id >= 0 else ['CPUExecutionProvider'])


def anonymize_video(
    input_path: Path,
    output_path: Path,
    donor_dir: Path,
    model_dir: Path,
    prefer_gpu: bool = True,
) -> None:
    ctx_id = resolve_ctx_id(prefer_gpu=prefer_gpu)
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    donor_faces = load_donor_faces(app, donor_dir)
    swapper = prepare_swapper(model_dir, ctx_id)
    video = cv2.VideoCapture(str(input_path))
    if not video.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    tracks = TrackManager()
    frame_index = 0
    donor_cycle = 0
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            faces = app.get(frame)
            tracks.cleanup(frame_index)
            if faces:
                for face in faces:
                    embedding = face.normed_embedding.astype(np.float32)
                    donor_face = donor_faces[donor_cycle % len(donor_faces)]
                    track = tracks.match(embedding, frame_index, donor_face)
                    donor_face = track.donor_face
                    frame = swapper.get(frame, face, donor_face, paste_back=True)
                    donor_cycle += 1
            writer.write(frame)
            frame_index += 1
    finally:
        video.release()
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-person face de-identification for MP4 videos")
    parser.add_argument("--input", required=True, type=Path, help="Input MP4 video")
    parser.add_argument("--output", required=True, type=Path, help="Path to anonymized MP4 video (no audio)")
    parser.add_argument("--donors", required=True, type=Path, help="Directory containing donor face images")
    parser.add_argument("--models", required=True, type=Path, help="Directory containing InsightFace models")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
    args = parser.parse_args()

    anonymize_video(
        input_path=args.input,
        output_path=args.output,
        donor_dir=args.donors,
        model_dir=args.models,
        prefer_gpu=not args.cpu,
    )


if __name__ == "__main__":
    main()
