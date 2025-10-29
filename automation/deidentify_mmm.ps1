<#
.SYNOPSIS
    End-to-end PowerShell automation for multi-person facial de-identification of Desktop\\mmm.mp4.
.DESCRIPTION
    - Installs required runtime dependencies (Git, Python 3.10+, FFmpeg) via winget when missing.
    - Clones hanweikung/face_anon_simple into %LOCALAPPDATA%\\face-deid.
    - Sets up a Python virtual environment and installs required packages (insightface, onnxruntime, opencv-python, etc.).
    - Downloads InsightFace's inswapper_128 model and a pool of synthetic donor faces.
    - Uses FFmpeg to demux audio, runs automation/multi_face_anon.py to anonymize faces while preserving hair/ears,
      and muxes the original audio back into the final MP4.
    - Supports both GPU (CUDA/DirectML) and CPU execution automatically.
#>
[CmdletBinding()]
param(
    [string]$InputVideo = (Join-Path ([System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::Desktop)) 'mmm.mp4'),
    [string]$OutputDirectory = (Join-Path ([System.Environment]::GetFolderPath([System.Environment+SpecialFolder]::MyVideos)) 'Deidentified'),
    [switch]$ForceCPU
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Write-Section([string]$Message) {
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Assert-FileExists([string]$Path, [string]$Message) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Message ($Path)"
    }
}

function Ensure-Winget {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw 'winget is required. Install Windows App Installer from the Microsoft Store and re-run.'
    }
}

function Ensure-Package {
    param(
        [string]$Command,
        [string]$PackageId,
        [string]$DisplayName
    )
    if (Get-Command $Command -ErrorAction SilentlyContinue) {
        Write-Host "✔ $DisplayName already installed" -ForegroundColor Green
        return
    }
    Write-Host "Installing $DisplayName via winget ..." -ForegroundColor Yellow
    winget install --id $PackageId -e --accept-package-agreements --accept-source-agreements | Out-Null
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Failed to install $DisplayName"
    }
}

function Invoke-GitCloneOrUpdate {
    param(
        [string]$RepoUrl,
        [string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Destination)) {
        git clone $RepoUrl $Destination | Out-Null
        return
    }
    Push-Location $Destination
    try {
        git pull | Out-Null
    } finally {
        Pop-Location
    }
}

function New-VenvIfMissing {
    param(
        [string]$PythonExe,
        [string]$VenvPath
    )
    if (-not (Test-Path -LiteralPath $VenvPath)) {
        & $PythonExe -m venv $VenvPath
    }
}

function Install-PythonPackages {
    param(
        [string]$PipExe,
        [string[]]$Packages
    )
    & $PipExe install --upgrade pip setuptools wheel | Out-Null
    & $PipExe install $Packages | Out-Null
}

function Get-ActivationScript {
    param(
        [string]$VenvPath
    )
    return Join-Path $VenvPath 'Scripts\\Activate.ps1'
}

function Download-FileIfMissing {
    param(
        [string]$Uri,
        [string]$Destination
    )
    if (Test-Path -LiteralPath $Destination) {
        return
    }
    Write-Host "Downloading $Uri ..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $Uri -OutFile $Destination
}

function Initialize-DonorPool {
    param(
        [string]$DonorDir,
        [int]$Count = 6
    )
    New-Item -ItemType Directory -Force -Path $DonorDir | Out-Null
    for ($i = 0; $i -lt $Count; $i++) {
        $target = Join-Path $DonorDir ("donor_{0:D2}.jpg" -f $i)
        if (Test-Path -LiteralPath $target) {
            continue
        }
        Start-Sleep -Milliseconds 400
        try {
            Invoke-WebRequest -Uri 'https://thispersondoesnotexist.com/image' -OutFile $target -Headers @{ 'User-Agent' = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' }
        } catch {
            Write-Warning "Failed to download donor face $_"
        }
    }
    if (-not (Get-ChildItem -Path $DonorDir -File -Include *.jpg, *.png, *.jpeg)) {
        throw 'Unable to download donor faces. Provide custom donor images in the donors directory and rerun.'
    }
}

function Write-MultiFaceScript {
    param(
        [string]$Destination
    )
    $content = @'
#!/usr/bin/env python3
"""Multi-person face de-identification pipeline built on InsightFace."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except ImportError:
    ort = None

from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

SIMILARITY_THRESHOLD = 0.35
MAX_TRACK_GAP = 60


class IdentityTrack:
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
        self.tracks = {}
        self.next_track_id = 0

    def cleanup(self, frame_index: int) -> None:
        expired = [track_id for track_id, track in self.tracks.items() if frame_index - track.last_seen > MAX_TRACK_GAP]
        for track_id in expired:
            del self.tracks[track_id]

    def match(self, embedding: np.ndarray, frame_index: int, donor_face):
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


def load_donor_faces(app: FaceAnalysis, donor_dir: Path):
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
        donor_faces.append(faces[0])
    if not donor_faces:
        raise RuntimeError(f"Failed to load any donor faces from {donor_dir}")
    return donor_faces


def prepare_swapper(model_dir: Path, ctx_id: int) -> INSwapper:
    model_path = model_dir / "inswapper_128.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Download inswapper_128.onnx and place it inside the models directory."
        )
    return INSwapper(str(model_path), providers=None if ctx_id >= 0 else ['CPUExecutionProvider'])


def anonymize_video(input_path: Path, output_path: Path, donor_dir: Path, model_dir: Path, prefer_gpu: bool = True) -> None:
    ctx_id = resolve_ctx_id(prefer_gpu=prefer_gpu)
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    donor_faces = load_donor_faces(app, donor_dir)
    swapper = prepare_swapper(model_dir, ctx_id)

    video = cv2.VideoCapture(str(input_path))
    if not video.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

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
    parser.add_argument('--input', required=True, type=Path, help='Input MP4 video')
    parser.add_argument('--output', required=True, type=Path, help='Output MP4 video without audio')
    parser.add_argument('--donors', required=True, type=Path, help='Directory containing donor face images')
    parser.add_argument('--models', required=True, type=Path, help='Directory containing InsightFace models')
    parser.add_argument('--cpu', action='store_true', help='Force CPU execution')
    args = parser.parse_args()

    anonymize_video(
        input_path=args.input,
        output_path=args.output,
        donor_dir=args.donors,
        model_dir=args.models,
        prefer_gpu=not args.cpu,
    )


if __name__ == '__main__':
    main()
'@
    Set-Content -Path $Destination -Value $content -Encoding UTF8
}

Ensure-Winget
Ensure-Package -Command git -PackageId 'Git.Git' -DisplayName 'Git'
Ensure-Package -Command python -PackageId 'Python.Python.3.10' -DisplayName 'Python 3.10'
Ensure-Package -Command ffmpeg -PackageId 'Gyan.FFmpeg' -DisplayName 'FFmpeg'

Assert-FileExists -Path $InputVideo -Message 'Input video not found'

$Root = Join-Path $env:LOCALAPPDATA 'face-deid'
$RepoPath = Join-Path $Root 'face_anon_simple'
$VenvPath = Join-Path $RepoPath '.venv'
$ModelsPath = Join-Path $RepoPath 'models'
$DonorPath = Join-Path $RepoPath 'donors'
$IntermediatePath = Join-Path $RepoPath 'work'
$OutputDirectory = (New-Item -ItemType Directory -Force -Path $OutputDirectory).FullName
$OutputVideoNoAudio = Join-Path $IntermediatePath 'video_no_audio.mp4'
$OutputAudio = Join-Path $IntermediatePath 'original_audio.m4a'
$FinalVideo = Join-Path $OutputDirectory 'mmm_deidentified.mp4'

Write-Section 'Fetching face_anon_simple repository'
New-Item -ItemType Directory -Force -Path $Root | Out-Null
Invoke-GitCloneOrUpdate -RepoUrl 'https://github.com/hanweikung/face_anon_simple.git' -Destination $RepoPath

Write-Section 'Adding multi-face anonymization helper script'
$PyScriptPath = Join-Path $RepoPath 'multi_face_anon.py'
Write-MultiFaceScript -Destination $PyScriptPath

Write-Section 'Setting up Python environment'
$PythonExe = (Get-Command python).Source
New-VenvIfMissing -PythonExe $PythonExe -VenvPath $VenvPath
$ActivateScript = Get-ActivationScript -VenvPath $VenvPath
. $ActivateScript
try {
    $PipExe = Join-Path $VenvPath 'Scripts\\pip.exe'
    $RequiredPackages = @(
        'insightface==0.7.3',
        'onnxruntime-gpu; platform_system=="Windows"',
        'onnxruntime; platform_system!="Windows"',
        'opencv-python',
        'numpy'
    )
    Install-PythonPackages -PipExe $PipExe -Packages $RequiredPackages
    # Prime InsightFace model downloads for detection
    python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1, det_size=(640,640))" | Out-Null
finally {
    deactivate
}

Write-Section 'Downloading models and donor faces'
New-Item -ItemType Directory -Force -Path $ModelsPath | Out-Null
Download-FileIfMissing -Uri 'https://huggingface.co/deepinsight/insightface/resolve/main/models/inswapper_128.onnx' -Destination (Join-Path $ModelsPath 'inswapper_128.onnx')
Initialize-DonorPool -DonorDir $DonorPath -Count 6
New-Item -ItemType Directory -Force -Path $IntermediatePath | Out-Null

Write-Section 'Extracting original audio with FFmpeg'
ffmpeg -y -i $InputVideo -vn -acodec copy $OutputAudio | Out-Null

Write-Section 'Running anonymization pipeline'
. $ActivateScript
try {
    $PyScript = Join-Path $RepoPath 'multi_face_anon.py'
    $cpuArgs = if ($ForceCPU.IsPresent) { @('--cpu') } else { @() }
    python $PyScript --input $InputVideo --output $OutputVideoNoAudio --donors $DonorPath --models $ModelsPath @cpuArgs
finally {
    deactivate
}

Write-Section 'Combining anonymized video with original audio'
ffmpeg -y -i $OutputVideoNoAudio -i $OutputAudio -c copy $FinalVideo | Out-Null

Write-Section "Done. Output saved to $FinalVideo"
