"""PyQt5 interface for interactive speech denoising with a trained U-Net."""

from __future__ import annotations

import sys
import os
import csv
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
MPL_CONFIG_DIR = (PROJECT_ROOT / ".matplotlib").resolve()
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import librosa
import numpy as np
import soundfile as sf
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QHeaderView,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from torch.amp import autocast

try:
    from src.evaluate import compute_pesq, compute_sdr, compute_stoi
    from src.model import UNetConfig, build_model
except ModuleNotFoundError:
    from evaluate import compute_pesq, compute_sdr, compute_stoi
    from model import UNetConfig, build_model


DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gui"
TEMP_AUDIO_DIR = DEFAULT_OUTPUT_DIR / "temp"
LOG_PATH = DEFAULT_OUTPUT_DIR / "gui.log"
TARGET_SAMPLE_RATE = 8000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512
WINDOW = "hann"
INFERENCE_CHUNK_FRAMES = 256
INFERENCE_OVERLAP_FRAMES = 32
MAX_PLOT_POINTS = 20000
MAX_DISPLAY_FRAMES = 1200
MAX_GUI_PESQ_SECONDS = 30.0

DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def excepthook(exc_type: type[BaseException], exc_value: BaseException, exc_traceback: Any) -> None:
    """Log uncaught GUI exceptions before Qt exits."""

    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    traceback.print_exception(exc_type, exc_value, exc_traceback)


sys.excepthook = excepthook


THEME = {
    "window": "#eaf3ff",
    "panel": "#f7fbff",
    "panel_alt": "#ffffff",
    "border": "#bfd7f3",
    "border_strong": "#78aee8",
    "text": "#102033",
    "muted": "#5f7288",
    "primary": "#1769c2",
    "primary_hover": "#0f5aa8",
    "primary_pressed": "#0b4788",
    "accent": "#0891b2",
    "accent_hover": "#047f9f",
    "danger": "#b4232f",
    "danger_hover": "#981b25",
    "plot_bg": "#f8fbff",
    "axis_bg": "#ffffff",
    "grid": "#9bbfe8",
    "noisy": "#2f6fb4",
    "denoised": "#0891b2",
    "clean": "#5b7cfa",
    "warning": "#d97706",
}


def apply_blue_theme(app: QApplication) -> None:
    """Apply a polished blue visual theme to the Qt application."""

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(THEME["window"]))
    palette.setColor(QPalette.WindowText, QColor(THEME["text"]))
    palette.setColor(QPalette.Base, QColor(THEME["panel_alt"]))
    palette.setColor(QPalette.AlternateBase, QColor(THEME["panel"]))
    palette.setColor(QPalette.ToolTipBase, QColor(THEME["primary"]))
    palette.setColor(QPalette.ToolTipText, QColor("#ffffff"))
    palette.setColor(QPalette.Text, QColor(THEME["text"]))
    palette.setColor(QPalette.Button, QColor(THEME["primary"]))
    palette.setColor(QPalette.ButtonText, QColor("#ffffff"))
    palette.setColor(QPalette.Highlight, QColor(THEME["primary"]))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(
        f"""
        QMainWindow, QWidget {{
            background: {THEME["window"]};
            color: {THEME["text"]};
            font-family: Segoe UI, Arial, sans-serif;
            font-size: 10pt;
        }}
        QGroupBox {{
            background: {THEME["panel"]};
            border: 1px solid {THEME["border"]};
            border-radius: 8px;
            margin-top: 14px;
            padding: 14px 12px 12px 12px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: {THEME["primary"]};
            background: {THEME["window"]};
        }}
        QLabel {{
            background: transparent;
            color: {THEME["text"]};
        }}
        QLabel#PathLabel {{
            color: {THEME["muted"]};
            background: {THEME["panel_alt"]};
            border: 1px solid {THEME["border"]};
            border-radius: 6px;
            padding: 7px 9px;
        }}
        QLabel#MetricsText {{
            color: {THEME["text"]};
            background: {THEME["panel_alt"]};
            border: 1px solid {THEME["border"]};
            border-radius: 6px;
            padding: 10px;
            line-height: 130%;
        }}
        QLineEdit {{
            background: {THEME["panel_alt"]};
            border: 1px solid {THEME["border"]};
            border-radius: 6px;
            padding: 8px 9px;
            selection-background-color: {THEME["primary"]};
        }}
        QLineEdit:focus {{
            border: 1px solid {THEME["primary"]};
        }}
        QPushButton {{
            background: {THEME["primary"]};
            color: #ffffff;
            border: 1px solid {THEME["primary"]};
            border-radius: 6px;
            padding: 8px 12px;
            min-height: 18px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background: {THEME["primary_hover"]};
            border-color: {THEME["primary_hover"]};
        }}
        QPushButton:pressed {{
            background: {THEME["primary_pressed"]};
            border-color: {THEME["primary_pressed"]};
        }}
        QPushButton#SecondaryButton {{
            background: {THEME["panel_alt"]};
            color: {THEME["primary"]};
            border: 1px solid {THEME["border_strong"]};
        }}
        QPushButton#SecondaryButton:hover {{
            background: #e5f1ff;
        }}
        QPushButton#AccentButton {{
            background: {THEME["accent"]};
            border-color: {THEME["accent"]};
        }}
        QPushButton#AccentButton:hover {{
            background: {THEME["accent_hover"]};
            border-color: {THEME["accent_hover"]};
        }}
        QPushButton#DangerButton {{
            background: {THEME["danger"]};
            border-color: {THEME["danger"]};
        }}
        QPushButton#DangerButton:hover {{
            background: {THEME["danger_hover"]};
            border-color: {THEME["danger_hover"]};
        }}
        QTabWidget::pane {{
            background: {THEME["panel"]};
            border: 1px solid {THEME["border"]};
            border-radius: 8px;
            top: -1px;
        }}
        QTabBar::tab {{
            background: #d8eaff;
            color: {THEME["primary"]};
            border: 1px solid {THEME["border"]};
            border-bottom: none;
            border-top-left-radius: 7px;
            border-top-right-radius: 7px;
            padding: 8px 16px;
            margin-right: 3px;
            font-weight: 600;
        }}
        QTabBar::tab:selected {{
            background: {THEME["primary"]};
            color: #ffffff;
            border-color: {THEME["primary"]};
        }}
        QTabBar::tab:hover:!selected {{
            background: #c7e0fb;
        }}
        QTableWidget {{
            background: {THEME["panel_alt"]};
            alternate-background-color: #edf6ff;
            border: 1px solid {THEME["border"]};
            border-radius: 6px;
            gridline-color: {THEME["border"]};
            selection-background-color: #d8eaff;
            selection-color: {THEME["text"]};
        }}
        QHeaderView::section {{
            background: {THEME["primary"]};
            color: #ffffff;
            border: none;
            padding: 8px;
            font-weight: 600;
        }}
        QStatusBar {{
            background: {THEME["primary"]};
            color: #ffffff;
            padding: 4px 8px;
        }}
        QMessageBox {{
            background: {THEME["window"]};
        }}
        """
    )


@dataclass
class DenoisingResult:
    """Signals and spectral products generated by one denoising run."""

    noisy_audio: np.ndarray
    denoised_audio: np.ndarray
    sample_rate: int
    noisy_magnitude: np.ndarray
    denoised_magnitude: np.ndarray
    predicted_mask: np.ndarray
    clean_audio: np.ndarray | None = None
    clean_magnitude: np.ndarray | None = None


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a denoising model checkpoint."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model_config = UNetConfig(**checkpoint.get("model_config", {}))
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def resolve_path(path_text: str) -> Path:
    """Resolve absolute or workspace-relative paths used by the GUI."""

    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    workspace_candidate = WORKSPACE_ROOT / path
    if workspace_candidate.exists():
        return workspace_candidate
    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate
    return workspace_candidate


def load_mono_audio(path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load a mono waveform resampled to ``sample_rate``."""

    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32)


def run_denoising(
    noisy_path: Path,
    checkpoint_path: Path,
    clean_path: Path | None = None,
) -> DenoisingResult:
    """Run model inference and reconstruct a denoised waveform."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Starting denoising noisy=%s checkpoint=%s clean=%s device=%s", noisy_path, checkpoint_path, clean_path, device)
    model = load_checkpoint_model(checkpoint_path, device)
    logging.info("Model loaded")
    noisy_audio = load_mono_audio(noisy_path)
    logging.info("Noisy loaded samples=%s duration=%.3fs", len(noisy_audio), len(noisy_audio) / TARGET_SAMPLE_RATE)
    clean_audio = load_mono_audio(clean_path) if clean_path is not None else None
    if clean_audio is not None:
        logging.info("Clean loaded samples=%s duration=%.3fs", len(clean_audio), len(clean_audio) / TARGET_SAMPLE_RATE)
    clean_magnitude = None
    if clean_audio is not None:
        clean_stft = librosa.stft(
            clean_audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=WINDOW,
        )
        clean_magnitude = np.abs(clean_stft).astype(np.float32)
        logging.info("Clean STFT shape=%s", clean_magnitude.shape)

    noisy_stft = librosa.stft(
        noisy_audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
    )
    noisy_magnitude = np.abs(noisy_stft).astype(np.float32)
    noisy_phase = np.angle(noisy_stft).astype(np.float32)
    noisy_log_magnitude = np.log1p(noisy_magnitude).astype(np.float32)
    logging.info("Noisy STFT shape=%s", noisy_magnitude.shape)

    predicted_mask = predict_mask_in_chunks(
        model=model,
        noisy_log_magnitude=noisy_log_magnitude,
        device=device,
        chunk_frames=INFERENCE_CHUNK_FRAMES,
        overlap_frames=INFERENCE_OVERLAP_FRAMES,
    )
    logging.info("Mask predicted shape=%s min=%.6f max=%.6f", predicted_mask.shape, float(predicted_mask.min()), float(predicted_mask.max()))

    denoised_magnitude = predicted_mask * noisy_magnitude
    denoised_stft = denoised_magnitude * np.exp(1j * noisy_phase)
    denoised_audio = librosa.istft(
        denoised_stft,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        length=len(noisy_audio),
    ).astype(np.float32)
    logging.info("ISTFT complete samples=%s", len(denoised_audio))

    return DenoisingResult(
        noisy_audio=noisy_audio,
        denoised_audio=denoised_audio,
        sample_rate=TARGET_SAMPLE_RATE,
        noisy_magnitude=noisy_magnitude,
        denoised_magnitude=denoised_magnitude,
        predicted_mask=predicted_mask,
        clean_audio=clean_audio,
        clean_magnitude=clean_magnitude,
    )


def predict_mask_in_chunks(
    model: torch.nn.Module,
    noisy_log_magnitude: np.ndarray,
    device: torch.device,
    chunk_frames: int = INFERENCE_CHUNK_FRAMES,
    overlap_frames: int = INFERENCE_OVERLAP_FRAMES,
) -> np.ndarray:
    """Predict a full mask using overlapped time chunks to avoid GPU OOM."""

    total_frames = noisy_log_magnitude.shape[1]
    if total_frames <= chunk_frames:
        return predict_mask_chunk(model, noisy_log_magnitude, device)[:, :total_frames]

    if overlap_frames >= chunk_frames:
        raise ValueError("overlap_frames must be smaller than chunk_frames")

    step = chunk_frames - overlap_frames
    accumulator = np.zeros_like(noisy_log_magnitude, dtype=np.float32)
    weights = np.zeros_like(noisy_log_magnitude, dtype=np.float32)
    base_weight = np.hanning(chunk_frames).astype(np.float32)
    if not np.any(base_weight):
        base_weight = np.ones(chunk_frames, dtype=np.float32)
    base_weight = np.maximum(base_weight, 1e-3)

    starts = list(range(0, total_frames, step))
    if starts[-1] + chunk_frames < total_frames:
        starts.append(total_frames - chunk_frames)

    for start in starts:
        end = min(start + chunk_frames, total_frames)
        actual_frames = end - start
        chunk = noisy_log_magnitude[:, start:end]
        if actual_frames < chunk_frames:
            chunk = np.pad(chunk, ((0, 0), (0, chunk_frames - actual_frames)), mode="edge")
        prediction = predict_mask_chunk(model, chunk, device)[:, :actual_frames]
        weight = base_weight[:actual_frames][None, :]
        accumulator[:, start:end] += prediction * weight
        weights[:, start:end] += weight

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return accumulator / np.maximum(weights, 1e-8)


def predict_mask_chunk(model: torch.nn.Module, chunk: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict a mask for one spectrogram chunk."""

    tensor = torch.from_numpy(chunk[None, None, :, :].astype(np.float32)).to(device)
    with torch.no_grad(), autocast(device_type=device.type, enabled=device.type == "cuda"):
        prediction = model(tensor)
    return prediction.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def align_audio(*signals: np.ndarray) -> tuple[np.ndarray, ...]:
    """Trim waveforms to the same minimum length."""

    min_length = min(len(signal) for signal in signals)
    return tuple(np.asarray(signal[:min_length], dtype=np.float32) for signal in signals)


def align_magnitudes(*magnitudes: np.ndarray) -> tuple[np.ndarray, ...]:
    """Trim spectrogram magnitudes to a shared time-frequency shape."""

    min_freq = min(magnitude.shape[0] for magnitude in magnitudes)
    min_time = min(magnitude.shape[1] for magnitude in magnitudes)
    return tuple(np.asarray(magnitude[:min_freq, :min_time], dtype=np.float32) for magnitude in magnitudes)


def rms(signal: np.ndarray) -> float:
    """Return root mean square amplitude."""

    return float(np.sqrt(np.mean(np.asarray(signal, dtype=np.float32) ** 2)))


def compute_comparison_metrics(result: DenoisingResult) -> dict[str, tuple[float | None, float | None, float | None]]:
    """Compute before/after metrics against a clean reference."""

    if result.clean_audio is None:
        return {}
    clean, noisy, denoised = align_audio(result.clean_audio, result.noisy_audio, result.denoised_audio)
    input_snr = compute_sdr(clean, noisy)
    output_snr = compute_sdr(clean, denoised)
    input_stoi = compute_stoi(result.sample_rate, clean, noisy)
    output_stoi = compute_stoi(result.sample_rate, clean, denoised)
    duration = len(clean) / result.sample_rate
    if duration <= MAX_GUI_PESQ_SECONDS:
        input_pesq = compute_pesq(result.sample_rate, clean, noisy)
        output_pesq = compute_pesq(result.sample_rate, clean, denoised)
    else:
        logging.info("Skipping GUI PESQ for %.3fs audio; limit is %.3fs", duration, MAX_GUI_PESQ_SECONDS)
        input_pesq = None
        output_pesq = None
    noisy_residual_rms = rms(noisy - clean)
    denoised_residual_rms = rms(denoised - clean)
    metrics = {
        "SNR/SDR (dB)": (input_snr, output_snr, output_snr - input_snr),
        "STOI": (
            input_stoi,
            output_stoi,
            None if input_stoi is None or output_stoi is None else output_stoi - input_stoi,
        ),
        "PESQ": (
            input_pesq,
            output_pesq,
            None if input_pesq is None or output_pesq is None else output_pesq - input_pesq,
        ),
        "Residual RMS": (
            noisy_residual_rms,
            denoised_residual_rms,
            noisy_residual_rms - denoised_residual_rms,
        ),
    }
    return metrics


def format_metric_value(value: float | None, metric_name: str) -> str:
    """Format a metric value for display in the GUI table."""

    if value is None:
        return "N/A"
    if "SNR" in metric_name:
        return f"{value:.3f} dB"
    if metric_name == "Residual RMS":
        return f"{value:.6f}"
    return f"{value:.4f}"


def decimate_for_plot(time: np.ndarray, signal: np.ndarray, max_points: int = MAX_PLOT_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Downsample waveform data for responsive plotting."""

    if len(signal) <= max_points:
        return time, signal
    step = int(np.ceil(len(signal) / max_points))
    return time[::step], signal[::step]


def limit_spectrogram_frames(spec: np.ndarray, max_frames: int = MAX_DISPLAY_FRAMES) -> np.ndarray:
    """Reduce displayed spectrogram frames for responsive plotting."""

    if spec.shape[1] <= max_frames:
        return spec
    step = int(np.ceil(spec.shape[1] / max_frames))
    return spec[:, ::step]


class PlotCanvas(FigureCanvas):
    """Matplotlib canvas embedded in the Qt window."""

    def __init__(self) -> None:
        """Create a blank figure canvas."""

        self.figure = Figure(figsize=(8, 5), tight_layout=True, facecolor=THEME["plot_bg"])
        super().__init__(self.figure)
        self.setStyleSheet(f"background: {THEME['plot_bg']}; border-radius: 8px;")

    def clear(self) -> None:
        """Clear the figure."""

        self.figure.clear()
        self.figure.set_facecolor(THEME["plot_bg"])
        self.draw()

    def _reset_figure(self) -> None:
        """Clear the figure and restore the themed background."""

        self.figure.clear()
        self.figure.set_facecolor(THEME["plot_bg"])

    @staticmethod
    def _style_axis(ax: Any) -> None:
        """Apply the shared visual style to a Matplotlib axis."""

        ax.set_facecolor(THEME["axis_bg"])
        ax.tick_params(colors=THEME["muted"], labelsize=8)
        ax.xaxis.label.set_color(THEME["muted"])
        ax.yaxis.label.set_color(THEME["muted"])
        ax.title.set_color(THEME["primary"])
        for spine in ax.spines.values():
            spine.set_color(THEME["border"])
        ax.grid(True, color=THEME["grid"], alpha=0.22, linewidth=0.7)

    def plot_waveforms(self, result: DenoisingResult) -> None:
        """Plot noisy, denoised, and optional clean waveforms."""

        self._reset_figure()
        subplot_count = 3 if result.clean_audio is not None else 2
        axes = self.figure.subplots(subplot_count, 1, sharex=True)
        if subplot_count == 1:
            axes = [axes]

        t_noisy = np.arange(len(result.noisy_audio)) / result.sample_rate
        plot_time, plot_signal = decimate_for_plot(t_noisy, result.noisy_audio)
        axes[0].plot(plot_time, plot_signal, color=THEME["noisy"], linewidth=0.9)
        axes[0].set_title("Noisy waveform")
        axes[0].set_ylabel("Amplitude")
        self._style_axis(axes[0])

        t_denoised = np.arange(len(result.denoised_audio)) / result.sample_rate
        plot_time, plot_signal = decimate_for_plot(t_denoised, result.denoised_audio)
        axes[1].plot(plot_time, plot_signal, color=THEME["denoised"], linewidth=0.9)
        axes[1].set_title("Denoised waveform")
        axes[1].set_ylabel("Amplitude")
        self._style_axis(axes[1])

        if result.clean_audio is not None:
            clean = result.clean_audio[: len(result.noisy_audio)]
            t_clean = np.arange(len(clean)) / result.sample_rate
            plot_time, plot_signal = decimate_for_plot(t_clean, clean)
            axes[2].plot(plot_time, plot_signal, color=THEME["clean"], linewidth=0.9)
            axes[2].set_title("Clean reference waveform")
            axes[2].set_ylabel("Amplitude")
            self._style_axis(axes[2])

        axes[-1].set_xlabel("Time (s)")
        self.draw()

    def plot_spectrograms(self, result: DenoisingResult) -> None:
        """Plot noisy and denoised log-magnitude spectrograms."""

        self._reset_figure()
        specs = [
            ("Noisy log magnitude", limit_spectrogram_frames(np.log1p(result.noisy_magnitude))),
            ("Denoised log magnitude", limit_spectrogram_frames(np.log1p(result.denoised_magnitude))),
        ]
        if result.clean_magnitude is not None:
            specs.append(("Clean log magnitude", limit_spectrogram_frames(np.log1p(result.clean_magnitude))))
        axes = self.figure.subplots(1, len(specs))
        axes = np.atleast_1d(axes)
        for ax, (title, spec) in zip(axes, specs):
            image = ax.imshow(spec, origin="lower", aspect="auto", interpolation="nearest", cmap="Blues")
            ax.set_title(title)
            ax.set_xlabel("Frame")
            ax.set_ylabel("Frequency bin")
            self._style_axis(ax)
            self.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        self.draw()

    def plot_mask(self, result: DenoisingResult) -> None:
        """Plot the predicted denoising mask."""

        self._reset_figure()
        ax = self.figure.add_subplot(111)
        image = ax.imshow(
            limit_spectrogram_frames(result.predicted_mask),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title("Predicted mask")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Frequency bin")
        self._style_axis(ax)
        self.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        self.draw()

    def plot_comparison_errors(self, result: DenoisingResult) -> None:
        """Plot waveform residuals and magnitude errors against clean reference."""

        self._reset_figure()
        if result.clean_audio is None or result.clean_magnitude is None:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Load a clean reference and run comparison.",
                ha="center",
                va="center",
                color=THEME["muted"],
            )
            ax.set_facecolor(THEME["axis_bg"])
            ax.axis("off")
            self.draw()
            return

        clean, noisy, denoised = align_audio(result.clean_audio, result.noisy_audio, result.denoised_audio)
        noisy_error = noisy - clean
        denoised_error = denoised - clean
        noisy_mag, denoised_mag, clean_mag = align_magnitudes(
            result.noisy_magnitude,
            result.denoised_magnitude,
            result.clean_magnitude,
        )
        noisy_mag_error = np.log1p(np.abs(noisy_mag - clean_mag))
        denoised_mag_error = np.log1p(np.abs(denoised_mag - clean_mag))

        axes = self.figure.subplots(2, 2)
        time = np.arange(len(clean)) / result.sample_rate
        plot_time, plot_signal = decimate_for_plot(time, noisy_error)
        axes[0, 0].plot(plot_time, plot_signal, color=THEME["noisy"], linewidth=0.8)
        axes[0, 0].set_title("Noisy residual (noisy - clean)")
        axes[0, 0].set_ylabel("Amplitude")
        self._style_axis(axes[0, 0])
        plot_time, plot_signal = decimate_for_plot(time, denoised_error)
        axes[0, 1].plot(plot_time, plot_signal, color=THEME["denoised"], linewidth=0.8)
        axes[0, 1].set_title("Denoised residual (denoised - clean)")
        self._style_axis(axes[0, 1])
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 1].set_xlabel("Time (s)")

        image0 = axes[1, 0].imshow(
            limit_spectrogram_frames(noisy_mag_error),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="Blues",
        )
        axes[1, 0].set_title("Noisy magnitude error")
        axes[1, 0].set_xlabel("Frame")
        axes[1, 0].set_ylabel("Frequency bin")
        self._style_axis(axes[1, 0])
        self.figure.colorbar(image0, ax=axes[1, 0], fraction=0.046, pad=0.04)

        image1 = axes[1, 1].imshow(
            limit_spectrogram_frames(denoised_mag_error),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="Blues",
        )
        axes[1, 1].set_title("Denoised magnitude error")
        axes[1, 1].set_xlabel("Frame")
        self._style_axis(axes[1, 1])
        self.figure.colorbar(image1, ax=axes[1, 1], fraction=0.046, pad=0.04)
        self.draw()


class DenoisingWindow(QMainWindow):
    """Main PyQt5 window for denoising and visualization."""

    def __init__(self) -> None:
        """Initialize the GUI."""

        super().__init__()
        self.setWindowTitle("Speech Denoising U-Net")
        self.resize(1180, 760)
        self.noisy_path: Path | None = None
        self.clean_path: Path | None = None
        self.result: DenoisingResult | None = None
        self.sound_effect = QSoundEffect(self)
        self.sound_effect.setVolume(0.9)

        self.checkpoint_edit = QLineEdit(str(DEFAULT_CHECKPOINT))
        self.noisy_label = QLabel("No noisy wav selected")
        self.noisy_label.setObjectName("PathLabel")
        self.clean_label = QLabel("No clean reference selected")
        self.clean_label.setObjectName("PathLabel")
        self.metrics_label = QLabel("Metrics will appear here after processing.")
        self.metrics_label.setObjectName("MetricsText")
        self.metrics_label.setAlignment(Qt.AlignTop)
        self.metrics_label.setWordWrap(True)
        self.metrics_table = QTableWidget(0, 4)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Noisy vs clean", "Denoised vs clean", "Improvement"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setShowGrid(False)

        self.wave_canvas = PlotCanvas()
        self.spec_canvas = PlotCanvas()
        self.mask_canvas = PlotCanvas()
        self.comparison_canvas = PlotCanvas()
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.wave_canvas, "Waveforms")
        self.tabs.addTab(self.spec_canvas, "Spectrograms")
        self.tabs.addTab(self.mask_canvas, "Mask")
        self.tabs.addTab(self.comparison_canvas, "Comparison")

        self.setCentralWidget(self._build_layout())
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _build_layout(self) -> QWidget:
        """Build the main layout."""

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(12)

        controls = QGroupBox("Controls")
        grid = QGridLayout(controls)
        grid.setContentsMargins(12, 14, 12, 12)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(9)
        checkpoint_button = QPushButton("Browse checkpoint")
        checkpoint_button.setObjectName("SecondaryButton")
        checkpoint_button.clicked.connect(self.choose_checkpoint)
        load_noisy_button = QPushButton("Load noisy wav")
        load_noisy_button.setObjectName("SecondaryButton")
        load_noisy_button.clicked.connect(self.choose_noisy)
        load_clean_button = QPushButton("Load clean reference")
        load_clean_button.setObjectName("SecondaryButton")
        load_clean_button.clicked.connect(self.choose_clean)
        clear_clean_button = QPushButton("Clear clean")
        clear_clean_button.setObjectName("SecondaryButton")
        clear_clean_button.clicked.connect(self.clear_clean)
        process_button = QPushButton("Process")
        process_button.setObjectName("AccentButton")
        process_button.clicked.connect(self.process_audio)
        compare_button = QPushButton("Compare with clean")
        compare_button.setObjectName("AccentButton")
        compare_button.clicked.connect(self.process_comparison)
        save_button = QPushButton("Save denoised wav")
        save_button.setObjectName("SecondaryButton")
        save_button.clicked.connect(self.save_denoised)
        export_button = QPushButton("Export comparison CSV")
        export_button.setObjectName("SecondaryButton")
        export_button.clicked.connect(self.export_comparison_csv)
        play_noisy_button = QPushButton("Play noisy")
        play_noisy_button.setObjectName("SecondaryButton")
        play_noisy_button.clicked.connect(lambda: self.play_audio("noisy"))
        play_denoised_button = QPushButton("Play denoised")
        play_denoised_button.setObjectName("SecondaryButton")
        play_denoised_button.clicked.connect(lambda: self.play_audio("denoised"))
        play_clean_button = QPushButton("Play clean")
        play_clean_button.setObjectName("SecondaryButton")
        play_clean_button.clicked.connect(lambda: self.play_audio("clean"))
        stop_button = QPushButton("Stop")
        stop_button.setObjectName("DangerButton")
        stop_button.clicked.connect(self.sound_effect.stop)

        grid.addWidget(QLabel("Checkpoint"), 0, 0)
        grid.addWidget(self.checkpoint_edit, 0, 1, 1, 4)
        grid.addWidget(checkpoint_button, 0, 5)
        grid.addWidget(load_noisy_button, 1, 0)
        grid.addWidget(self.noisy_label, 1, 1, 1, 5)
        grid.addWidget(load_clean_button, 2, 0)
        grid.addWidget(self.clean_label, 2, 1, 1, 4)
        grid.addWidget(clear_clean_button, 2, 5)
        grid.addWidget(process_button, 3, 0)
        grid.addWidget(compare_button, 3, 1)
        grid.addWidget(save_button, 3, 2)
        grid.addWidget(export_button, 3, 3)
        grid.addWidget(play_noisy_button, 4, 0)
        grid.addWidget(play_denoised_button, 4, 1)
        grid.addWidget(play_clean_button, 4, 2)
        grid.addWidget(stop_button, 4, 3)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(4, 1)

        content = QHBoxLayout()
        content.setSpacing(12)
        content.addWidget(self.tabs, stretch=4)
        metrics_box = QGroupBox("Metrics")
        metrics_box.setMinimumWidth(330)
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(12, 14, 12, 12)
        metrics_layout.setSpacing(10)
        metrics_layout.addWidget(self.metrics_label)
        metrics_layout.addWidget(self.metrics_table)
        content.addWidget(metrics_box, stretch=1)

        outer.addWidget(controls)
        outer.addLayout(content)
        return root

    def choose_checkpoint(self) -> None:
        """Select a model checkpoint."""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select checkpoint",
            str(PROJECT_ROOT / "checkpoints"),
            "PyTorch checkpoint (*.pt)",
        )
        if path:
            self.checkpoint_edit.setText(path)

    def choose_noisy(self) -> None:
        """Select a noisy wav file."""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select noisy wav",
            str(PROJECT_ROOT),
            "Wav files (*.wav)",
        )
        if path:
            self.noisy_path = Path(path)
            self.noisy_label.setText(str(self.noisy_path))

    def choose_clean(self) -> None:
        """Select an optional clean reference wav file."""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select clean reference wav",
            str(PROJECT_ROOT),
            "Wav files (*.wav)",
        )
        if path:
            self.clean_path = Path(path)
            self.clean_label.setText(str(self.clean_path))

    def clear_clean(self) -> None:
        """Clear the optional clean reference."""

        self.clean_path = None
        self.clean_label.setText("No clean reference selected")
        self.metrics_table.setRowCount(0)
        self.metrics_label.setText("Clean reference cleared.")

    def process_audio(self) -> None:
        """Run denoising and update plots."""

        if self.noisy_path is None:
            QMessageBox.warning(self, "Missing noisy wav", "Select a noisy wav file first.")
            return
        checkpoint_path = resolve_path(self.checkpoint_edit.text())
        try:
            self.statusBar().showMessage("Processing...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.result = run_denoising(self.noisy_path, checkpoint_path, self.clean_path)
            self.wave_canvas.plot_waveforms(self.result)
            self.spec_canvas.plot_spectrograms(self.result)
            self.mask_canvas.plot_mask(self.result)
            self.comparison_canvas.plot_comparison_errors(self.result)
            self.metrics_label.setText(self._format_metrics(self.result))
            self._update_metrics_table(self.result)
            self.statusBar().showMessage("Denoising complete")
        except Exception as exc:
            logging.exception("Processing failed")
            QMessageBox.critical(self, "Processing failed", f"{exc}\n\nLog: {LOG_PATH}")
            self.statusBar().showMessage("Processing failed")
        finally:
            QApplication.restoreOverrideCursor()

    def process_comparison(self) -> None:
        """Run denoising in comparison mode, requiring a clean reference."""

        if self.clean_path is None:
            QMessageBox.warning(self, "Missing clean reference", "Select a clean reference wav first.")
            return
        self.process_audio()
        if self.result is not None and self.result.clean_audio is not None:
            self.tabs.setCurrentWidget(self.comparison_canvas)

    def _format_metrics(self, result: DenoisingResult) -> str:
        """Format available metrics for display."""

        lines = [
            f"Sample rate: {result.sample_rate} Hz",
            f"Duration: {len(result.noisy_audio) / result.sample_rate:.3f} s",
            f"Mask range: {result.predicted_mask.min():.4f} to {result.predicted_mask.max():.4f}",
        ]
        if result.clean_audio is None:
            lines.append("")
            lines.append("Load a clean reference to compute SNR, STOI, and PESQ.")
            return "\n".join(lines)

        metrics = compute_comparison_metrics(result)
        input_snr, output_snr, snr_gain = metrics["SNR/SDR (dB)"]
        input_stoi, output_stoi, stoi_gain = metrics["STOI"]
        input_pesq, output_pesq, pesq_gain = metrics["PESQ"]
        noisy_rms, denoised_rms, rms_reduction = metrics["Residual RMS"]
        lines.extend(
            [
                "",
                "Comparison mode:",
                f"Input SNR: {input_snr:.3f} dB",
                f"Output SNR/SDR: {output_snr:.3f} dB",
                f"SNR improvement: {snr_gain:.3f} dB",
                f"Input STOI: {input_stoi:.4f}" if input_stoi is not None else "Input STOI: unavailable",
                f"Output STOI: {output_stoi:.4f}" if output_stoi is not None else "Output STOI: unavailable",
                f"STOI improvement: {stoi_gain:.4f}" if stoi_gain is not None else "STOI improvement: unavailable",
                f"Input PESQ: {input_pesq:.4f}" if input_pesq is not None else "Input PESQ: unavailable",
                f"Output PESQ: {output_pesq:.4f}" if output_pesq is not None else "Output PESQ: unavailable",
                f"PESQ improvement: {pesq_gain:.4f}" if pesq_gain is not None else "PESQ improvement: unavailable",
                f"Noisy residual RMS: {noisy_rms:.6f}",
                f"Denoised residual RMS: {denoised_rms:.6f}",
                f"Residual RMS reduction: {rms_reduction:.6f}",
            ]
        )
        if snr_gain < 0 or (stoi_gain is not None and stoi_gain < 0):
            lines.extend(
                [
                    "",
                    "Warning: denoising reduced at least one reference metric.",
                    "This can happen with audio outside the training distribution.",
                ]
            )
        return "\n".join(lines)

    def _update_metrics_table(self, result: DenoisingResult) -> None:
        """Fill the comparison metrics table."""

        metrics = compute_comparison_metrics(result)
        self.metrics_table.setRowCount(0)
        if not metrics:
            return
        self.metrics_table.setRowCount(len(metrics))
        for row_index, (metric_name, values) in enumerate(metrics.items()):
            before, after, improvement = values
            cells = [
                metric_name,
                format_metric_value(before, metric_name),
                format_metric_value(after, metric_name),
                format_metric_value(improvement, metric_name),
            ]
            for column_index, value in enumerate(cells):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter if column_index else Qt.AlignLeft | Qt.AlignVCenter)
                self.metrics_table.setItem(row_index, column_index, item)

    def save_denoised(self) -> None:
        """Save the last denoised waveform."""

        if self.result is None:
            QMessageBox.warning(self, "No result", "Process an audio file before saving.")
            return
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        default_name = "denoised.wav"
        if self.noisy_path is not None:
            default_name = f"{self.noisy_path.stem}_denoised.wav"
        default_path = DEFAULT_OUTPUT_DIR / default_name
        path, _ = QFileDialog.getSaveFileName(self, "Save denoised wav", str(default_path), "Wav files (*.wav)")
        if not path:
            return
        sf.write(path, self.result.denoised_audio, self.result.sample_rate)
        self.statusBar().showMessage(f"Saved {path}")

    def export_comparison_csv(self) -> None:
        """Export current comparison metrics to CSV."""

        if self.result is None:
            QMessageBox.warning(self, "No result", "Process an audio file before exporting metrics.")
            return
        metrics = compute_comparison_metrics(self.result)
        if not metrics:
            QMessageBox.warning(self, "No clean reference", "Load a clean reference and run comparison first.")
            return
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        default_name = "comparison_metrics.csv"
        if self.noisy_path is not None:
            default_name = f"{self.noisy_path.stem}_comparison_metrics.csv"
        default_path = DEFAULT_OUTPUT_DIR / default_name
        path, _ = QFileDialog.getSaveFileName(self, "Export comparison CSV", str(default_path), "CSV files (*.csv)")
        if not path:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["metric", "noisy_vs_clean", "denoised_vs_clean", "improvement"])
            for metric_name, values in metrics.items():
                writer.writerow([metric_name, *values])
        self.statusBar().showMessage(f"Exported comparison metrics: {path}")

    def play_audio(self, kind: str) -> None:
        """Play noisy, denoised, or clean audio."""

        try:
            path = self._audio_path_for_playback(kind)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot play audio", str(exc))
            return
        self.sound_effect.stop()
        self.sound_effect.setSource(QUrl.fromLocalFile(str(path.resolve())))
        self.sound_effect.play()
        self.statusBar().showMessage(f"Playing {kind}: {path}")

    def _audio_path_for_playback(self, kind: str) -> Path:
        """Return a wav path suitable for Qt playback."""

        TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        if kind == "noisy":
            if self.result is not None:
                path = TEMP_AUDIO_DIR / "noisy_8k.wav"
                sf.write(path, self.result.noisy_audio, self.result.sample_rate)
                return path
            if self.noisy_path is not None:
                return self.noisy_path
            raise ValueError("Load a noisy wav first.")
        if kind == "denoised":
            if self.result is None:
                raise ValueError("Process an audio file before playing denoised audio.")
            path = TEMP_AUDIO_DIR / "denoised_8k.wav"
            sf.write(path, self.result.denoised_audio, self.result.sample_rate)
            return path
        if kind == "clean":
            if self.result is not None and self.result.clean_audio is not None:
                path = TEMP_AUDIO_DIR / "clean_8k.wav"
                sf.write(path, self.result.clean_audio, self.result.sample_rate)
                return path
            if self.clean_path is not None:
                return self.clean_path
            raise ValueError("Load a clean reference first.")
        raise ValueError(f"Unknown audio kind: {kind}")


def main() -> None:
    """Run the PyQt5 application."""

    app = QApplication(sys.argv)
    apply_blue_theme(app)
    window = DenoisingWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
