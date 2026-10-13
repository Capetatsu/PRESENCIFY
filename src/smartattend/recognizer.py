"""
recognizer.py – Face recognition using EdgeFace XS (Phase 3, verified).

Model: EdgeFace XS gamma=0.6 (IEEE TBIOM 2024, Idiap Research Institute)
License: CC BY-NC-SA 4.0 — non-commercial research use only
Checkpoint: edgeface_xs_gamma_06.pt (7.17 MB, bundled in torch.hub cache)
CPU latency: avg 37 ms / max 68 ms (verified on Intel i3-class, 10 runs)
Embedding: 512-d, L2-normalised; cosine similarity in [-1, 1]

This module:
  - Loads the model ONCE from the bundled hub cache (no internet needed after first pull)
  - Produces 512-d L2-normalised embeddings
  - Compares via cosine similarity (dot product of unit vectors)
  - NEVER labels similarity as probability
  - Routes uncertain matches to REVIEW_REQUIRED
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .aligner import preprocess_for_recognition

# Bundled checkpoint path (downloaded via torch.hub on first use)
_HUB_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub",
                         "otroshi_edgeface_main")
_CKPT_PATH = os.path.join(_HUB_DIR, "checkpoints", "edgeface_xs_gamma_06.pt")


class MatchStatus(str, Enum):
    PRESENT         = "PRESENT"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    UNKNOWN         = "UNKNOWN"
    MODEL_ERROR     = "MODEL_ERROR"


@dataclass
class MatchResult:
    status: MatchStatus
    student_id: Optional[str]
    student_name: Optional[str]
    similarity: Optional[float]          # cosine similarity; NOT a probability
    threshold_accept: float
    threshold_review: float
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "student_id": self.student_id,
            "student_name": self.student_name,
            "similarity": round(self.similarity, 4) if self.similarity is not None else None,
            "threshold_accept": self.threshold_accept,
            "threshold_review": self.threshold_review,
            "latency_ms": round(self.latency_ms, 1),
        }


class FaceRecognizer:
    """
    EdgeFace XS-based face recognizer.

    Thresholds are starting defaults. Calibrate on a held-out validation
    split before running final experiments.
    """

    MODEL_NAME       = "EdgeFace XS gamma=0.6"
    MODEL_CHECKPOINT = "edgeface_xs_gamma_06.pt (7.17 MB)"
    MODEL_PAPER      = "IEEE TBIOM 2024, DOI 10.1109/TBIOM.2024.3352164"
    LICENSE          = "CC BY-NC-SA 4.0 — non-commercial research only"
    PREPROCESSING    = "BGR→RGB, /127.5-1, 112×112 NCHW float32"

    # Starting thresholds — MUST be calibrated on validation data
    DEFAULT_THRESHOLD_ACCEPT = 0.40
    DEFAULT_THRESHOLD_REVIEW = 0.30

    def __init__(
        self,
        threshold_accept: float = DEFAULT_THRESHOLD_ACCEPT,
        threshold_review: float = DEFAULT_THRESHOLD_REVIEW,
    ) -> None:
        self._threshold_accept = threshold_accept
        self._threshold_review = threshold_review
        self._model = None
        self._load_time_ms: Optional[float] = None
        self._load_error: Optional[str] = None

    def load(self) -> bool:
        """Load model from bundled hub cache. Returns True on success."""
        if self._model is not None:
            return True

        # Path 1: bundled in hub repo directory
        if os.path.exists(_CKPT_PATH) and os.path.exists(_HUB_DIR):
            try:
                sys.path.insert(0, _HUB_DIR)
                import hubconf  # type: ignore
                t0 = time.perf_counter()
                model = hubconf.edgeface_xs_gamma_06(pretrained=False)
                state = torch.load(_CKPT_PATH, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                model.eval()
                self._model = model
                self._load_time_ms = (time.perf_counter() - t0) * 1000
                return True
            except Exception as exc:
                self._load_error = f"local load failed: {exc}"

        # Path 2: torch.hub (requires internet; may prompt for trust)
        try:
            t0 = time.perf_counter()
            model = torch.hub.load(
                "otroshi/edgeface",
                "edgeface_xs_gamma_06",
                source="github",
                pretrained=True,
                trust_repo=True,
                verbose=False,
            )
            model.eval()
            self._model = model
            self._load_time_ms = (time.perf_counter() - t0) * 1000
            return True
        except Exception as exc:
            self._load_error = str(exc)
            return False

    def embed(self, face_112_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute a 512-d L2-normalised embedding.
        Returns None on failure.
        """
        if self._model is None:
            if not self.load():
                return None
        try:
            tensor = preprocess_for_recognition(face_112_bgr)
            with torch.no_grad():
                emb = self._model(tensor)
            return F.normalize(emb, p=2, dim=1)[0].numpy()
        except Exception:
            return None

    def match(
        self,
        query_embedding: np.ndarray,
        enrolled_embeddings: dict[str, tuple[str, np.ndarray]],
    ) -> MatchResult:
        """
        Compare query to all enrolled templates. Return best match decision.

        Similarity = cosine similarity in [-1, 1]. It is NOT a probability.
        Status:
          PRESENT          — best_sim >= threshold_accept
          REVIEW_REQUIRED  — threshold_review <= best_sim < threshold_accept
          UNKNOWN          — best_sim < threshold_review
        """
        t0 = time.perf_counter()

        if not enrolled_embeddings:
            return MatchResult(
                status=MatchStatus.UNKNOWN, student_id=None, student_name=None,
                similarity=None, threshold_accept=self._threshold_accept,
                threshold_review=self._threshold_review,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        best_id, best_name, best_sim = None, None, -2.0
        for sid, (sname, emb) in enrolled_embeddings.items():
            sim = float(np.dot(query_embedding, emb))
            if sim > best_sim:
                best_sim, best_id, best_name = sim, sid, sname

        lat = (time.perf_counter() - t0) * 1000

        if best_sim >= self._threshold_accept:
            status = MatchStatus.PRESENT
        elif best_sim >= self._threshold_review:
            status = MatchStatus.REVIEW_REQUIRED
        else:
            status = MatchStatus.UNKNOWN
            best_id = best_name = None

        return MatchResult(
            status=status, student_id=best_id, student_name=best_name,
            similarity=best_sim, threshold_accept=self._threshold_accept,
            threshold_review=self._threshold_review, latency_ms=lat,
        )

    @property
    def status(self) -> dict:
        return {
            "model": self.MODEL_NAME,
            "checkpoint": self.MODEL_CHECKPOINT,
            "paper": self.MODEL_PAPER,
            "license": self.LICENSE,
            "preprocessing": self.PREPROCESSING,
            "checkpoint_path": _CKPT_PATH,
            "checkpoint_exists": os.path.exists(_CKPT_PATH),
            "loaded": self._model is not None,
            "load_time_ms": round(self._load_time_ms, 1) if self._load_time_ms else None,
            "load_error": self._load_error,
            "threshold_accept": self._threshold_accept,
            "threshold_review": self._threshold_review,
            "verified_latency": "avg 37 ms / max 68 ms (Intel i3, CPU, 10 runs)",
            "note": "Thresholds are defaults — calibrate on validation split before experiments.",
        }


# Singleton
_RECOGNIZER: Optional[FaceRecognizer] = None


def get_recognizer() -> FaceRecognizer:
    global _RECOGNIZER
    if _RECOGNIZER is None:
        _RECOGNIZER = FaceRecognizer()
        _RECOGNIZER.load()
    return _RECOGNIZER
