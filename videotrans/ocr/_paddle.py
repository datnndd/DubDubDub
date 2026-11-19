"""PaddleOCR local provider.

Defensive, optional-dependency adapter. Importing this module must NOT import
paddleocr/paddle/cv2/numpy/torch. All heavy imports happen lazily inside
``_ensure_engine`` so the rest of the repository can import ``videotrans.ocr``
without pulling Paddle. The provider implements the ``BaseOcrProvider`` contract
and is registered lazily from ``videotrans.ocr``.
"""
from collections import OrderedDict
import threading
from typing import Optional

from videotrans.ocr._provider_base import BaseOcrProvider
from videotrans.ocr._text import join_ordered_lines, order_lines
from videotrans.ocr._types import OcrLine, OcrResult

# pyVideoTrans language code -> PaddleOCR rec model language.
_LANG_MAP = {
    "zh": "ch",
    "zh-cn": "ch",
    "zh-hans": "ch",
    "zh-tw": "chinese_cht",
    "zh-hant": "chinese_cht",
    "cht": "chinese_cht",
    "en": "en",
    "ja": "japan",
    "ja-jp": "japan",
    "ko": "korean",
    "ko-kr": "korean",
    "fr": "french",
    "de": "german",
    "ru": "cyrillic",
    "ar": "arabic",
    "es": "latin",
    "pt": "latin",
    "it": "latin",
    "vi": "latin",
    "vi-vn": "latin",
    "yue": "ch",
    "auto": "ch",
}

# PaddleOCR 2.x keys that appear in old-style result boxes can be ignored; we
# only read text + score. Key order within a text unit differs by version.
_PADDLE_MIN_CONF = 0.0
_WIN_DLL_HANDLES = []
_WIN_DLL_PATHS = set()


class PaddleOcrProvider(BaseOcrProvider):
    """Local PaddleOCR adapter with GPU-preference and CPU fallback."""

    def __init__(
        self,
        device: str = "auto",
        use_angle_cls: bool = False,
        paragraph_mode: str = "preserve_lines",
        cpu_threads: Optional[int] = None,
        det_db_thresh: float = 0.2,
        det_db_box_thresh: float = 0.3,
        det_db_unclip_ratio: float = 2.0,
        **kwargs,
    ) -> None:
        # store the normalized storage-mode engine key, init lazily
        self._prefer_device = device or "auto"
        self._use_angle_cls = bool(use_angle_cls)
        self._paragraph_mode = paragraph_mode or "preserve_lines"
        self._cpu_threads = cpu_threads
        self._det_db_thresh = float(det_db_thresh)
        self._det_db_box_thresh = float(det_db_box_thresh)
        self._det_db_unclip_ratio = float(det_db_unclip_ratio)
        self._engine = None
        self._engines: OrderedDict = OrderedDict()
        self._active_device = None
        self._active_lang = None
        self._current_lang = None
        self._model_version = None
        self._init_error = None
        self._lock = threading.RLock()

    # -- availability -----------------------------------------------------
    def is_available(self) -> bool:
        try:
            self._ensure_win_dlls()
            import paddleocr  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def _ensure_win_dlls():
        import sys
        if sys.platform == "win32":
            import os, site
            for s in site.getsitepackages():
                tlib = os.path.join(s, "torch", "lib")
                if os.path.exists(tlib) and tlib not in _WIN_DLL_PATHS:
                    try:
                        _WIN_DLL_HANDLES.append(os.add_dll_directory(tlib))
                        _WIN_DLL_PATHS.add(tlib)
                    except Exception:
                        pass
            # Paddle and Torch both load native libraries; Torch needs to load first.
            try:
                import torch  # noqa: F401
            except Exception:
                pass

    def _require_engine(self, device: str, lang: Optional[str] = None):
        """Lazily import paddle and build a PaddleOCR engine.

        Prefers GPU when requested and possible, otherwise falls back to CPU.
        Disables OneDNN (mkldnn) operator fusion and IR optimization passes to prevent fused_conv2d crashes.
        """
        import os
        os.environ["FLAGS_use_onednn"] = "0"
        os.environ["FLAGS_use_mkldnn"] = "0"

        with self._lock:
            mapped_lang = lang or self._current_lang or "ch"
            cache_key = (device, mapped_lang)
            if cache_key in self._engines:
                engine, actual_dev = self._engines[cache_key]
                self._engines.move_to_end(cache_key)
                self._engine = engine
                self._active_device = actual_dev
                self._active_lang = mapped_lang
                return self._engine

            self._ensure_win_dlls()

            try:
                try:
                    import torch  # noqa: F401
                except Exception:
                    pass
                import paddle
                from paddleocr import PaddleOCR

                try:
                    paddle.set_flags({"FLAGS_use_onednn": False, "FLAGS_use_mkldnn": False})
                except Exception:
                    pass

                try:
                    import paddle.inference as inference
                    if not getattr(inference, "_videotrans_patched", False):
                        orig_cp = inference.create_predictor
                        def safe_create_predictor(config):
                            try:
                                config.disable_mkldnn()
                            except Exception:
                                pass
                            try:
                                config.disable_onednn()
                            except Exception:
                                pass
                            try:
                                config.switch_ir_optim(False)
                            except Exception:
                                pass
                            return orig_cp(config)
                        inference.create_predictor = safe_create_predictor
                        inference._videotrans_patched = True
                except Exception:
                    pass
            except Exception as exc:
                self._init_error = str(exc)
                raise RuntimeError(
                    f"PaddleOCR could not load: {exc}. Check paddleocr, paddlepaddle, and their dependencies."
                ) from exc

            threads = (
                int(self._cpu_threads)
                if self._cpu_threads is not None
                else min(4, max(1, (os.cpu_count() or 4) // 2))
            )
            use_gpu = self._resolve_use_gpu(device)

            def _make_engine(gpu_mode: bool):
                kwargs = {
                    "use_angle_cls": self._use_angle_cls,
                    "lang": mapped_lang,
                    "use_gpu": gpu_mode,
                    "det_db_thresh": self._det_db_thresh,
                    "det_db_box_thresh": self._det_db_box_thresh,
                    "det_db_unclip_ratio": self._det_db_unclip_ratio,
                    "enable_mkldnn": False,
                    "ir_optim": False,
                    "show_log": False,
                }
                if not gpu_mode:
                    kwargs["cpu_threads"] = threads
                pop_order = [
                    "enable_mkldnn",
                    "ir_optim",
                    "cpu_threads",
                    "det_db_thresh",
                    "det_db_box_thresh",
                    "det_db_unclip_ratio",
                    "show_log",
                    "use_gpu",
                ]
                while True:
                    try:
                        return PaddleOCR(**kwargs)
                    except TypeError:
                        popped = False
                        while pop_order:
                            k = pop_order.pop(0)
                            if k in kwargs:
                                kwargs.pop(k, None)
                                popped = True
                                break
                        if not popped:
                            raise

            try:
                engine = _make_engine(use_gpu)
                actual_dev = "gpu" if use_gpu else "cpu"
            except Exception as exc:
                if use_gpu:
                    self._active_device = None
                    try:
                        engine = _make_engine(False)
                        actual_dev = "cpu"
                    except Exception as exc2:
                        self._init_error = str(exc2)
                        raise RuntimeError(f"PaddleOCR init failed: {exc2}") from exc2
                else:
                    self._init_error = str(exc)
                    raise RuntimeError(f"PaddleOCR init failed: {exc}") from exc

            self._engines[(actual_dev, mapped_lang)] = (engine, actual_dev)
            if device != actual_dev:
                self._engines[(device, mapped_lang)] = (engine, actual_dev)
            while len(self._engines) > 4:
                self._engines.popitem(last=False)
            self._engine = engine
            self._active_device = actual_dev
            self._active_lang = mapped_lang
            self._model_version = getattr(engine, "rec_model_version", None)
            return self._engine

    def _resolve_use_gpu(self, device) -> bool:
        if device in ("cpu", "cpu_only"):
            return False
        self._ensure_win_dlls()
        try:
            import paddle
            if paddle.is_compiled_with_cuda():
                try:
                    return paddle.device.cuda.device_count() > 0
                except Exception:
                    return True
        except Exception:
            pass
        return False

    # -- device -----------------------------------------------------------
    def select_device(self, prefer: Optional[str] = None) -> str:
        """Resolve the actual device (\"gpu\"/\"cpu\") for a device preference."""
        self._ensure_win_dlls()
        pref = prefer or self._prefer_device
        if pref in ("gpu", "cuda", "auto"):
            try:
                import paddle
                if paddle.is_compiled_with_cuda():
                    try:
                        if paddle.device.cuda.device_count() > 0:
                            return "gpu"
                    except Exception:
                        return "gpu"
            except Exception:
                pass
        return "cpu"

    # -- language ---------------------------------------------------------
    def map_language(self, source_language_code):
        return _LANG_MAP.get(str(source_language_code or "").lower(), "ch")

    # -- recognition ------------------------------------------------------
    def recognize(self, image, language=None) -> OcrResult:
        """Run OCR on ``image`` (BGR ndarray or file path) and return OcrResult."""
        ocr_img = image
        scale_factor = 1.0
        if hasattr(image, "shape") and len(image.shape) >= 2:
            h, w = int(image.shape[0]), int(image.shape[1])
            if 0 < h < 48 and w > 0:
                scale_factor = 2.0
                try:
                    import cv2
                    ocr_img = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                except Exception:
                    import numpy as np
                    ocr_img = np.repeat(np.repeat(image, 2, axis=0), 2, axis=1)

        def _call_ocr(eng, img):
            if hasattr(eng, "ocr"):
                try:
                    return eng.ocr(img, cls=self._use_angle_cls)
                except TypeError:
                    return eng.ocr(img)
            return eng.predict(img)

        with self._lock:
            self._ensure_win_dlls()
            device = self.select_device(self._prefer_device)
            mapped_lang = self.map_language(language)
            self._current_lang = mapped_lang
            engine = self._require_engine(device)
            try:
                raw = _call_ocr(engine, ocr_img)
            except Exception as exc:
                err_str = str(exc)
                if "OneDnnContext" in err_str or "onednn" in err_str.lower() or "fused_conv2d" in err_str:
                    import os
                    os.environ["FLAGS_use_onednn"] = "0"
                    os.environ["FLAGS_use_mkldnn"] = "0"
                    self._engines.pop((device, mapped_lang), None)
                    self._engines.pop(("cpu", mapped_lang), None)
                    self._engine = None
                    self._current_lang = mapped_lang
                    engine = self._require_engine("cpu")
                    try:
                        raw = _call_ocr(engine, ocr_img)
                    except Exception as exc2:
                        raise RuntimeError(f"PaddleOCR recognition failed: {exc2}") from exc2
                else:
                    raise RuntimeError(f"PaddleOCR recognition failed: {exc}") from exc
            actual_device = self._active_device or device

        units = self._normalize_units(raw)
        raw_lines = []
        for unit in units:
            text = (unit.get("text") or "").strip()
            if not text:
                continue
            conf = float(unit.get("confidence") or 0.0)
            box = unit.get("box")
            if box is not None and scale_factor != 1.0:
                box = tuple(float(v) / scale_factor for v in box)
            raw_lines.append(OcrLine(text=text, confidence=conf, box=box))

        unboxed, rows = order_lines(raw_lines, return_rows=True)
        ordered_lines = list(unboxed) + [ln for row in rows for ln in row]
        text = join_ordered_lines(ordered_lines, paragraph_mode=self._paragraph_mode)
        total_conf = sum(ln.confidence for ln in ordered_lines)
        avg_conf = (total_conf / len(ordered_lines)) if ordered_lines else 0.0

        return OcrResult(
            text=text,
            confidence=avg_conf,
            lines=ordered_lines,
            timestamp_ms=0,
            device=actual_device,
        )

    @staticmethod
    def _normalize_units(raw):
        """Normalize PaddleOCR 2.x list/3.x dict output to unit dicts.

        Each unit has keys text, confidence, box. box is the 4-element
        (x, y, w, h) when a 4-point polygon is available, else None.
        """
        if raw is None:
            return []
        if isinstance(raw, (list, tuple)) and len(raw) == 0:
            return []

        # 3.x dict form.
        if isinstance(raw, dict):
            texts = raw.get("rec_texts")
            if texts is None:
                texts = []
            scores = raw.get("rec_scores")
            if scores is None:
                scores = []
            polys = raw.get("rec_polys")
            if polys is None or len(polys) == 0:
                polys = raw.get("rec_boxes")
            if polys is None:
                polys = []
            units = []
            for i, text in enumerate(texts):
                score = scores[i] if i < len(scores) else 0.0
                poly = polys[i] if i < len(polys) else None
                units.append(
                    {"text": text, "confidence": score, "box": _poly_to_xywh(poly)}
                )
            return units

        # 2.x form (or 3.x [dict] list wrapper)
        if isinstance(raw, list):
            if len(raw) == 1 and isinstance(raw[0], dict):
                return PaddleOcrProvider._normalize_units(raw[0])
            units = []
            _walk_paddle_lines(raw, units)
            return units

        return []

    @staticmethod
    def _unit_order_key(unit):
        """Return (y, x) sort key from a normalized box for line ordering."""
        box = unit.get("box")
        if not box:
            return (0.0, 0.0)
        try:
            return (float(box[1]), float(box[0]))
        except (TypeError, IndexError):
            return (0.0, 0.0)


def _poly_to_xywh(poly):
    """Convert a 4-point polygon (list of [x, y]) to (x, y, w, h) or None."""
    if poly is None:
        return None
    if hasattr(poly, "tolist"):
        poly = poly.tolist()
    if not poly:
        return None
    try:
        if len(poly) == 4 and all(
            isinstance(p, (list, tuple)) and len(p) >= 2 for p in poly
        ):
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            x, y = min(xs), min(ys)
            return (x, y, max(xs) - x, max(ys) - y)
        if len(poly) >= 4:
            # flat [x0,y0,x1,y1,x2,y2,x3,y3]
            xs = [float(poly[i]) for i in range(0, len(poly), 2)]
            ys = [float(poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
            x, y = min(xs), min(ys)
            return (x, y, max(xs) - x, max(ys) - y)
    except (TypeError, IndexError, ValueError):
        return None
    return None


def _walk_paddle_lines(node, out):
    """Recursively collect [poly, (text, conf)] line units from Paddle 2.x."""
    if not isinstance(node, (list, tuple)):
        return
    if len(node) == 2 and _looks_like_text_pair(node[1]):
        out.append(
            {"text": node[1][0], "confidence": node[1][1], "box": _poly_to_xywh(node[0])}
        )
        return
    for child in node:
        _walk_paddle_lines(child, out)


def _looks_like_text_pair(pair):
    """True when pair is (text_str, numeric_confidence)."""
    if not isinstance(pair, (list, tuple)) or len(pair) < 2:
        return False
    return isinstance(pair[0], str) and isinstance(pair[1], (int, float)) and not isinstance(pair[1], bool)
