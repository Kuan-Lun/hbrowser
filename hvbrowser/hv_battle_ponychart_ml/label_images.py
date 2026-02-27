"""
簡易圖片標註工具：
- 掃描 rawimage/ 下的圖片
- 支援 1..6 六個標籤（對應你的主題/角色等），可多選
- 標註結果存為 labels.json ：{"rawimage/filename.png": [1,3]}
- 支援裁切功能：按 C 進入裁切模式，拖曳選取區域，Enter 確認存檔
- 支援跳轉功能：按 G 或點擊計數器跳到指定圖片
使用：
  python label_images.py
"""

import glob
import json
import re
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, simpledialog
from typing import Literal

from PIL import Image, ImageTk

# 所有路徑以此腳本所在目錄為基準
_SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_SUBDIR = "rawimage"  # labels.json 中 key 的前綴
IMAGE_DIR = _SCRIPT_DIR / "rawimage"
LABEL_FILE = _SCRIPT_DIR / "labels.json"
MAX_SIZE = 800
LABEL_MAP = {
    1: "Twilight Sparkle",
    2: "Rarity",
    3: "Fluttershy",
    4: "Rainbow Dash",
    5: "Pinkie Pie",
    6: "Applejack",
}

CHECKPOINT_FILE = _SCRIPT_DIR / "checkpoint.pt"
THRESHOLDS_FILE = _SCRIPT_DIR / "thresholds.json"

# Suspicious sample thresholds
SUSPICIOUS_FN_PROB = 0.10  # label=1 but prob below this → possible mislabel
SUSPICIOUS_FP_PROB = 0.80  # label=0 but prob above this → possible missing label
SUSPICIOUS_MARGIN = 0.15  # |prob - threshold| below this → ambiguous

CLASS_NAMES_LIST = [LABEL_MAP[i] for i in range(1, 7)]
CLASS_SHORT = ["TS", "Ra", "FS", "RD", "PP", "AJ"]


class LabelStore:
    """標籤資料的載入、正規化、查詢與持久化。"""

    def __init__(self, label_file: Path, image_subdir: str):
        self._file = label_file
        self._subdir = image_subdir
        self._data: dict[str, list[int]] = {}
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
            self._data = self._normalize(raw)
        except Exception:
            self._data = {}

    def _normalize(self, raw: dict[str, list[int]]) -> dict[str, list[int]]:
        """正規化舊的 key 格式（去掉 data/ 前綴、處理絕對路徑等）。"""
        norm: dict[str, list[int]] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                continue
            kk = k.replace("\\", "/")
            if kk.startswith("data/"):
                kk = kk[len("data/") :]
            if self._subdir + "/" not in kk and not kk.startswith(self._subdir + "/"):
                pos = kk.find("/" + self._subdir + "/")
                if pos != -1:
                    kk = kk[pos + 1 :]
            if kk.startswith(self._subdir + "/"):
                norm[kk] = v
        return norm

    def get(self, key: str) -> list[int]:
        return list(self._data.get(key, []))

    def set(self, key: str, labels: list[int]) -> None:
        if labels:
            self._data[key] = labels
        elif key in self._data:
            del self._data[key]

    def has(self, key: str) -> bool:
        return key in self._data

    def save(self) -> None:
        self._file.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def path_to_key(self, p: Path) -> str:
        return self._subdir + "/" + p.name


def is_raw_image(p: Path) -> bool:
    """判斷是否為原始圖片（pony_chart_YYYYMMDD_HHMMSS，無額外後綴）。"""
    return bool(re.fullmatch(r"pony_chart_\d{8}_\d{6}", p.stem))


class ImageNavigator:
    """圖片列表管理、過濾與導航。"""

    def __init__(self, all_paths: list[Path], store: LabelStore):
        self._all_paths = all_paths
        self._store = store
        self._paths = list(all_paths)
        self._idx = 0
        self._filter_fn: Callable[[Path], bool] | None = None

    @property
    def total(self) -> int:
        return len(self._paths)

    @property
    def index(self) -> int:
        return self._idx

    @property
    def is_empty(self) -> bool:
        return len(self._paths) == 0

    @property
    def is_filtered(self) -> bool:
        return self._filter_fn is not None

    @property
    def current_path(self) -> Path:
        return self._paths[self._idx]

    @property
    def current_key(self) -> str:
        return self._store.path_to_key(self._paths[self._idx])

    @property
    def all_paths(self) -> list[Path]:
        return self._all_paths

    def go_next(self) -> None:
        self._idx = (self._idx + 1) % len(self._paths)

    def go_prev(self) -> None:
        self._idx = (self._idx - 1) % len(self._paths)

    def go_to(self, n: int) -> bool:
        """跳到第 n 張（1-based）。回傳是否成功。"""
        if 1 <= n <= len(self._paths):
            self._idx = n - 1
            return True
        return False

    def go_to_key(self, key: str) -> None:
        """跳到指定 key 的圖片。"""
        self._idx = self._find_index_by_key(key)

    def apply_filter(self, fn: Callable[[Path], bool] | None) -> bool:
        """套用篩選函式。回傳 False 表示篩選結果為空。"""
        current_key = self.current_key if not self.is_empty else None
        self._filter_fn = fn
        if fn is not None:
            self._paths = [p for p in self._all_paths if fn(p)]
            if not self._paths:
                self._filter_fn = None
                return False
        else:
            self._paths = list(self._all_paths)
        if current_key:
            self._idx = self._find_index_by_key(current_key)
        else:
            self._idx = 0
        return True

    def refresh_filter(self) -> None:
        """重新套用目前的過濾設定（例如 save 後）。"""
        if self._filter_fn is not None:
            self._paths = [p for p in self._all_paths if self._filter_fn(p)]
        else:
            self._paths = list(self._all_paths)
        if self._paths:
            self._idx = min(self._idx, len(self._paths) - 1)
        else:
            self._idx = 0

    def advance_after_label(self, labeled_key: str) -> None:
        """在過濾模式下，從 labeled_key 往後找下一個符合篩選的圖片。"""
        try:
            start = next(
                i
                for i, p in enumerate(self._all_paths)
                if self._store.path_to_key(p) == labeled_key
            )
        except StopIteration:
            self._idx = 0
            return

        all_len = len(self._all_paths)
        for offset in range(1, all_len + 1):
            candidate = self._all_paths[(start + offset) % all_len]
            if self._filter_fn is None or self._filter_fn(candidate):
                try:
                    self._idx = next(
                        i for i, p in enumerate(self._paths) if p == candidate
                    )
                except StopIteration:
                    self._idx = 0
                return
        self._idx = 0

    def add_path(self, path: Path) -> None:
        """加入新圖片（裁切產生）並跳轉到該圖片。"""
        self._all_paths.append(path)
        self._all_paths.sort()
        self.refresh_filter()
        try:
            self._idx = next(i for i, p in enumerate(self._paths) if p == path)
        except StopIteration:
            pass

    def contains_key(self, key: str) -> bool:
        """目前篩選後的列表中是否包含指定 key。"""
        return any(self._store.path_to_key(p) == key for p in self._paths)

    def _find_index_by_key(self, key: str) -> int:
        try:
            return next(
                i
                for i, p in enumerate(self._paths)
                if self._store.path_to_key(p) == key
            )
        except StopIteration:
            return min(self._idx, len(self._paths) - 1) if self._paths else 0


class CropHandler:
    """裁切模式的狀態管理與畫布事件處理。"""

    def __init__(
        self,
        canvas: tk.Canvas,
        on_selection_complete: Callable[[], None] | None = None,
    ):
        self._canvas = canvas
        self._on_selection_complete = on_selection_complete
        self.active: bool = False
        self._start: tuple[int, int] | None = None
        self._end: tuple[int, int] | None = None
        self._rect_id: int | None = None

        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)
        canvas.bind("<ButtonRelease-1>", self._on_release)

    def enter(self) -> None:
        self.active = True
        self._start = None
        self._end = None
        self._clear_rect()

    def exit(self) -> None:
        self.active = False
        self._start = None
        self._end = None
        self._clear_rect()

    def get_selection(self) -> tuple[int, int, int, int] | None:
        """回傳選取區域 (x1, y1, x2, y2)，若無有效選取則回傳 None。"""
        if self._start is None or self._end is None:
            return None
        sx, sy = self._start
        ex, ey = self._end
        x1, x2 = sorted((sx, ex))
        y1, y2 = sorted((sy, ey))
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None
        return x1, y1, x2, y2

    def _clear_rect(self) -> None:
        if self._rect_id is not None:
            self._canvas.delete(self._rect_id)
            self._rect_id = None

    def _on_press(self, e: "tk.Event[tk.Canvas]") -> None:
        if not self.active:
            return
        self._start = (e.x, e.y)
        self._end = None
        self._clear_rect()

    def _on_drag(self, e: "tk.Event[tk.Canvas]") -> None:
        if not self.active or self._start is None:
            return
        self._clear_rect()
        self._rect_id = self._canvas.create_rectangle(
            self._start[0],
            self._start[1],
            e.x,
            e.y,
            outline="red",
            width=2,
            dash=(4, 4),
        )

    def _on_release(self, e: "tk.Event[tk.Canvas]") -> None:
        if not self.active or self._start is None:
            return
        self._end = (e.x, e.y)
        if self._on_selection_complete:
            self._on_selection_complete()


class LabelApp:
    """圖片標註工具的主 UI 協調器。"""

    def __init__(self, root: tk.Tk, image_paths: list[Path]):
        self.root = root
        self.store = LabelStore(LABEL_FILE, IMAGE_SUBDIR)
        self.nav = ImageNavigator(image_paths, self.store)

        root.title(
            "Pony Chart Labeler" " (1..6 標記 | A/D 切換 | S 儲存 | C 裁切 | G 跳轉)"
        )

        # Canvas 用於圖片顯示與裁切
        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack()
        self._canvas_image_id: int | None = None

        self.crop = CropHandler(
            self.canvas,
            on_selection_complete=self._on_crop_selection_complete,
        )
        self.scale: float = 1.0
        self.current_pil_image: Image.Image | None = None

        # 數字與角色對應
        mapping_text = "  |  ".join(f"{k}: {v}" for k, v in LABEL_MAP.items())
        tk.Label(root, text=mapping_text, fg="#666", font=("Consolas", 11)).pack(
            pady=(4, 2)
        )

        tk.Label(
            root,
            text=(
                "1..6 加/取消標籤  |  A 上一張  |  D 下一張"
                "  |  S 儲存  |  C 裁切  |  G 跳轉"
            ),
            fg="#666",
        ).pack(pady=(0, 6))

        # 篩選選項
        filter_frame = tk.Frame(root)
        filter_frame.pack(pady=(0, 4))

        self.raw_only_var = tk.BooleanVar(value=False)
        self.raw_only_cb = tk.Checkbutton(
            filter_frame,
            text="只顯示原圖",
            variable=self.raw_only_var,
            command=self._on_raw_toggle,
        )
        self.raw_only_cb.pack(side="left", padx=(0, 8))

        self.uncropped_only_var = tk.BooleanVar(value=False)
        self.uncropped_only_cb = tk.Checkbutton(
            filter_frame,
            text="只顯示未裁切原圖",
            variable=self.uncropped_only_var,
            command=self._on_uncropped_toggle,
        )
        self.uncropped_only_cb.pack(side="left", padx=(0, 8))

        self.filter_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            filter_frame,
            text="只顯示未標註",
            variable=self.filter_var,
            command=self._on_filter_toggle,
        ).pack(side="left")

        # Model analysis state
        self._model_probs: dict[str, list[float]] | None = None
        self._model_thresholds: list[float] | None = None
        self._analysis_thread: threading.Thread | None = None
        self._analysis_result: tuple[dict[str, list[float]], list[float]] | None = None
        self._analysis_error: str | None = None

        # Analyze button
        analyze_frame = tk.Frame(root)
        analyze_frame.pack(pady=(0, 4))
        self._analyze_btn = tk.Button(
            analyze_frame,
            text="Analyze labels",
            command=self._start_analysis,
        )
        self._analyze_btn.pack(side="left", padx=(0, 8))
        self._analyze_status = tk.Label(analyze_frame, text="", fg="#999")
        self._analyze_status.pack(side="left")

        # Suspicious filter frame (initially disabled)
        suspicious_frame = tk.Frame(root)
        suspicious_frame.pack(pady=(0, 4))

        tk.Label(suspicious_frame, text="Class:").pack(side="left")
        self._suspicious_class_var = tk.StringVar(value="All")
        self._suspicious_class_menu = tk.OptionMenu(
            suspicious_frame,
            self._suspicious_class_var,
            *["All", *CLASS_NAMES_LIST],
            command=lambda _: self._apply_filters(),
        )
        self._suspicious_class_menu.pack(side="left", padx=(2, 8))

        self._mislabel_var = tk.BooleanVar(value=False)
        self._mislabel_cb = tk.Checkbutton(
            suspicious_frame,
            text="Mislabel",
            variable=self._mislabel_var,
            command=self._apply_filters,
        )
        self._mislabel_cb.pack(side="left", padx=(0, 4))

        self._missing_var = tk.BooleanVar(value=False)
        self._missing_cb = tk.Checkbutton(
            suspicious_frame,
            text="Missing label",
            variable=self._missing_var,
            command=self._apply_filters,
        )
        self._missing_cb.pack(side="left", padx=(0, 4))

        self._ambiguous_var = tk.BooleanVar(value=False)
        self._ambiguous_cb = tk.Checkbutton(
            suspicious_frame,
            text="Ambiguous",
            variable=self._ambiguous_var,
            command=self._apply_filters,
        )
        self._ambiguous_cb.pack(side="left")

        self._set_suspicious_controls_state("disabled")

        # 計數器（可點擊跳轉）
        self.counter_label = tk.Label(
            root, text="", font=("Consolas", 12), fg="#0066cc", cursor="hand2"
        )
        self.counter_label.pack()
        self.counter_label.bind("<Button-1>", lambda _: self._jump_to_image())

        # 標籤與檔名資訊
        self.info_label = tk.Label(root, text="", font=("Consolas", 12))
        self.info_label.pack()

        self.current_labels: list[int] = []
        root.bind("<Key>", self._on_key)
        self._refresh()

    # ── UI 更新 ──────────────────────────────────────────────

    def _update_display(self, extra: str = "") -> None:
        self.counter_label.configure(text=f"{self.nav.index + 1} / {self.nav.total}")
        label_names = [LABEL_MAP.get(i, str(i)) for i in self.current_labels]
        info = f"labels: {label_names}\n"
        if self._model_probs is not None:
            key = self.nav.current_key
            if key in self._model_probs:
                probs = self._model_probs[key]
                parts = []
                for i, (short, prob) in enumerate(zip(CLASS_SHORT, probs)):
                    marker = "*" if (i + 1) in self.current_labels else " "
                    parts.append(f"{short}={prob:.2f}{marker}")
                info += "Model: " + "  ".join(parts) + "\n"
        info += self.nav.current_key
        if extra:
            info += f"  ({extra})"
        self.info_label.configure(text=info)

    def _refresh(self) -> None:
        if self.nav.is_empty:
            if self.nav.is_filtered:
                messagebox.showinfo("Info", "篩選結果為空。")
                self._reset_all_filters()
                if self.nav.is_empty:
                    self._show_no_images()
                    return
            else:
                self._show_no_images()
                return

        self.crop.exit()
        self._load_image(self.nav.current_path)
        self.current_labels = sorted(set(self.store.get(self.nav.current_key)))
        self._update_display()

    def _load_image(self, path: Path) -> None:
        try:
            im = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open {path}: {e}")
            return

        self.current_pil_image = im
        w, h = im.size
        self.scale = min(MAX_SIZE / max(1, w), MAX_SIZE / max(1, h), 1.0)
        display_im = im
        if self.scale < 1.0:
            display_im = im.resize((int(w * self.scale), int(h * self.scale)))

        self.tk_im = ImageTk.PhotoImage(display_im)
        dw, dh = display_im.size
        self.canvas.configure(width=dw, height=dh)
        if self._canvas_image_id is not None:
            self.canvas.delete(self._canvas_image_id)
        self._canvas_image_id = self.canvas.create_image(
            0, 0, anchor="nw", image=self.tk_im
        )

    def _show_no_images(self) -> None:
        messagebox.showinfo("Info", "No images found under ./rawimage")
        self.root.destroy()

    # ── 操作 ─────────────────────────────────────────────────

    def _toggle_label(self, v: int) -> None:
        if v in self.current_labels:
            self.current_labels.remove(v)
        else:
            self.current_labels.append(v)
            self.current_labels.sort()
        self._update_display()

    def _save(self) -> None:
        key = self.nav.current_key
        self.store.set(key, self.current_labels)
        self.store.save()

        if not self.nav.is_filtered:
            self._update_display("saved")
            return

        self.nav.refresh_filter()
        if self.nav.is_empty:
            messagebox.showinfo("Info", "篩選結果為空。")
            self._reset_all_filters()
            self.nav.go_to_key(key)
        elif not self.nav.contains_key(key):
            self.nav.advance_after_label(key)
        self._refresh()

    def _on_filter_toggle(self) -> None:
        self._apply_filters()

    def _on_raw_toggle(self) -> None:
        if not self.raw_only_var.get():
            self.uncropped_only_var.set(False)
            self.uncropped_only_cb.configure(state="normal")
        self._apply_filters()

    def _on_uncropped_toggle(self) -> None:
        if self.uncropped_only_var.get():
            self.raw_only_var.set(True)
            self.raw_only_cb.configure(state="disabled")
        else:
            self.raw_only_cb.configure(state="normal")
        self._apply_filters()

    def _build_filter_fn(self) -> Callable[[Path], bool] | None:
        """根據所有篩選 checkbox 的狀態組合出篩選函式。"""
        raw_only = self.raw_only_var.get()
        uncropped_only = self.uncropped_only_var.get()
        unlabeled_only = self.filter_var.get()

        show_mislabel = self._mislabel_var.get()
        show_missing = self._missing_var.get()
        show_ambiguous = self._ambiguous_var.get()
        suspicious_active = self._model_probs is not None and (
            show_mislabel or show_missing or show_ambiguous
        )

        if (
            not raw_only
            and not uncropped_only
            and not unlabeled_only
            and not suspicious_active
        ):
            return None

        # 預計算已有裁切圖的 raw stem 集合（避免 O(n²)）
        raw_stems_with_crops: set[str] | None = None
        if uncropped_only:
            raw_stems_with_crops = set()
            for p in self.nav.all_paths:
                if not is_raw_image(p):
                    m = re.match(r"(pony_chart_\d{8}_\d{6})", p.stem)
                    if m:
                        raw_stems_with_crops.add(m.group(1))

        store = self.store
        model_probs = self._model_probs
        model_thresholds = self._model_thresholds
        selected_class = self._suspicious_class_var.get()

        def predicate(p: Path) -> bool:
            if raw_only or uncropped_only:
                if not is_raw_image(p):
                    return False
            if raw_stems_with_crops is not None and p.stem in raw_stems_with_crops:
                return False
            if unlabeled_only and store.has(store.path_to_key(p)):
                return False
            if (
                suspicious_active
                and model_probs is not None
                and model_thresholds is not None
            ):
                key = store.path_to_key(p)
                if key not in model_probs:
                    return False
                probs = model_probs[key]
                labels = store.get(key)
                if selected_class == "All":
                    classes_to_check: range | list[int] = range(len(CLASS_NAMES_LIST))
                else:
                    classes_to_check = [CLASS_NAMES_LIST.index(selected_class)]
                match = False
                for ci in classes_to_check:
                    has_label = (ci + 1) in labels
                    prob = probs[ci]
                    thr = model_thresholds[ci]
                    if show_mislabel and has_label and prob < SUSPICIOUS_FN_PROB:
                        match = True
                        break
                    if show_missing and not has_label and prob > SUSPICIOUS_FP_PROB:
                        match = True
                        break
                    if show_ambiguous and abs(prob - thr) < SUSPICIOUS_MARGIN:
                        match = True
                        break
                if not match:
                    return False
            return True

        return predicate

    def _apply_filters(self) -> None:
        """套用所有篩選 checkbox 的組合結果。"""
        fn = self._build_filter_fn()
        if not self.nav.apply_filter(fn):
            messagebox.showinfo("Info", "篩選結果為空。")
            self._reset_all_filters()
        self._refresh()

    def _reset_all_filters(self) -> None:
        """重置所有篩選 checkbox 並清除篩選。"""
        self.raw_only_var.set(False)
        self.uncropped_only_var.set(False)
        self.filter_var.set(False)
        self.raw_only_cb.configure(state="normal")
        self.uncropped_only_cb.configure(state="normal")
        self._mislabel_var.set(False)
        self._missing_var.set(False)
        self._ambiguous_var.set(False)
        self._suspicious_class_var.set("All")
        self.nav.apply_filter(None)

    def _jump_to_image(self) -> None:
        n = simpledialog.askinteger(
            "跳轉",
            f"輸入圖片編號 (1-{self.nav.total})：",
            minvalue=1,
            maxvalue=self.nav.total,
            parent=self.root,
        )
        if n is not None:
            self.nav.go_to(n)
            self._refresh()

    # ── Model analysis ─────────────────────────────────────────

    def _set_suspicious_controls_state(
        self, state: Literal["normal", "disabled"],
    ) -> None:
        """Enable or disable suspicious filter controls."""
        self._suspicious_class_menu.configure(state=state)
        self._mislabel_cb.configure(state=state)
        self._missing_cb.configure(state=state)
        self._ambiguous_cb.configure(state=state)

    def _start_analysis(self) -> None:
        """Start model analysis in a background thread."""
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            return
        if not CHECKPOINT_FILE.exists():
            messagebox.showerror("Error", f"Checkpoint not found: {CHECKPOINT_FILE}")
            return
        if not THRESHOLDS_FILE.exists():
            messagebox.showerror("Error", f"Thresholds not found: {THRESHOLDS_FILE}")
            return

        self._analyze_btn.configure(state="disabled")

        # Build sample list (only labeled images)
        samples: list[tuple[str, list[int]]] = []
        keys: list[str] = []
        for p in self.nav.all_paths:
            key = self.store.path_to_key(p)
            if self.store.has(key):
                samples.append((str(p), self.store.get(key)))
                keys.append(key)

        self._analyze_status.configure(
            text=f"Analyzing {len(samples)} images..."
        )
        self._analysis_result = None
        self._analysis_error = None
        self._analysis_thread = threading.Thread(
            target=self._run_analysis_thread,
            args=(samples, keys),
            daemon=True,
        )
        self._analysis_thread.start()
        self.root.after(200, self._poll_analysis)

    def _run_analysis_thread(
        self,
        samples: list[tuple[str, list[int]]],
        keys: list[str],
    ) -> None:
        """Background thread: run model inference on all labeled images."""
        try:
            import numpy as np
            import torch

            from .common import (
                BACKBONE,
                BATCH_SIZE,
                build_model,
                get_device,
                get_transforms,
                make_dataloader,
            )
            from .common.data import PonyChartDataset

            device = get_device()
            model = build_model(backbone=BACKBONE, pretrained=False).to(device)
            state_dict = torch.load(
                CHECKPOINT_FILE, map_location=device, weights_only=True
            )
            model.load_state_dict(state_dict)
            model.eval()

            with open(THRESHOLDS_FILE, encoding="utf-8") as f:
                thr_data: dict[str, float] = json.load(f)
            thresholds = [thr_data[name] for name in CLASS_NAMES_LIST]

            transform = get_transforms(is_train=False)
            dataset = PonyChartDataset(samples, transform)
            loader = make_dataloader(
                dataset,
                BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                device=device,
            )

            all_probs_list: list[np.ndarray] = []
            with torch.no_grad():
                for images, _ in loader:
                    logits = model(images.to(device))
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs_list.append(probs)

            all_probs = np.concatenate(all_probs_list)
            result: dict[str, list[float]] = {}
            for i, key in enumerate(keys):
                result[key] = all_probs[i].tolist()

            self._analysis_result = (result, thresholds)
        except Exception as e:
            self._analysis_error = str(e)

    def _poll_analysis(self) -> None:
        """Poll background analysis thread for completion."""
        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            self.root.after(200, self._poll_analysis)
            return

        self._analysis_thread = None

        if self._analysis_error is not None:
            self._analyze_btn.configure(state="normal")
            self._analyze_status.configure(text="")
            messagebox.showerror("Analysis Error", self._analysis_error)
            self._analysis_error = None
            return

        if self._analysis_result is not None:
            self._model_probs, self._model_thresholds = self._analysis_result
            self._analysis_result = None
            count = len(self._model_probs)
            self._analyze_status.configure(text=f"Done ({count} images)")
            self._set_suspicious_controls_state("normal")
            self._refresh()
        else:
            self._analyze_btn.configure(state="normal")
            self._analyze_status.configure(text="")

    # ── 裁切 ─────────────────────────────────────────────────

    def _on_crop_selection_complete(self) -> None:
        self.info_label.configure(text="裁切模式：Enter 確認儲存，Escape 取消")

    def _save_crop(self) -> None:
        if self.current_pil_image is None:
            return
        sel = self.crop.get_selection()
        if sel is None:
            messagebox.showwarning("裁切", "選取區域太小，請重新拖曳。")
            return

        x1, y1, x2, y2 = sel
        w, h = self.current_pil_image.size
        orig = (
            max(0, min(int(x1 / self.scale), w)),
            max(0, min(int(y1 / self.scale), h)),
            max(0, min(int(x2 / self.scale), w)),
            max(0, min(int(y2 / self.scale), h)),
        )
        cropped = self.current_pil_image.crop(orig)

        current_path = self.nav.current_path
        base_stem = re.sub(r"_crop\d+$", "", current_path.stem)
        base_path = current_path.parent / f"{base_stem}{current_path.suffix}"
        save_path = self._next_crop_name(base_path)

        cropped.save(save_path)
        self.nav.add_path(save_path)
        self._refresh()
        self._update_display(f"已儲存裁切圖：{save_path.name}")

    @staticmethod
    def _next_crop_name(base_path: Path) -> Path:
        """產生下一個可用的 _cropN 檔名。"""
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        n = 1
        while True:
            candidate = parent / f"{stem}_crop{n}{suffix}"
            if not candidate.exists():
                return candidate
            n += 1

    # ── 鍵盤事件 ─────────────────────────────────────────────

    def _on_key(self, e: "tk.Event[tk.Misc]") -> None:
        k = e.keysym.lower()

        if self.crop.active:
            if k == "return":
                self._save_crop()
            elif k == "escape":
                self.crop.exit()
                self._refresh()
            return

        if k in ("1", "2", "3", "4", "5", "6"):
            self._toggle_label(int(k))
        elif k == "a":
            self.nav.go_prev()
            self._refresh()
        elif k == "d":
            self.nav.go_next()
            self._refresh()
        elif k == "s":
            self._save()
        elif k == "c":
            self.crop.enter()
            self.info_label.configure(
                text="裁切模式：拖曳選取區域，Enter 確認，Escape 取消"
            )
        elif k == "g":
            self._jump_to_image()


def main() -> None:
    if not IMAGE_DIR.exists():
        messagebox.showerror("Error", f"找不到資料夾: {IMAGE_DIR}")
        return
    paths = [Path(p) for p in glob.glob(str(IMAGE_DIR / "*"))]
    paths = [
        p
        for p in paths
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and p.is_file()
    ]
    paths.sort()
    root = tk.Tk()
    LabelApp(root, paths)
    root.mainloop()


if __name__ == "__main__":
    main()
