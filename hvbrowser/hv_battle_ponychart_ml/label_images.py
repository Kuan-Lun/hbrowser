"""
簡易圖片標註工具：
- 掃描 rawimage/ 下的圖片
- 支援 1..6 六個標籤（對應你的主題/角色等），可多選
- 標註結果存為 labels.json ：{"pony_chart/filename.png": [1,3]}
- 支援裁切功能：按 C 進入裁切模式，拖曳選取區域，Enter 確認存檔
使用：
  python label_images.py
"""

import glob
import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

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


class LabelApp:
    def __init__(self, root: tk.Tk, image_paths: list[Path]):
        self.root = root
        # 全部圖片列表（不過濾）
        self.all_paths = image_paths
        # 目前顯示中的（可能被未標註過濾）
        self.image_paths = list(self.all_paths)
        self.idx = 0
        self.labels: dict[str, list[int]] = {}
        if LABEL_FILE.exists():
            try:
                raw = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
                # 正規化舊的 key：
                # 1. 去掉前綴 data/
                # 2. 若含有絕對/其它路徑，擷取 pony_chart 之後的子路徑
                norm = {}
                for k, v in raw.items():
                    if not isinstance(k, str):
                        continue
                    kk = k.replace("\\", "/")
                    if kk.startswith("data/"):
                        kk = kk[len("data/") :]
                    if IMAGE_SUBDIR + "/" not in kk and not kk.startswith(
                        IMAGE_SUBDIR + "/"
                    ):
                        # 嘗試尋找 /pony_chart/ 片段
                        pos = kk.find("/" + IMAGE_SUBDIR + "/")
                        if pos != -1:
                            kk = kk[pos + 1 :]
                    # 最終必須以 IMAGE_SUBDIR/ 開頭才視為有效
                    if kk.startswith(IMAGE_SUBDIR + "/"):
                        norm[kk] = v
                self.labels = norm
            except Exception:
                self.labels = {}

        root.title(
            "Pony Chart Labeler (1..6 標記 | A/D 切換 | S 儲存 | C 裁切)"
        )

        # 用 Canvas 取代 Label 以支援滑鼠繪製裁切框
        self.canvas = tk.Canvas(root, highlightthickness=0)
        self.canvas.pack()
        self._canvas_image_id: int | None = None

        # 裁切模式狀態
        self.crop_mode: bool = False
        self.crop_start: tuple[int, int] | None = None
        self.crop_end: tuple[int, int] | None = None
        self.crop_rect_id: int | None = None
        self.scale: float = 1.0
        self.current_pil_image: Image.Image | None = None

        # Canvas 滑鼠事件
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        # 顯示數字與角色對應
        mapping_text = "  |  ".join(f"{k}: {v}" for k, v in LABEL_MAP.items())
        self.mapping_label = tk.Label(
            root, text=mapping_text, fg="#666", font=("Consolas", 11)
        )
        self.mapping_label.pack(pady=(4, 2))

        self.help = tk.Label(
            root,
            text="1..6 加/取消標籤  |  A 上一張  |  D 下一張  |  S 儲存  |  C 裁切",
            fg="#666",
        )
        self.help.pack(pady=(0, 6))

        # 只顯示未標註的 checkbox
        self.filter_unlabeled_var = tk.BooleanVar(value=False)
        self.filter_checkbox = tk.Checkbutton(
            root,
            text="只顯示未標註",
            variable=self.filter_unlabeled_var,
            command=self.on_filter_toggle,
        )
        self.filter_checkbox.pack(pady=(0, 4))

        self.current_labels: list[int] = []
        self.preview = tk.Label(root, text="labels: []", font=("Consolas", 12))
        self.preview.pack()

        root.bind("<Key>", self.on_key)
        self.refresh()

    def image_key(self) -> str:
        """回傳 labels.json 中的 key，格式為 pony_chart/filename.png"""
        p = self.image_paths[self.idx]
        return IMAGE_SUBDIR + "/" + p.name

    def path_to_key(self, p: Path) -> str:
        return IMAGE_SUBDIR + "/" + p.name

    def on_filter_toggle(self) -> None:
        """切換是否只顯示未標註圖片"""
        if self.filter_unlabeled_var.get():
            unlabeled = [
                p for p in self.all_paths if self.path_to_key(p) not in self.labels
            ]
            self.image_paths = unlabeled
            self.idx = 0
            if not self.image_paths:
                messagebox.showinfo("Info", "所有圖片都已標註。")
                self.filter_unlabeled_var.set(False)
                self.image_paths = list(self.all_paths)
        else:
            # 取消過濾，回到全部
            current_key = self.image_key() if self.image_paths else None
            self.image_paths = list(self.all_paths)
            # 嘗試保持目前圖片位置
            if current_key:
                try:
                    self.idx = next(
                        i
                        for i, p in enumerate(self.image_paths)
                        if self.path_to_key(p) == current_key
                    )
                except StopIteration:
                    self.idx = 0
        self.refresh()

    def refresh(self) -> None:
        if not self.image_paths:
            # 若是過濾未標註造成的空 -> 已在 on_filter_toggle 處理；這裡保險再處理一次
            if self.filter_unlabeled_var.get():
                messagebox.showinfo("Info", "所有圖片都已標註。")
                self.filter_unlabeled_var.set(False)
                self.image_paths = list(self.all_paths)
                if not self.image_paths:
                    messagebox.showinfo("Info", "No images found under ./pony_chart")
                    self.root.destroy()
                    return
            else:
                messagebox.showinfo("Info", "No images found under ./pony_chart")
                self.root.destroy()
                return

        # 離開裁切模式
        self._exit_crop_mode()

        p = self.image_paths[self.idx]
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open {p}: {e}")
            return

        self.current_pil_image = im
        w, h = im.size
        self.scale = min(MAX_SIZE / max(1, w), MAX_SIZE / max(1, h), 1.0)
        display_im = im
        if self.scale < 1.0:
            display_im = im.resize(
                (int(w * self.scale), int(h * self.scale))
            )

        self.tk_im = ImageTk.PhotoImage(display_im)
        dw, dh = display_im.size
        self.canvas.configure(width=dw, height=dh)
        if self._canvas_image_id is not None:
            self.canvas.delete(self._canvas_image_id)
        self._canvas_image_id = self.canvas.create_image(
            0, 0, anchor="nw", image=self.tk_im
        )

        key = self.image_key()
        self.current_labels = sorted(list(set(self.labels.get(key, []))))
        label_names = [LABEL_MAP.get(i, str(i)) for i in self.current_labels]
        self.preview.configure(
            text=f"labels: {label_names}  ({self.idx+1}/{len(self.image_paths)})\n{key}"
        )

    def toggle_label(self, v: int) -> None:
        if v in self.current_labels:
            self.current_labels.remove(v)
        else:
            self.current_labels.append(v)
            self.current_labels.sort()
        label_names = [LABEL_MAP.get(i, str(i)) for i in self.current_labels]
        key = self.image_key()
        count = f"{self.idx + 1}/{len(self.image_paths)}"
        self.preview.configure(
            text=f"labels: {label_names}  ({count})\n{key}"
        )

    def save(self) -> None:
        key = self.image_key()
        if self.current_labels:
            self.labels[key] = self.current_labels
        elif key in self.labels:
            del self.labels[key]
        LABEL_FILE.write_text(
            json.dumps(self.labels, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        label_names = [LABEL_MAP.get(i, str(i)) for i in self.current_labels]
        key = self.image_key()
        count = f"{self.idx + 1}/{len(self.image_paths)}"
        self.preview.configure(
            text=f"labels: {label_names}  ({count})\n{key}  (saved)"
        )
        # 若當前開啟 "只顯示未標註"，且此圖片已標好 -> 重新過濾並跳下一張
        if self.filter_unlabeled_var.get():
            current_path_key = key
            self.image_paths = [
                p for p in self.all_paths if self.path_to_key(p) not in self.labels
            ]
            # 找下一個未標註
            if not self.image_paths:
                messagebox.showinfo("Info", "所有圖片都已標註。")
                self.filter_unlabeled_var.set(False)
                self.image_paths = list(self.all_paths)
            # 尋找下一張（以原 all_paths 順序為基礎）
            try:
                # 取當前圖片在 all_paths 的 index，往後找第一個未標註
                start_index = next(
                    i
                    for i, p in enumerate(self.all_paths)
                    if self.path_to_key(p) == current_path_key
                )
            except StopIteration:
                start_index = -1
            if self.filter_unlabeled_var.get():  # 仍在過濾模式
                # 從 start_index+1 開始找下一個未標註 key
                all_len = len(self.all_paths)
                for offset in range(1, all_len + 1):
                    candidate = self.all_paths[(start_index + offset) % all_len]
                    if self.path_to_key(candidate) not in self.labels:
                        # 在新的 filtered list 中的 index
                        try:
                            self.idx = next(
                                i
                                for i, p in enumerate(self.image_paths)
                                if p == candidate
                            )
                        except StopIteration:
                            self.idx = 0
                        break
            else:
                # 已退回顯示全部
                if self.image_paths:
                    try:
                        self.idx = next(
                            i
                            for i, p in enumerate(self.image_paths)
                            if self.path_to_key(p) == current_path_key
                        )
                    except StopIteration:
                        self.idx = min(self.idx, len(self.image_paths) - 1)
            self.refresh()

    # ── 裁切模式 ──────────────────────────────────────────────

    def _enter_crop_mode(self) -> None:
        self.crop_mode = True
        self.crop_start = None
        self.crop_end = None
        if self.crop_rect_id is not None:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None
        self.preview.configure(
            text="裁切模式：拖曳選取區域，Enter 確認，Escape 取消"
        )

    def _exit_crop_mode(self) -> None:
        self.crop_mode = False
        self.crop_start = None
        self.crop_end = None
        if self.crop_rect_id is not None:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None

    def on_mouse_press(self, e: "tk.Event[tk.Canvas]") -> None:
        if not self.crop_mode:
            return
        self.crop_start = (e.x, e.y)
        self.crop_end = None
        if self.crop_rect_id is not None:
            self.canvas.delete(self.crop_rect_id)
            self.crop_rect_id = None

    def on_mouse_drag(self, e: "tk.Event[tk.Canvas]") -> None:
        if not self.crop_mode or self.crop_start is None:
            return
        if self.crop_rect_id is not None:
            self.canvas.delete(self.crop_rect_id)
        self.crop_rect_id = self.canvas.create_rectangle(
            self.crop_start[0],
            self.crop_start[1],
            e.x,
            e.y,
            outline="red",
            width=2,
            dash=(4, 4),
        )

    def on_mouse_release(self, e: "tk.Event[tk.Canvas]") -> None:
        if not self.crop_mode or self.crop_start is None:
            return
        self.crop_end = (e.x, e.y)
        self.preview.configure(
            text="裁切模式：Enter 確認儲存，Escape 取消"
        )

    def _next_crop_name(self, base_path: Path) -> Path:
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

    def _save_crop(self) -> None:
        """將選取區域裁切並存檔。"""
        if (
            self.crop_start is None
            or self.crop_end is None
            or self.current_pil_image is None
        ):
            return

        sx, sy = self.crop_start
        ex, ey = self.crop_end
        # 確保左上 / 右下
        x1, x2 = sorted((sx, ex))
        y1, y2 = sorted((sy, ey))

        # 忽略太小的選取
        if x2 - x1 < 5 or y2 - y1 < 5:
            messagebox.showwarning("裁切", "選取區域太小，請重新拖曳。")
            return

        # 顯示座標 → 原圖座標
        orig_x1 = int(x1 / self.scale)
        orig_y1 = int(y1 / self.scale)
        orig_x2 = int(x2 / self.scale)
        orig_y2 = int(y2 / self.scale)

        # 限制在原圖範圍內
        w, h = self.current_pil_image.size
        orig_x1 = max(0, min(orig_x1, w))
        orig_y1 = max(0, min(orig_y1, h))
        orig_x2 = max(0, min(orig_x2, w))
        orig_y2 = max(0, min(orig_y2, h))

        cropped = self.current_pil_image.crop(
            (orig_x1, orig_y1, orig_x2, orig_y2)
        )

        # 決定存檔路徑
        current_path = self.image_paths[self.idx]
        # 取基底名稱（去掉已有的 _cropN 後綴，避免巢狀命名）
        base_stem = re.sub(r"_crop\d+$", "", current_path.stem)
        base_path = current_path.parent / f"{base_stem}{current_path.suffix}"
        save_path = self._next_crop_name(base_path)

        cropped.save(save_path)

        # 將新圖加入列表（排序後插入正確位置）
        self.all_paths.append(save_path)
        self.all_paths.sort()
        # 重建過濾列表
        if self.filter_unlabeled_var.get():
            self.image_paths = [
                p
                for p in self.all_paths
                if self.path_to_key(p) not in self.labels
            ]
        else:
            self.image_paths = list(self.all_paths)

        # 跳到新裁切的圖片
        try:
            self.idx = next(
                i
                for i, p in enumerate(self.image_paths)
                if p == save_path
            )
        except StopIteration:
            pass

        self._exit_crop_mode()
        self.refresh()
        self.preview.configure(
            text=f"已儲存裁切圖：{save_path.name}\n"
            f"({self.idx+1}/{len(self.image_paths)})"
        )

    # ── 鍵盤事件 ─────────────────────────────────────────────

    def on_key(self, e: "tk.Event[tk.Misc]") -> None:
        k = e.keysym.lower()

        # 裁切模式下的按鍵
        if self.crop_mode:
            if k == "return":
                self._save_crop()
            elif k == "escape":
                self._exit_crop_mode()
                self.refresh()
            return

        if k in ["1", "2", "3", "4", "5", "6"]:
            self.toggle_label(int(k))
        elif k == "a":
            self.idx = (self.idx - 1) % len(self.image_paths)
            self.refresh()
        elif k == "d":
            self.idx = (self.idx + 1) % len(self.image_paths)
            self.refresh()
        elif k == "s":
            self.save()
        elif k == "c":
            self._enter_crop_mode()


def main() -> None:
    # 掃描 rawimage/ 下的影像
    if not IMAGE_DIR.exists():
        messagebox.showerror("Error", f"找不到資料夾: {IMAGE_DIR}")
        return
    paths = [Path(p) for p in glob.glob(str(IMAGE_DIR / "*"))]
    paths = [
        p
        for p in paths
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"] and p.is_file()
    ]
    paths.sort()
    root = tk.Tk()
    LabelApp(root, paths)
    root.mainloop()


if __name__ == "__main__":
    main()
