"""
簡易圖片標註工具：
- 掃描 rawimage/ 下的圖片
- 支援 1..6 六個標籤（對應你的主題/角色等），可多選
- 標註結果存為 labels.json ：{"pony_chart/filename.png": [1,3]}
使用：
  python label_images.py
"""

import glob
import json
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

        root.title("Pony Chart Labeler (1..6 標記 | A 上一張 | D 下一張 | S 儲存)")
        self.canvas = tk.Label(root)
        self.canvas.pack()

        # 顯示數字與角色對應
        mapping_text = "  |  ".join(f"{k}: {v}" for k, v in LABEL_MAP.items())
        self.mapping_label = tk.Label(
            root, text=mapping_text, fg="#666", font=("Consolas", 11)
        )
        self.mapping_label.pack(pady=(4, 2))

        self.help = tk.Label(
            root,
            text="1..6 加/取消標籤  |  A 上一張  |  D 下一張  |  S 儲存",
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

    def on_filter_toggle(self):
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

    def refresh(self):
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
        p = self.image_paths[self.idx]
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open {p}: {e}")
            return
        w, h = im.size
        scale = min(MAX_SIZE / max(1, w), MAX_SIZE / max(1, h), 1.0)
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))
        self.tk_im = ImageTk.PhotoImage(im)
        self.canvas.configure(image=self.tk_im)

        key = self.image_key()
        self.current_labels = sorted(list(set(self.labels.get(key, []))))
        label_names = [LABEL_MAP.get(i, str(i)) for i in self.current_labels]
        self.preview.configure(
            text=f"labels: {label_names}  ({self.idx+1}/{len(self.image_paths)})\n{key}"
        )

    def toggle_label(self, v: int):
        if v in self.current_labels:
            self.current_labels.remove(v)
        else:
            self.current_labels.append(v)
            self.current_labels.sort()
        label_names = [LABEL_MAP.get(i, str(i)) for i in self.current_labels]
        self.preview.configure(
            text=f"labels: {label_names}  ({self.idx+1}/{len(self.image_paths)})\n{self.image_key()}"
        )

    def save(self):
        key = self.image_key()
        if self.current_labels:
            self.labels[key] = self.current_labels
        elif key in self.labels:
            del self.labels[key]
        LABEL_FILE.write_text(
            json.dumps(self.labels, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        label_names = [LABEL_MAP.get(i, str(i)) for i in self.current_labels]
        self.preview.configure(
            text=f"labels: {label_names}  ({self.idx+1}/{len(self.image_paths)})\n{self.image_key()}  (saved)"
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
                next_idx = None
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

    def on_key(self, e: tk.Event):
        k = e.keysym.lower()
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


def main():
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
    app = LabelApp(root, paths)
    root.mainloop()


if __name__ == "__main__":
    main()
