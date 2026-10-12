# HBrowser

HBrowser 是讓 Python 程式透過瀏覽器操作 E-Hentai／ExHentai 的非同步函式庫。
你可以用它登入帳號、搜尋圖庫、依 GID 尋找圖庫、提交 H@H 封存下載，以及每日簽到。
它適合整合進自己的腳本或應用程式；目前沒有獨立的命令列工具。

## 從 0.44 升級

0.45 移除公開模組 `hbrowser.gallery.browser.mapper`，屬於不向後相容的 Python API
變更。下列匯出不再提供：

- `ZENDRIVER_MAPPER_CLEANUP_INTERVAL_SECONDS`
- `prune_zendriver_browser_mappers`
- `prune_zendriver_connection_mapper`
- `run_zendriver_mapper_janitor`
- `start_zendriver_mapper_janitor`
- `stop_zendriver_mapper_janitor`

新的 Zendriver 0.17.1 cohort 已負責 event transaction 清理，因此不保留舊 mapper
清理函式或相容 shim。若應用程式直接匯入上述名稱，請移除舊 janitor 的啟動、停止與
定時清理呼叫；瀏覽器生命週期仍透過 `Driver`／`EHDriver`／`ExHDriver` 的
`async with` 管理。依賴 prune 回傳計數或自訂排程的程式需要人工調整，無法靠通用
的一次性程式改寫保證原本語義。

這項 API 移除不改變儲存資料格式，既有 cookie、profile 與下載檔案可保留，不需要
刪除或重建資料。

## 開始使用

你需要 Python 3.14 以上版本、E-Hentai 帳號與網路連線。
支援 Windows、macOS 與 Linux；以下範例會開啟瀏覽器視窗，因此需要圖形桌面。
首次啟動時會自動安裝 Chrome for Testing，也可以指定已安裝的 Chrome。

### 1. 安裝

在這份 checkout 的根目錄建立虛擬環境並安裝套件。

macOS／Linux：

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Windows PowerShell：

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
```

### 2. 設定帳號

在執行腳本的終端機設定環境變數。請填入自己的帳號密碼，避免將密碼寫進程式碼或
提交到版本庫。

macOS／Linux：

```bash
export EH_USERNAME='your_username'
export EH_PASSWORD='your_password'
export USE_TOR=0
```

Windows PowerShell：

```powershell
$env:EH_USERNAME = 'your_username'
$env:EH_PASSWORD = 'your_password'
$env:USE_TOR = '0'
```

`USE_TOR=0` 選擇直接連線。若不設定，HBrowser 偵測到本機 Tor 執行檔時會自動使用它。
其他連線方式見下方「瀏覽器與連線設定」。

### 3. 執行第一次搜尋

將以下內容存為 `search_galleries.py`，修改搜尋條件後執行
`python search_galleries.py`：

```python
import asyncio

from hbrowser import EHDriver, SearchRequest


async def main() -> None:
    async with EHDriver(headless=False) as driver:
        result = await driver.search(
            SearchRequest(
                scope_url="https://e-hentai.org/",
                query="language:chinese$",
            )
        )
        for gallery in result.galleries:
            print(gallery.gid, gallery.url)
        print(f"找到 {len(result.galleries)} 筆圖庫，讀取 {result.pages_visited} 頁")


if __name__ == "__main__":
    asyncio.run(main())
```

進入 `async with` 時會啟動瀏覽器、登入並前往首頁；離開區塊時會關閉瀏覽器。
第一次使用建議保留 `headless=False`，以便在必要時手動完成驗證。

| 使用網站 | 匯入並建立的 driver | `scope_url` |
| --- | --- | --- |
| E-Hentai | `EHDriver` | `https://e-hentai.org/` |
| ExHentai | `ExHDriver` | `https://exhentai.org/` |

使用 ExHentai 時，帳號必須具有該站存取權；將範例的匯入、driver 與網址一起替換即可。

搜尋會收集去重後的結果，預設上限為 100 頁、5,000 筆。可以在 `SearchRequest`
設定較小的 `max_pages` 或 `max_results`，但不能超過上述上限。
如果尚有結果卻已達上限，會拋出 `SearchLimitExceededError`；請縮小搜尋範圍。

## 常見操作

以下片段放在上例 `async with ... as driver:` 的區塊內；相應的 `import`
可放在腳本頂端。這些操作共用已登入的瀏覽器。

### 依 GID 尋找圖庫

```python
from hbrowser import ConfirmedGalleryMissing, GalleryFound

match await driver.lookup_gid(349189):
    case GalleryFound(gallery=gallery):
        print(gallery.url)
    case ConfirmedGalleryMissing(confirmations=confirmations):
        print(f"經過 {confirmations} 次獨立搜尋，未找到此圖庫")
```

`lookup_gid()` 接受正整數 GID。只有兩次獨立搜尋都明確回傳空結果，才會回傳
`ConfirmedGalleryMissing`。登入、驗證、導覽或頁面解析失敗會拋出例外，不能當成
圖庫不存在的證據。

### 提交 H@H 下載

先準備可用的 H@H 客戶端，以及網站要求的下載額度或費用，再提交圖庫網址：

```python
from h2h_galleryinfo_parser import GalleryURLParser

if await driver.checkh2h():
    # 改成要下載的圖庫網址。
    gallery = GalleryURLParser("https://e-hentai.org/g/123/456/")
    accepted = await driver.download(gallery)
    print(f"下載提交成功：{accepted}")
else:
    print("請先讓 H@H 客戶端上線")
```

`download()` 的 `True` 表示已排入 H@H 下載，不表示檔案已下載完成；圖庫無法取得等
情況可能回傳 `False`。檔案接收與儲存位置由 H@H 處理，HBrowser 不回傳封存檔內容。
若收到 `ArchiveDownloadOutcomeUnknownError`，請先檢查 H@H 狀態再決定是否重試，
因為前一次操作可能已生效。

### 每日簽到

```python
from hbrowser import PunchInComplete, RandomEncounterFound

match await driver.punchin():
    case RandomEncounterFound(url=url):
        await driver.get(url)
    case PunchInComplete():
        print("簽到完成，沒有隨機遭遇")
```

`RandomEncounterFound` 表示簽到頁提供了 HentaiVerse 隨機遭遇；範例會開啟它。
若只要簽到，可以略過 `driver.get(url)`。遭遇網址包含短效的私人資訊，請勿記錄或分享。

## 瀏覽器與連線設定

請在建立 driver 之前設定需要的環境變數。

| 環境變數 | 用途 |
| --- | --- |
| `HBROWSER_CHROME_EXECUTABLE` | 指定 Chrome 執行檔的絕對路徑，略過自動安裝；必須是可執行的檔案。 |
| `RP_DNS`、`RP_USERNAME`、`RP_PASSWORD` | 住宅 HTTP 代理的 `host:port` 與認證；三者須一起提供。 |
| `USE_TOR` | `0` 停用 Tor，`1` 啟用 Tor；未設定時自動偵測。 |
| `TOR_BINARY_PATH` | 自動偵測不到 Tor 時，指定其執行檔路徑。 |
| `FLARESOLVERR_URL` | 選用的 FlareSolverr `/v1` 端點，例如 `http://127.0.0.1:8191/v1`。 |

FlareSolverr 可協助處理支援的 Cloudflare 與登入驗證；它必須與 HBrowser 使用相同的
對外網路路徑。啟用 Tor 或住宅代理時，HBrowser 會停用 FlareSolverr 整合。

住宅代理使用 Manifest V3 擴充功能。請使用支援命令列載入 unpacked extension 的
Chrome for Testing 或 Chromium；Chrome branded builds 自 137 起移除
[`--load-extension`](https://groups.google.com/a/chromium.org/g/chromium-extensions/c/1-g8EFx2BBY/m/S0ET5wPjCAAJ)，
因此這類指定執行檔不能用於住宅代理。HBrowser 在任何網路導航前，先核對本次
擴充功能的認證 listener 與 Chrome 實際代理設定，再驗證出口 IP。擴充功能未載入、
設定未生效或驗證失敗時會清理並拒絕啟動，不退回直接連線或 Manifest V2。

Driver 預設 `headless=True`。無法自動完成驗證時，這個模式會失敗；需要人工操作時改用
`headless=False`。人工驗證預設等待 180 秒，可以在建立 driver 時設定
`captcha_manual_timeout`；初始 FlareSolverr session 嘗試次數由
`flaresolverr_session_attempts` 設定，預設為 3 次。

## 排錯與日誌

| 遇到的情況 | 處理方式 |
| --- | --- |
| 瀏覽器無法啟動 | 確認可下載 Chrome，或檢查指定的 Chrome 路徑；有視窗模式需要圖形桌面。 |
| 登入或驗證失敗 | 確認帳號、密碼及網站存取權，使用 `headless=False` 查看並完成驗證。 |
| 搜尋超過上限 | 縮小搜尋條件或使用更明確的 `scope_url`。 |
| H@H 離線或額度不足 | 先恢復客戶端連線或處理帳號額度，再提交下載。 |
| 下載結果不明 | 檢查 H@H 是否已收到工作，避免直接重複提交。 |
| `ProcessOwnershipError` | 確認作業系統可正常管理子程序；精簡的 POSIX 環境需要 `ps` 與支援 `WNOWAIT` 的 `os.waitid`。 |

需要更詳細的日誌時，在搜尋範例頂端增加以下匯入，並將原本的
`if __name__ == "__main__":` 區塊替換為：

```python
import os

from hbrowser import LogLevel, close_logging, configure_logging

if __name__ == "__main__":
    os.environ["HBROWSER_LOG_DIR"] = "./private-run-log"
    configure_logging(console_level=LogLevel.INFO, file_level=LogLevel.DEBUG)
    try:
        asyncio.run(main())
    finally:
        close_logging()
```

每個同時執行的程式請使用不同的可寫目錄，設定後不要更換路徑。
日誌寫入 `events-*.jsonl`；舊檔不會自動刪除，可在程式關閉日誌後自行封存或清理。
必要的日誌寫入失敗會在設定或結束檢查時拋出 `LogPersistenceError`。

瀏覽器失敗時也可能儲存 HTML 診斷檔；這些檔案可能含有帳號相關內容，請妥善保管。
回報問題時請附上套件版本、作業系統、例外名稱，以及是否使用無視窗模式、Tor 或
FlareSolverr，並移除日誌中的私人資訊。問題可提交到
[issue tracker](https://github.com/Kuan-Lun/hbrowser/issues)。

## 共用開發工具

`scripts/_shared/h2h_dev_tools/` 是由獨立 `h2h-dev-tools` 專案產生並提交的
工具快照，包含版本與逐檔雜湊；不要直接修改。一般開發仍使用
`scripts/check-fast.sh`、`scripts/check-full.sh` 與 `scripts/git-flow-merge.sh`，
不需要 sibling checkout、安裝共用工具套件或在每次 commit 時連網。

程式變更合併時需要本機 `.venv` 與 PATH 中已登入的 `codex`。合併腳本會先
審查真正的 candidate，再執行離線 gate；首次審查涵蓋完整專案，通過並合併後
才可使用增量審查。新 checkout 缺少本機紀錄或審查基準失效時會重新完整審查。
純文件合併只執行文件 gate。詳細證據及中斷處理契約見 [AGENTS.md](AGENTS.md)。
大型專案的首次審查可明確延長至一小時：

```bash
H2H_CODE_REVIEW_TIMEOUT_SECONDS=3600 scripts/git-flow-merge.sh
```

更新時，在獨立工具環境安裝明確選定的 `h2h-dev-tools` wheel 或版本，再執行：

```bash
h2h-dev-tools sync --repository /absolute/path/to/this-checkout
```

檢查產生的差異、執行本 repository 的檢查後，將快照與 manifest 一起提交。
尚未發布的 candidate wheel 也可供同步；它不代表 PyPI 已有該版本。
`check-fast.sh` 會離線驗證快照，亦可單獨執行：

```bash
.venv/bin/python scripts/_shared/h2h_dev_tools/bundle_verify.py --repository .
```

## 授權

本專案採用 GPL-3.0-only，詳見 [LICENSE](LICENSE)。
