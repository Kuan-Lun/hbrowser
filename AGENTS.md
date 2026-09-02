# AGENTS.md

## 政策來源

- 本檔是此 repository 的唯一代理開發政策來源。
- 其他代理入口只能要求完整閱讀本檔，不得複製另一份政策。
- 可執行規則以 repository 內的 scripts 與設定檔為準。

## 溝通

- 最終回覆一律使用繁體中文。
- 程式碼、識別字、命令、檔名與 commit message 可使用英文。
- 不得為了承載回覆而新增 Markdown 文件。
- 移除 compatibility path、改變公開行為或採用例外時，必須在對話及
  最終回覆中明確說明。

## 設計與修改原則

- 不預設存在最小修改或向後相容要求。
- 在任務範圍內選擇架構、可讀性與可測試性最好的完整結果。
- 綜合考慮 SOLID、KISS、YAGNI、內聚性與低耦合。
- 必要的局部重構可直接納入任務。
- 若會實質擴大任務範圍、改變原要求未涵蓋的公開行為，或引入資料遷移，
  必須先取得使用者同意。
- 任務直接涉及的 legacy compatibility code 應移除，不保留 shim；不全面
  清理與任務無關的 legacy code。
- generated output 不得直接修改；必須修改 generator 或 source 後重新產生。

## 工作樹與 Git

- 唯讀分析不建立 branch。
- 凡會修改 tracked files 的任務，使用
  `scripts/detect-primary-branch.sh` 判定 primary，並在首次 tracked file 修改前
  建立專用 task branch；不得為尚未開始修改的下游或預備工作預先建立 branch。
- 不得 stash、reset、clean、覆寫或混入既有使用者修改。
- 工作樹不乾淨時，從 committed primary 建立獨立 worktree。
- task branch 可包含多個邏輯 Conventional Commits。避免巨大 commit；小而
  內聚的任務仍可只有一個 commit。
- 任務完成後從 task branch 執行 `scripts/git-flow-merge.sh`。若 task branch
  含有 primary 尚未包含的 commit，該腳本負責完整 gate、`--no-ff` merge、
  安全移除 task worktree，以及以 `git branch -d` 刪除已合併的本機 branch。
- 若且唯若 task tip 已由本機 primary 包含（`git merge-base --is-ancestor`
  成功），`scripts/git-flow-merge.sh` 執行 no-op cleanup：不得執行 gate或merge、
  建立空 commit或空 merge commit；必須先確認涉及的 worktree clean、沒有進行中
  的 Git operation、task與primary refs仍為已驗證的commit，再切回 primary或安全
  移除 task worktree，最後以 `git branch -d` 刪除本機 task branch。
- 任務取消時只可使用 `scripts/git-flow-merge.sh --cleanup-only`；若 task tip
  未由 primary 包含，或任何 safety check失敗，腳本必須 fail closed並保留 task
  branch及仍存在的worktree。no-op cleanup不得使用force、建立或刪除remote ref，
  亦不得 fetch、pull或push。
- merge conflict 或 gate failure 時必須 abort merge並保留 task branch。
- merge 後收到的任何 follow-up 都建立新的 task branch。
- 本機 task branch、commit、`--no-ff` merge與 `branch -d` 已獲預先授權。
- fetch、pull、push、remote branch、tag、release、publish、deploy與任何
  force 操作仍須逐次明確授權。
- 不得使用 `--no-verify`。

## 提交格式

- 所有非 merge commit 必須符合 Conventional Commits。
- Breaking change 使用 `type!:` 或 `BREAKING CHANGE:` footer。
- project version 更新使用獨立 commit：
  `chore(release): bump version to X.Y.Z`。

## 版本政策

- `pyproject.toml` 的 `[project].version` 是唯一 project version source。
- project version 固定使用 `X.Y.Z`。
- 1.0 前，`Y` 是 compatibility lane，`Z` 是同一 lane 內的相容 release
  counter。相容修正或功能遞增 `Z`；breaking change遞增 `Y` 並將 `Z`
  歸零。
- 1.0 後使用標準 Semantic Versioning。
- 整個 task branch只在整合前更新一次 project version。
- shipped runtime或 deployment surface 有變更時，至少需要相容升版。
- Breaking API、CLI、config、schema、protocol、資料格式或 Python/platform
  support變更必須提高 compatibility lane或 major。
- tests、一般文件、IDE、hooks、CI與 dev-only tooling 單獨變更時不升版。
- 未分類路徑必須明確判定 impact，不得靜默當作 `none`。
- `Version-Impact: none` 必須附具體理由，並在最終回覆揭露。
- project version變更必須觸發完整 direct dependency audit。
- 先更新候選版本，再執行
  `scripts/audit-dependencies.py --review-note "相容性驗證摘要"`；升版 commit
  必須包含與候選版本及 dependency manifest相符的
  `.release/dependency-audit.json`。
- `scripts/check-version.py`驗證整個 task branch；pre-merge gate以
  `--index`驗證實際 staged merge candidate。

## 依賴與環境

- repository 必須能從單一乾淨 checkout重建，不得依賴固定 sibling clone
  路徑。
- 明確跨 repository任務可使用傳入的 wheel、Git URL/ref或 repository
  path；sibling discovery只能是選擇性的效能優化。
- Python registry dependencies原則上使用 `>=` lower bound；合理 upper
  bound與 `!=` 可以保留，但必須有相容性依據。
- 精確版本只允許經驗證且有文件理由的特殊契約。
- dependency audit必須涵蓋 build、runtime、optional與 development direct
  dependencies，並搜尋現有 upper bound之外的候選版本。
- 有新版時必須檢查 release notes、驗證相容性並嘗試修正問題。
- `uv.lock` 與 `package-lock.json` 不得成為 committed或重建輸入。
  `scripts/rebuild-env.sh` 可使用 `uv venv` 與 `uv pip`，但不得使用會依賴
  project lockfile的同步流程。
- Node tooling使用 `npm install --package-lock=false`。
- 不得依賴 system-wide lint、format、type-check或 Markdown工具。
- `requires-python` 使用 `>=3.14`；只有經驗證的壞版本可使用 `!=`。

## 品質工具

- `pyproject.toml` 是 Ruff與 mypy的唯一規則來源。
- 使用 Ruff lint與 Ruff formatter，不使用 Black。
- Ruff使用適合專案的嚴格規則集，不從 `ALL` 出發；每個停用規則必須
  記錄理由。
- mypy使用標準 `strict = true`。不得保留 `mypy.ini`。
- module例外使用精確 TOML overrides。
- `type: ignore` 必須指定 error code並附理由。
- `noqa` 必須指定 rule code並附理由。
- Markdown使用 repository-local `markdownlint-cli2`。
- VS Code使用相同設定與 repository-local environment；CLI gate是最終
  權威，IDE diagnostics為即時輔助。

## 檢查分層

- `scripts/format.sh`：明確執行會修改檔案的 formatter或 fixer。
- `scripts/check-fast.sh`：離線、唯讀的 Ruff、format check、mypy與
  markdownlint；每次非 merge commit執行。
- `scripts/check-full.sh`：fast gate、完整測試、適用時的 build與wheel smoke及本
  repository的特殊檢查；整合候選只跑一次。
- dependency audit可連網，但 hooks只驗證本機 receipt，不在 commit過程
  連網。
- GitHub Actions只呼叫相同 scripts，並保留 trusted publishing、平台特有
  或本機無法可靠重現的檢查。
- 不使用 Claude、Codex或其他 provider-specific Stop hooks重複檢查。

## 測試與例外

- runtime行為變更必須新增或更新測試；bug fix必須有 regression test。
- 新功能涵蓋正常、邊界與錯誤路徑。
- 數值測試固定隨機種子；容許誤差需有依據。
- flaky test視為失敗，不得以重跑掩蓋。
- 不設定跨 repository的統一 coverage百分比。
- live account、network、production或 destructive probe不得進入 hooks、
  一般 pytest或自動 merge gate。
- `skip` 或 `xfail` 必須有理由；`xfail` 原則上使用 `strict=True`。
- 不得為通過檢查而全域放寬工具設定。

## 完成回報

最終回覆必須包含：

- 實作及公開行為變化。
- 移除的 compatibility path。
- project version與 dependency audit結果。
- commits與完整檢查結果。
- primary branch與 merge commit。
- branch/worktree是否已清除。
- 本次任務建立的 task branch/worktree之整合或 no-op cleanup結果；未清除時列出
  精確 ref/path與原因。
- 是否仍未 push、publish或 deploy。

## Repository-specific policy

`hbrowser` 是 E-Hentai／ExHentai browser automation library。它擁有底層
Zendriver browser、challenge、login、gallery navigation與owned-process
lifecycle；上層 battle策略不得下沉到本 repository。

- `Driver`、`EHDriver`與`ExHDriver`維持可替換的driver階層；所有driver
  支援 `async with`，且browser ownership、login與cleanup必須成對。
- browser、page、driver與child process ownership必須單一且可稽核；建立失敗、
  綁定錯誤、timeout或cleanup不確定時 fail closed，不得留下背景operation。
- Zendriver目前是經 lifecycle suite驗證的exact cohort。升級前必須檢查
  release notes並重跑ownership、navigation、challenge、timeout及process-tree
  regression tests。
- library module只取得namespace logger；只有明確parent composition lifecycle
  可設定logger並擁有append-only segments。child forwarding使用bounded、
  authenticated capability與bounded drain deadline。
- owned child environment只繼承文件化OS、profile/temp、locale/display/XDG與
  Python runtime allowlist；credentials、battle control、log-directory及任意
  inherited state不得外洩。
- `EH_USERNAME`、`EH_PASSWORD`及log forwarding token等secret不得硬編碼、
  記錄、顯示或納入fixtures。FlareSolverr是明確啟用的optional endpoint；
  unsupported challenge只能在GUI人工處理或fail closed。
- gallery search只追蹤trusted main-frame URLs。缺少或無效pagination時fail
  closed；`lookup_gid()`只有在兩次獨立明確empty search後才能回報confirmed
  missing。
- 一般pytest與merge gate完全離線。Windows process ownership等平台特有檢查
  可留在CI，但本機可重現的lint、typing、pytest與build不得複製成另一套規則。
- full gate必須執行deterministic pytest、sdist/wheel build，以及從新建wheel
  path import `hbrowser`的 smoke test。
