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

- 除使用者在目前任務明確指定限制外，設計時不以最小修改或向後相容
  作為限制；在已授權任務範圍內以程式碼品質優先，選擇架構、可讀性與
  可測試性最好的完整結果。
- 開始實作時須說明上述設計原則；完成時須如實說明是否遵守，以及具體
  的設計取捨。若實際受到最小修改或向後相容限制，須揭露限制來源及
  影響，不得宣稱未受限制。
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
- Primary 不得 rebase；`pre-rebase` hook 必須拒絕目前或明確指定的 primary。
  `scripts/install-git-hooks.sh` 設定 `pull.rebase=false`、primary 的
  `branch.<primary>.rebase=false` 與 `pull.ff=only`；pull 只可 fast-forward，
  分歧時停止並明確處理。Task 整合仍使用 `--no-ff` 保留 merge commit。
- 任務完成後從 task branch 執行 `scripts/git-flow-merge.sh`。若 task branch
  含有 primary 尚未包含的 commit，該腳本先準備真正的
  `--no-ff --no-commit` candidate，依範圍執行 code review 與文件或完整 gate，
  成功才建立 merge commit、安全移除 task worktree，並以 `git branch -d`
  刪除已合併的本機 branch。Review 失敗依下述 cleanup／abort 規則處理。
- 若且唯若 task tip 已由本機 primary 包含（`git merge-base --is-ancestor`
  成功），`scripts/git-flow-merge.sh` 執行 no-op cleanup：不得執行 gate或merge、
  建立空 commit或空 merge commit；必須先確認涉及的 worktree clean、沒有進行中
  的 Git operation、task與primary refs仍為已驗證的commit，再切回 primary或安全
  移除 task worktree，最後以 `git branch -d` 刪除本機 task branch。
- 任務取消時只可使用 `scripts/git-flow-merge.sh --cleanup-only`；若 task tip
  未由 primary 包含，或任何 safety check失敗，腳本必須 fail closed並保留 task
  branch及仍存在的worktree。no-op cleanup不得使用force、建立或刪除remote ref，
  亦不得 fetch、pull或push。
- merge conflict 或 gate failure 時，確認 owned review／hook process tree 已清空後
  必須 abort merge 並保留 task branch。若無法證明 owned process cleanup，這是明確
  例外：不得讓仍存活的 hook 與 abort 競爭，保留 candidate 與 task 待確認及安全處理。
  已成功建立的 merge commit 不因後續中斷或清理失敗而回滾。
- Canonical POSIX nested owner 必須在啟動新 session 前登記清理責任；外層先關閉
  admission，只有內層證明自身 group 與所有 nested obligations 已清空，才能
  強制終止該 owner 或回報整棵樹清空。Lock 等待共用原 deadline，不以延長 grace
  取代 ownership 證據。內層非預期死亡或 ledger 不可驗證時保留 candidate、task
  與 scope，不能把外層 group 清空當作巢狀清理成功。
- Merge workflow 的 checkout、merge、failure recovery 與 task branch deletion
  統一由 canonical process owner 管理 Git hooks。包括已合併 task 的 no-op／
  `--cleanup-only` 都需要 repository-local `.venv`；缺少時先執行
  `scripts/rebuild-env.sh`。Linked worktree 移除前，必須先確認並改用 primary
  checkout 的 `.venv` 完成剩餘清理；缺少時保留 task worktree 與 branch。
  正常操作可由外層中斷；failure recovery 的每次 abort 或 checkout 各有
  10 秒 hard budget（含 6 秒 teardown reserve），abort 失敗不得繼續 checkout。
  `--cleanup-only` 仍可安全重試清理，不重跑已完成 merge 的 review 或 gate。
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

## 共用開發工具

- `scripts/_shared/h2h_dev_tools/` 是獨立 `h2h-dev-tools` repository 的 generated
  source snapshot；其中 Python source 與 `manifest.json` 不得手改。
- 更新須從明確選定的 toolkit version 或 candidate wheel 執行
  `h2h-dev-tools sync --repository <path>`，審查完整差異後提交。不得讓一般
  checkout、hooks 或 CI 依賴固定 sibling 路徑或自動取得最新版。
- `check-fast.sh` 必須離線驗證 snapshot 的完整檔案集合與雜湊，再執行原有
  品質檢查。格式修正入口略過 generated snapshot；共用 source 的修改與
  格式化必須回到 toolkit repository，再重新產生。
- Repository 的版本、audit、release gate、Git lifecycle 與測試選擇政策仍由
  本 repository 擁有；shared mechanisms 不取代這些政策。

- `[tool.h2h-dev-tools].components` 明列本 repo 使用的共用機制；快照使用
  schema 2，必須驗證選擇、完整依賴檔案與雜湊。舊 schema 1 僅用明確的
  `h2h-dev-tools migrate-v1 --repository` 一次性轉換，不在 hooks 保留 fallback。
- Git、version、audit、環境、formal 執行與 process ownership 的通用機制可由
  toolkit 擁有；release paths、review/gate 入口、pytest selectors、timeout
  budgets、formal targets 與領域驗收仍由本 repo 宣告並驗證。

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
  markdownlint；非純文件的非 merge commit 執行。
- `scripts/check-full.sh`：fast gate、完整測試、適用時的 build與wheel smoke及本
  repository的特殊檢查；非純文件的整合候選只跑一次。手動呼叫 fast／full
  入口仍執行原本完整內容，不自行縮減檢查。
- `scripts/check_change_scope.py --index --base HEAD` 以 Git tree 與 staged
  index 判定 `documentation` 或 `full`。只有 `README.md`、`docs/**/*.md`、
  `benchmarks/README.md`、`verification/README.md` 的普通非執行檔
  （Git mode `100644`）新增、修改或刪除可視為純文件；rename 的兩端都須符合。
  `AGENTS.md`、`CLAUDE.md`、程式、tests、scripts、hooks、設定、metadata、
  symlink、mode 變更及未知路徑一律走原有檢查。空差異走 `full`，分類失敗則
  阻止提交或合併，不得猜測為文件。
- `pre-commit` 對純文件執行
  `.venv/bin/python scripts/check-docs.py --index --base HEAD`，其餘執行 fast；
  `pre-merge-commit` 保留 staged version check，純文件執行同一文件檢查，
  其餘執行 full。合併分類涵蓋 primary parent 到 staged candidate 的完整差異，
  不只看最後一個 commit。文件檢查從 exact staged tree 暫存匯出所有普通
  Markdown 文件與設定，執行 repository-local Markdown lint 及差異 whitespace
  check，並從 candidate metadata 驗證 `project.readme` 引用存在的普通檔案；
  inline text 不需外部文件。檢查不讀取未 stage 的內容，也不執行 Ruff、mypy、
  pytest、build 或 online review。
- Repository 提供的 pytest child 統一使用 `scripts/pytest-source.py`，先載入
  經驗證的 source snapshot，再啟動 pytest、plugins 與 workers；沿用原有
  selectors、Git environment boundary 與 process ownership。手動需要相同來源
  保證時使用 `.venv/bin/python scripts/pytest-source.py <pytest arguments>`；
  普通 `pytest` 僅為手動便利入口，不承諾排除既有 bytecode。
- dependency audit可連網，但 hooks只驗證本機 receipt，不在 commit過程
  連網。
- GitHub Actions只呼叫相同 scripts，並保留 trusted publishing、平台特有
  或本機無法可靠重現的檢查。
- 不使用 Claude、Codex或其他 provider-specific Stop hooks重複檢查。

## Code Review Rules

- 每次 task 合併先由 `scripts/git-flow-merge.sh` 準備真正的 two-parent
  `--no-ff --no-commit` candidate，再依 primary parent 到 staged candidate
  的完整差異分類。Full profile 在 Git hooks 之外呼叫
  `scripts/review-code.py run --index`；documentation profile 不啟動 review。
- 首次 review 必須審查完整 candidate project，建立成功的 full baseline；
  之後才可 incremental review。增量審查必須證明 primary first-parent history
  上存在有效 full baseline，且其後每個非文件變更都有有效 review evidence。
  缺少 baseline 或任何 coverage、舊 v1 receipt、政策或 reviewer dependency
  closure 變更，均重新執行 full review，不得只審查目前 task diff。
- v2 receipt 必須綁定 full／incremental scope 與可驗證的 baseline／history
  coverage；不能以同名檔案、未驗證舊紀錄或單純 ancestor 關係代替完整覆蓋。
  Passing full candidate 可立即 offline verify；只有其 exact tree 與 ordered
  parents 真正成為 primary first-parent commit，才可作後續 baseline。
  Aborted candidate、側分支與舊版差異 receipt 都不構成基準；新 clone 缺少
  本機 evidence 時重新 full review，offline verify 不會連線補做。
- Online review 可使用已登入的 Codex 服務，需要 repository-local `.venv`
  與 PATH 中的 `codex`。使用 read-only sandbox；不得修改檔案、執行
  merge/gate、啟動服務或自動修正，不要求 GitHub PR、Actions 或 Stop hook。
- Review 使用一般 Codex CLI session，保留 CLI rollout，讓審查子代理可以
  找到 parent thread；不得使用 `--ephemeral`，亦不得自動刪除 CLI 紀錄。
  Read-only sandbox、完整審查覆蓋、approval policy 與 owned cleanup 契約
  維持不變。
- Review 的單次 Codex invocation 固定 `notify=[]`，防止非同步完成通知在
  CLI 退出後留下子程序。保留既有模型、認證、其他工具、plugins、MCP 與安全
  hooks 設定，不修改全域配置；其他 descendants 仍須符合相同 cleanup 契約。
- Reviewer 必須完整閱讀本檔；首次審查完整 candidate，包括既存缺陷，
  incremental 審查完整差異與相關未修改程式碼的互動，依本 repository 的架構、ownership、side-effect 與測試契約評估。
  只回報具體觸發條件及影響的 P0、P1、P2，附檔案、行號與理由；格式交由 lint。
  已授權的破壞性變更本身不是缺陷，但版本誤判或遺漏必要轉換仍須回報。
- 任何 finding、`incomplete`、逾時、非零退出或無效輸出都阻止 merge。
  上下文或工具不足須回報 `incomplete`，不得聲稱通過。失敗須先確認 owned
  process cleanup 再 abort；無法證明 cleanup 時依前述例外保留 candidate。
  修正後重新執行 merge 流程，不得忽略 finding 或手寫 passing receipt。
- Online review 目前只支援 POSIX，以獨立 process group 清理逾時或中斷的
  Codex 及一般 descendants；刻意脫離 group 者不在保證內。Windows 仍可
  offline verify；online review 必須先實作並驗證 Job Object 生命週期。
  預設 timeout 為 900 秒，`run --timeout-seconds` 接受 1..3600 秒；merge 可用
  `H2H_CODE_REVIEW_TIMEOUT_SECONDS` 明確傳入同一上限。
- Merge review stage 先以獨立 owned process group 完成分類，確認清空後，
  只有 full profile 才要求 local review 入口、取得 candidate lock 並啟動 reviewer；
  純文件不要求 local review 入口或 Codex。CLI version preflight、reviewer 與
  Codex 由外部 owner 管理同一 process group，owner 使用真正 staged candidate
  匯出的完整 review closure，並持有 candidate lock 至清理及 evidence publication
  完成。Managed reviewer 只寫 pending evidence；owner 證明 owned tree 已清空、
  重新驗證 exact candidate 與 coverage 後才原子發布，缺少 pending evidence
  亦須阻止合併，不能以 reviewer exit 0 取代證據。
- `review-code.py verify` 僅離線驗證，絕不呼叫模型。紀錄位於 Git common
  metadata 的 `h2h-review`，綁定 exact tree、ordered parents、`AGENTS.md`、
  local CLI 與 merge review 入口，以及 shared reviewer、merge review shell、
  process ownership、change-scope 與完整匯出 dependency closure 的 candidate
  及實際執行 bytes；
  任一變更須重審，不能沿用舊 full baseline。
  `--revision` 可審查既有 commit，正式 merge 必須使用 `--index`。
- 同一 candidate 的每次 `run` 先廢止舊紀錄，失敗不能沿用舊通過結果。
  同一 candidate 只允許一個 active run；重疊請求必須在啟動 review 前拒絕，
  不得取代正在執行者；POSIX offline verify 在 review 執行中也須拒絕。
- Pending merge 的 `pre-commit` 委派 `pre-merge-commit`，避免以手動
  `git commit` 繞過 gate。Full gate 前後均離線驗證 review evidence，並以
  `--expected-tree` 綁定開始檢查的 staged tree；Git hooks 不啟動 online review。
  Documentation profile 保留版本及文件檢查，不能重用為程式驗收。
- Code review evidence 與完整測試結果是不同證據。AI 未發現問題不代表程式
  正確性的證明，本機紀錄也不是防竄改簽章。原始輸出、診斷、CLI 版本與 model
  override 保留於 metadata；未指定 model 時沿用 Codex 設定，不聲稱知道 CLI
  未回報的解析後模型。

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

- 明確說明本次是否在不考慮最小修改、也不以向後相容為限制的條件下，
  遵守程式碼品質優先要求，並提供具體依據。
- 明確判定本次是「不向後相容」或「品質優先後恰好向後相容」，說明受
  影響的介面、行為或資料格式及判定依據；不得只回覆「相容修正」或
  版本分類。若使用者明確要求保留相容，須如實說明此限制，不得稱為
  「恰好相容」。純文件或政策變更須說明沒有執行期介面或資料格式變更。
- 若不向後相容，進一步明確判定「可提供一次性相容性轉換工具」或
  「無法提供一次性相容性轉換工具」，並說明理由。尚未確認可行性時，
  須明示待確認事項，不得把未調查視為不可行。
- 轉換工具可行時，分開回報「可提供」與「已提供」；在已授權範圍內提供
  並驗證工具，附位置、使用方式、適用範圍及限制。尚未提供時須說明
  原因與待辦，不得宣稱相容性問題已解決。
- 無法提供轉換工具時，列出資料影響與確切的人工處理、刪除或重建範圍，
  並明示可保留的項目；不得只籠統要求「全部重建」。
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
- 非純文件變更的 full gate 必須執行 deterministic pytest、sdist/wheel build，
  以及從新建 wheel path import `hbrowser` 的 smoke test。
