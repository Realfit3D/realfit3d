# 開發與提交流程

## 維護紀錄

程式修改與遷移歷程記於 [MODIFICATIONS.md](MODIFICATIONS.md)，
資產查核記於 [REMOTE_DISCOVERY_20261020.md](REMOTE_DISCOVERY_20261020.md)。
透明度實作查核與接入分工記於 [OPACITY_TRAINING_REVIEW_20261020.md](OPACITY_TRAINING_REVIEW_20261020.md)。
這些紀錄供開發、測試與回溯使用；README 與功能文件說明使用者可執行的流程。

## Repository 分工

在同一個 VS Code 視窗工作即可：主專案管理 Web／流程程式；`upstream/Gaussian-Garments` 管理 GG 內部修改。寫一次程式，但跨 repo 變更需要各自提交。

## 修改 RealFit3D 自己的程式

在主目錄建立分支、修改、測試，再針對本次檔案提交。這類修改不需要變更 GG commit。

## 修改 GG 內部

Submodule 由固定 commit 還原時通常處於 detached HEAD。先在子目錄建立具名分支，完成修改與測試後，將 commit 推送到 fork。

以下是流程示意，分支名稱與提交訊息需依實際工作選擇：

```bash
cd upstream/Gaussian-Garments
git switch -c realfit3d/feature-name
# 修改並測試後，只加入這次修改的檔案
git add <changed-files>
git commit -m "Describe the GG change"
git push -u origin realfit3d/feature-name
git rev-parse HEAD
cd ../..
```

再把 `upstream.lock.json` 的 `commit` 更新成新的完整 SHA，保留原始 `upstream_baseline`。主目錄執行：

```bash
git add upstream/Gaussian-Garments upstream.lock.json
python scripts/check_checkout.py
# 補上修改說明與測試，再提交主專案的版本引用
git commit -m "Pin GG to the tested RealFit3D revision"
```

先推送 GG commit，再發布主專案引用。需要 review 的變更先在 fork 開 PR，
合併後更新主專案。GG 的內部變更由 fork commit 管理，安裝時無需另套 patch。

## 更新與檢查

其他人取得新版主 repo 後，先用 `git submodule sync --recursive` 同步來源網址，
再用 `git submodule update --init --recursive` 還原指定版本。
`--remote` 用於主動升級依賴，會選取遠端分支版本；升級後需重新測試並更新 lock。

`python scripts/check_checkout.py` 比對：來源設定、gitlink、實際 HEAD、工作樹是否乾淨。檢查 dirty 時會回報失敗以提醒尚未固定的修改，不會刪除或還原任何檔案。

Submodule 使用自己的 `.gitignore`。GG 輸出、模型與本機設定保留在 Git 追蹤範圍外；
提交時以明確檔名加入變更，並分別檢查主 repo 與子 repo 的 staged diff。
