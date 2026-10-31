# Web 部署與 API

## 執行責任

瀏覽器 (`apps/web`) → Flask (`apps/api/server.py`) → `PreviewRuntime` adapter → 專用 ccraft Python 執行保留的 worker → 專用 GG Python 渲染 → PNG／MP4 與 status.json → 瀏覽器輪詢。

Web 環境只安裝 Flask，不 import torch、SMPL-X、Blender 或研究套件。worker 的 subprocess 命令使用參數陣列，不透過 shell；GPU 環境與網站 Python 分開。人體擬合仍同步等待，試衣以背景程序執行。

`scripts/check_checkout.py` 檢查依賴來源與版本。
`workers/contourcraft` 執行身形擬合、衣物適配與模擬。
Adapter 將網頁參數轉為各 worker 的命令，worker 產生的軌跡再交給 GG 渲染。

## GPU 先決條件

`configs/paths.example.json` 所有路徑皆需改成部署機器的絕對路徑：

- `ccraft_python`：已具備 ContourCraft、torch、SMPL-X 等依賴的環境。
- `gaugar_python`：已具備 GG、Blender／渲染依賴的環境。
- `contourcraft_root`：預設使用固定的 ContourCraft submodule；仍需建立其執行環境。
- `gaugar_root`：使用固定的團隊 GG fork，現在已包含 `render_lean.py`、`render_turntable.py`。
- `ccraft_data_root`、`cmu_root`：CC 資料根目錄與 CMU 動作目錄；catalog 的 `CMU/` 前綴會對應到 cmu_root，不再要求把資料放進程式 checkout。
- `gg_data_root`、`gg_output_root`、`gg_aux_root`：GG 輸入、訓練資產、輔助模型目錄，與試衣結果的 output_root 不同。
- `body_model_root`、`reference_motion`、`checkpoint`：有合法使用權的模型、參考動作與 CC 權重。
- `output_root`：既有衣物資產／預熱快取與新結果的根目錄；catalog 的衣物 ID 必須與該目錄及 GG 資產對應。缺少訓練衣物時，先執行資產匯入流程。

GG 先把模擬形狀轉成 UV 遮蔽／法線貼圖，再交給外觀網路和光柵器產生圖片。
`render_lean.py` 只輸出最終合成影格；`render_turntable.py` 固定一幀、
更換相機角度，產生網頁多角度預覽。算法與使用方式見
[GG 渲染說明](https://github.com/Realfit3D/Gaussian-Garments/blob/main/REALFIT3D_CHANGES.zh-TW.md)。

`runtime_entry.py` 在子程序啟動 CC／ActorsHQ 時設定其 DEFAULTS；不修改這兩個官方 checkout。GG fork 讀取 `REALFIT3D_GG_*` 環境變數。API 與 CLI 都透過相同設定傳遞這些值，並把專用 Python 所在目錄放到 PATH，讓上游內部的 `python` 子程序使用相同環境。

啟動檢查會回報缺少的路徑。`catalog.json` 提供衣物與動作設定範例；
部署時需另外準備對應資產、權重與 GPU 環境。

選衣後會檢查該衣物 `stage1/template_uv.obj`、`stage1/cameras.json`、
`stage3/ckpt/net.pt`。部署時將選定的外觀權重安裝到 `ckpt/net.pt`。
fork 另檢查 trajectory 的衣物名稱、UV 頂點總數與人體／衣物幀數。
面索引、UV 對應及座標慣例需要在資產匯入時核對。

## API 與安全界線

| API | 責任 |
|---|---|
| GET `/`、`/garment_thumbnail/<id>` | 頁面與衣物縮圖 |
| POST `/fit_body` | height（公尺）、weight（公斤）、gender；產生人體 |
| GET `/body_status`、`/body_preview.png` | 目前人體與預覽 |
| POST `/discard_body` | 閒置時清除目前人體 |
| POST `/generate` | upper、lower、motion（catalog 索引）；啟動試衣 |
| GET `/status/<id>`、`/latest_frame/<id>` | 程序狀態與渲染進度 |
| GET `/file/<id>/<path>` | 限該次任務中的圖片／影片，拒絕跨目錄與 symlink 逃逸 |

`/generate` 另接受 `opacity_mode`（direct／optical_depth／v3）及
`opacity_scale`（0–1），預設 direct、1。這組參數同時傳入轉盤與動畫渲染。
V3 權重由衣物 catalog 指定，兩件衣物都配置後才開放網頁選項。
設定與訓練流程見 [透明度指南](OPACITY.zh-TW.md)。

鎖與任務表僅在單一程序內有效，不支援多 WSGI worker 或多人服務。無認證、持久化排程、取消或重啟恢復；重啟不會重新認領既有程序，應待 GPU 任務結束後再重啟。body 資料不進 Git；替換／清除會刪除目前人體。程序重啟遺留的人體目錄需自行清理。

新任務不再自動刪除其他試衣結果。請定期檢查 output_root 空間，先確認目錄再清理。Web 不提供任意檔案／pkl 下載，也不接受上傳 pkl。

## 驗證

```bash
python -m unittest discover -s tests -v
python scripts/check_checkout.py
```

測試使用假 runtime，不計算人體或衣物。驗證頁面／縮圖、輸入錯誤、缺少環境、互斥、程序失敗、半寫入 JSON、結果路徑與命令順序。真實 GPU 驗收仍需在同組輸入上比較舊／新流程的 mesh、轉盤、影片、耗時與記憶體。

可選 UI 檢查：另安裝 `playwright` 與其 Chromium，在未設定 GPU runtime 的本機服務運行時執行 `python tests/browser_smoke.py`。可用 `REALFIT3D_BROWSER` 指定已有的 Chromium，`REALFIT3D_URL` 指定服務網址。此測試以 mock 人體狀態檢查選衣頁；截圖存於 `/tmp/realfit3d-*.png`。
