# 修改與遷移紀錄

維護紀錄：供版本整合、回溯與測試使用。匯入版本與原始檔案雜湊保留於
`source_manifest.json`；使用方式與模組流程見 README 及 GG 差異說明。

## 2026-11-05：影片預計算與初始化修復

- 新增 `library render`：從工作單展開衣物、身形、動作及透明度組合，呼叫既有
  CC worker 與 GG renderer，產出 MP4 和逐片執行紀錄。預設只檢查輸入並列出計畫。
- 批次以程式、模型、動作及修復設定的雜湊區分工作目錄；相同依賴可重用人體擬合、
  衣物準備快取和完成影片。輸入資產只讀，修改型快取複製到獨立輸出目錄。
- 新增 `/library/`：依影片索引提供已有組合與 V3 透明度影片，原 `/` 按需試衣保留。
  API 支援 MP4 range request，只公開索引、縮圖與影片，不公開執行日誌和模型。
- 共用 worker 在 rebind 後接入可選的參考衣物修復：以人體蒙皮權重辨識區域，
  以參考形狀與 harmonic transition 修正初始化，再通過幾何品質檢查才進入模擬。
  失敗時停止該工作並保留報告；未指定修復設定時沿用既有流程。
- GG V3 固定使用 Stage 3 checkpoint 載入後的參考 Gaussian 座標，避免動作改變
  controller 的空間輸入。新增指定衣物透明度，未選衣物保持原始 opacity。
- 本機 CPU 驗證：主專案 52 項、GG 25 項通過；幾何 21 項通過，1 項因缺少
  Open3D 跳過。Node 選項測試及桌面／手機瀏覽器播放測試通過。
  瀏覽器使用既有範例影片；本次未啟動 GPU，也未更新伺服器 checkout。
- 使用流程見 [VIDEO_LIBRARY.md](VIDEO_LIBRARY.md) 及
  [繁中指南](VIDEO_LIBRARY.zh-TW.md)；匯入版本和檔案雜湊見 `source_manifest.json`。

## 2026-10-31：組織轉移與文件連結

- RealFit3D 與 GG fork 轉移至 GitHub `Realfit3D` 組織；更新 clone 網址、
  `.gitmodules` 與版本鎖的 repository 來源，保留原有提交歷史和上游基線。
- README、Web 與中英文透明度指南直接連到 GG fork 的文件，修正 GitHub
  無法跨 submodule 解析相對路徑造成的 404。GG 文件中的主專案連結同步更新。
- 新增文件連結測試，檢查跨 submodule 路徑與所連結的 GG 文件是否存在。
  本次只調整來源設定、文件與測試，不修改模型或訓練算法。

## 2026-10-21：Stage 2 補件與狀態更新

- GG fork `5c03d7e` 新增 `opacity_training_cameras.py`：以完整校正保留主點與內外參，
  以 Stage 2 清單限定視角，核對尺寸、焦距與姿態後建立 V3 訓練相機。
  校正含影像尺寸時不讀原始圖片；舊的 PNG／JPEG 輸入仍可使用。
- RealFit3D 的 `offline opacity` 前置路徑檢查同步接受不含原始影像的配置。
  訓練設定記錄校正與視角清單雜湊，方便重現相機選擇。
- 本機通過 23 項 fork 測試與 36 項 RealFit3D 測試；伺服器 Actor02 上下衣
  各 60 個相機通過 CPU 資料檢查。未執行 GPU 重訓或修改既有 controller。
- 逐檔核對交付清單的 1,527 個 Stage 2 檔案，大小與 SHA256 全部一致。
- 同步英文／繁中透明度指南及整合狀態：mesh、AO／normal、點雲已補交，
  Actor02 完整相機字典已接入訓練讀取器。
- `doctor` 改為回報已整合的透明度模式，明確保留每件衣物資產尚未檢查的範圍；
  不再固定顯示 V3 程式與權重未交付。新增 CPU 回歸測試。
- 未修改 GG 的透明度算法、載入模型或執行 GPU 訓練；
  查核結果見 [透明度接入紀錄](OPACITY_INTEGRATION_20261020.md)。

## 2026-10-20：透明度接入

- fork 加入相容既有 V3 權重的 controller；Doll 在 Gaussian 光柵化前套用透明度。
- lean、turntable、原始 inference 共用 original／direct／optical_depth／v3。
- 新增參數化的離線 V3 訓練入口，固定 Stage 3，產出控制器及驗證資料，不覆寫既有目錄。
- RealFit3D 串接 catalog 權重、Web／CLI 參數、轉盤及影片命令，並新增 `offline opacity`。
- Actor02 既有權重已完成 GPU 渲染測試；完整重訓與 Web 全流程驗收另列於
  [接入紀錄](OPACITY_INTEGRATION_20261020.md)。

## 2026-10-20：文件與遠端來源查核（接入前）

- 發布首輪整合，固定版本記錄於 `upstream.lock.json`。
- 在遠端交付定位 V3 程式與 Actor02 controller 檔案，記錄路徑與 SHA256；
  目前仍使用原版 S3，V3 接入排在來源比對後。
- README 統一為英文入口；GG 差異文件提供英文與繁中版本。
- 整理文件、docstring 與註解，移除對話紀錄和主觀評語；修正 CLI help 的舊預設值說明。
  執行參數、算法與輸出格式維持不變。

## 2026-10-15：第二階段整合

- GG fork `8a52dc6648c587450aaf04cb7d560099d82116c0`：移入交接 lean／turntable，增加三個資料根目錄環境變數及 trajectory 基本契約檢查。官方基線未變；詳細差異見 fork 的 REALFIT3D_CHANGES.md。
- 加入固定版 CC、ActorsHQ 與其 GS2Mesh；使用 runtime_entry.py 在子程序內更新 DEFAULTS，沒有直接修改這兩個官方 repo。
- Web／CLI 共用 body／選衣驗證及試衣命令。CMU 路徑改用配置，pipeline 子程序也套用 CC runtime defaults；pipeline 中兩個 data-root 參數不再硬接 checkout/ccraft_data。其餘交接 worker 算法保留。
- 新增各離線階段 dry-run／明確執行入口、GauGarConverter CLI bridge、doctor、資產存在性／SHA256 檢查。
- 移入 8 個 backend result_optimization 工具，未改其算法或閾值。未自動把白名單接入上游 TSDF，未啟用實驗性 pose transfer。
- 本階段驗證採用 CPU 測試與瀏覽器檢查；GPU 部署與驗收流程見 INTEGRATION_STATUS.md。

## 2026-10-15：初始化

- 建立主專案來源與 submodule 連結。
- GG 固定至 `ceb66fc36598da05b3b3391f4562cdc54d2a2ed2`，未修改上游程式。
- 新增唯讀 checkout 檢查與 CPU 測試。
- 未建立新模型、未移入 Web、未重跑 GPU 推論或訓練。

## 第一階段 Web 整合紀錄

以下記錄第一階段的移入內容；GG 渲染入口與離線 CLI 已在上方的第二階段接入。

- 整合小組的 Web、試衣 worker 與衣物縮圖。
- `ContourCraft/preview_server.py` 的 HTML／CSS／JS 拆至 `apps/web/`；API 重組至 `apps/api/server.py`，GPU 命令集中至 `src/realfit3d/adapters/preview.py`。
- 同目錄其餘 13 個 Python worker 原樣搬至 `src/realfit3d/workers/contourcraft/`，不是新增演算法。衣物縮圖原樣沿用 `ContourCraft/catalog_thumbnails/`；預設 catalog 從 server 常數抽出。
- 改變行為：以設定檔指定 Python 與資料路徑；原子鎖保護單 GPU 操作；拒絕錯誤 motion index／衣物槽位；監測程序退出；讀取半寫入 status 時重試；限制結果下載為圖片／影片；不再自動清除舊試衣結果；fit 失敗保留原人體；補手機排版。
- 保留：人體先建立一次再選衣、下身先於上身、fixed-size／pin 參數、站姿轉盤與動畫順序。進度權重仍為交接估計，不是本次實測效能。
- 第一階段以設定指定已部署的 GG runtime；第二階段將渲染入口與輸入檢查移入 fork。
- 測試涵蓋 CPU API 與瀏覽器流程。真實模型的 GPU 驗收待執行。

## 模組責任與後續工作

| 功能 | 目標責任 | 驗收前的注意事項 |
|---|---|---|
| Web 與試衣流程 | RealFit3D 主專案 | 第一階段拆分完成，待實際 GPU 產物 parity 驗證 |
| 體型擬合、資產準備、fixed-size | 本組模組＋CC 適配 | 合併兩份 fixed-size 前比較 pin 等實際行為 |
| LBS／ICP 初始化 | 先保留獨立前處理 | 交付腳本在無合格 ICP 候選時的 fallback 與 PDF 描述不同，需對齊 |
| GG 精簡渲染／烘焙設定 | GG fork 或本組渲染入口 | 保留 baseline 與加速設定，不將低採樣當成同品質 |
| 透明度 | controller 訓練＋GG renderer＋Web／CLI | Actor02 兩件衣物渲染已驗證；完整重訓及重疊多層材質案例待驗證 |
| 固定選項快取 | RealFit3D 主專案 | 批次依賴指紋、身形／衣物快取與影片重用已實作；鄰近身形插值尚未實作 |

修改紀錄欄位：來源 repo／檔案／commit、修改目的、行為變更、測試與已知限制。
