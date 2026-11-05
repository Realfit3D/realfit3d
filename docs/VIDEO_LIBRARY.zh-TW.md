# 預計算影片庫

[English](VIDEO_LIBRARY.md)

## 資料如何流動

工作單列出衣物、身高體重、性別、動作和不透明度。CLI 展開這些組合後，
呼叫與按需試衣共用的 worker：擬合 SMPL-X 人體、準備與解纏衣物、重新綁定、
執行可選的初始化修復、ContourCraft 模擬、GG 外觀渲染，再組成 MP4。
批次模式跳過多角度預覽；確認影格數正確後才發布影片及 JSON 紀錄。

網頁 `/library/` 讀取影片索引與目錄，提供已有影片的選項。
透明度滑桿切換預先渲染的 V3 影片，不在瀏覽器重新運算模型。
原本按需計算的試衣頁仍在 `/`，兩個入口共用 Flask 服務。

## 準備資料與設定

在原本的 `configs/paths.local.json` 增加 `package_root`（資料包）與
`library_root`（影片庫）。GG／CC Python、checkpoint、人體模型和資料根目錄
沿用 `configs/paths.example.json` 的欄位。相對路徑以設定檔所在目錄為準。
`gg_output_root`、`output_root`、`library_root` 應使用資料包以外的專用可寫目錄。
不需要把 overlay 蓋進 GG／CC checkout。

資料包內容：

```text
catalog.json
motions/<動作檔>.npz
thumbnails/<衣物ID>.png
garments/<衣物ID>/
  stage1/template_uv.obj
  stage1/cameras.json
  stage3/ckpt/net.pt
  stage3/ckpt/info.txt
  cache/garment_dict.pkl
  cache/source_body_fit.pkl
  opacity_controller_v3/controller.pt
```

`catalog.json` 的 `garments`、`motions` 是以 ID 為 key 的物件。
衣物項目提供 `slot`、`source_gender`、`label`；動作提供相對於 `motions/` 的
`file` 與 `label`。可選的身高體重區間使用 `{id, label, value}`，工作單必須使用
區間代表值。批次身高單位為 **cm**，worker 接收前轉成公尺。
此目錄格式與按需試衣的 `configs/catalog.json` 不同。

先前保存的原始交付 ZIP 不等於上述執行資料包：可能缺少身形擬合快取及動作檔，
不能僅替換下載網址就開始模擬。只載入可信來源的 pickle 與 checkpoint。

## 執行

將 `configs/library-jobs.example.json` 的衣物與動作 ID 改成資料包內的項目。
建議先選一套衣物、一種身形與短動作。

```bash
realfit3d --config configs/paths.local.json library render \
  --jobs configs/library-jobs.example.json

realfit3d --config configs/paths.local.json library render \
  --jobs configs/library-jobs.example.json --execute

realfit3d --config configs/paths.local.json library index --execute

REALFIT3D_CONFIG=configs/paths.local.json python -m apps.api.server
```

第一條只檢查輸入、使用 CC Python 讀取動作長度、計算依賴雜湊與列出輸出；
不寫入資產、不擬合、不模擬、不渲染。加入 `--execute` 才執行。
索引完成後開啟 `http://127.0.0.1:5000/library/`。
已有相容影片庫時，可以只建立索引，不必重新產片。

程式將必要模型複製到 `gg_output_root`；若既有模型內容不同就拒絕覆蓋。
可寫衣物快取使用副本，不連回原始資料。
每次執行資料留在 `output_root/precompute/<fingerprint>/`，保留 trajectory、
修復前後字典、檢查報告與影格，方便逐段檢查。
相同輸入及程式版本可重用身形擬合與解纏結果；這不是身高插值，新動作仍需模擬。

重跑工作單時，參數和依賴雜湊相符的完成影片會跳過；失敗項目保留診斷資料，
用新執行目錄重試。影片庫有批次寫入鎖，但沒有跨程序 GPU 排程，
請勿同時執行批次產片和按需試衣的 GPU 工作。

## 可選衣物初始化修復

在 paths 設定檔加入：

```json
{
  "garment_repairs": {
    "Actor08_shirt": {
      "reference_dict": "/data/references/Actor08_shirt.pkl",
      "profile": "repair_profiles/upper.example.json",
      "genders": ["male", "female"]
    }
  }
}
```

修復輸入是重新綁定後的衣物字典，以及同拓撲、相同來源身形的參考衣物。
先以 SMPL-X 蒙皮權重識別手臂或腿部區域，再依參考衣物恢復縮短等損壞的初始形狀，
用依拓撲求解的調和過渡平滑連接相鄰區域，並重新轉移人體綁定權重。
通過位移、邊長變化、朝向及設定的間隙門檻後，才將候選交給模擬；未通過則停止該影片。

修改只作用在本次執行的初始化／綁定字典，保留原始 rest geometry、面索引、
UV、Stage 3 外觀權重及來源快取。修復前字典與 JSON 報告會一起保存。
預設不修復，必須逐件指定衣物及目標性別；範例 profile 不是所有衣物通用的核准參數。
同一設定也會套用到原本按需試衣的 Web／CLI。

## 透明度

工作單的 `opacity_garments` 指定需要變透明的衣物，省略代表整套。
V3 要求每件指定衣物都有自己的 controller，端點值也一樣；未選中的衣物維持原樣。
影片選項會標示透明度作用的衣物，避免混淆「只有上衣」與「整套」。

GG 先載入 Stage 3 權重，再保存參考 Gaussian 座標；動畫更新 mesh 時不改這份副本。
V3 用固定參考座標及 checkpoint 的正規化參數計算 opacity，實際渲染位置仍隨模擬變形。

整合已涵蓋 CPU 命令、檔案隔離、API 與瀏覽器測試；
修復衣物與 V3 組合後的完整 GPU 流程仍需用實際資產驗收。
