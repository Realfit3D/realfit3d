# 配準序列轉為模擬訓練資料

[English](TRAINING_CONVERSION.md)

`offline convert` 讀取一段已配準衣物 mesh 及對應的 SMPL-X 姿勢檔。
我們的 worker 擴充 ContourCraft 的 `utils/gaugar.py`：先整理衣物形狀與
人體動作，再將人體的蒙皮權重轉移到衣物，使用設定的預訓練模擬器鬆弛
還原姿勢後的衣物。產生的 garment dictionary、動作陣列、配準監督資料
與 YAML，會交給 `offline dynamics` 訓練。

沿用來源為
[Dolorousrtur/ContourCraft 的 utils/gaugar.py](https://github.com/Dolorousrtur/ContourCraft/blob/d606432c2f7e96b05664e6a4ffb9d1cb98705baf/utils/gaugar.py)，
固定 revision 為 `d606432c2f7e96b05664e6a4ffb9d1cb98705baf`。

worker 只轉換指定的 sequence，不把 Stage 2 的 `Template/` 當成動作。
45 維手部軸角參數會完整保留；6 維 PCA 參數則用所選人體模型的手部基底
展開。衣物與人體的影格編號必須一致、從零連續編號。若已有 Stage 4 輸出
或同名訓練設定，轉換會停止，避免覆寫。

在部署 JSON 中，`training_frame_rate` 填原始拍攝幀率，
`validation_frames` 指定保留末尾幾幀做驗證；預設分別為 30 fps 與 10 幀。
訓練與驗證各至少需要三幀，目前 ContourCraft 的資料設定只接受整數 fps。
這種切分檢查的是同一動作未參與訓練的尾段，不是不同人體或新動作的泛化能力。

輸出位置：

- `<gg_output_root>/<subject>/stage4/`：完整動作與 mesh 陣列，以及對應的
  `_train`、`_valid` 分段。
- `<ccraft_data_root>/aux_data/garment_dicts/finetune/`：衣物 dictionary。
- `<ccraft_data_root>/aux_data/datasplits/finetuning/<subject>/`：資料切分 CSV。
- `<contourcraft_root>/configs/finetune/<subject>.yaml`：訓練設定。

ContourCraft 讀取產生的設定後才開始訓練。動作取樣、rollout 長度、更新次數
與材質設定決定實驗內容；資料轉換本身不會訓練動態模型。

下衣請使用 `--garment-type lower`：worker 會從 `stage1/repair_quality.json` 讀取腰圍索引，
驗證 UV mesh 雜湊及完整邊界迴圈，並在初始放鬆前傳入 HANDLE。
設定會產生 training split 專用的物理 rollout loader；預設 1000 次更新、8 步 rollout、
每 100 次保存 checkpoint。`training_split_repeat` 預設 100。
`cmu_npz_smplx_preserve` 在 worker 程序中註冊，保留來源姿勢，拒絕改動身形或重取樣。
請透過 `offline dynamics` 使用它，不直接執行未註冊 loader 的上游 `train.py`。
完整設定與修補銜接見[重現指南](REPRODUCING_REPAIRED_GARMENTS.zh-TW.md)。
