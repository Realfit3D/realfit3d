# Stage 1 標定相機

Stage 1 讀取多視角照片與相機標定，先訓練靜態 Gaussian 場景，再渲染雙目圖片，
由 DLNR 估計深度，將衣物遮罩內的深度融合成 TSDF 表面。表面經過重新網格化，
成為後續配準使用的衣物模板。

標定投影選項將 COLMAP 的每台相機主點 `cx`、`cy` 傳入 Gaussian 訓練與雙目渲染，
讓渲染的像素位置與 TSDF 使用的標定內參一致。修正包含 rasterizer 的半像素座標換算，
以及圖片縮放時的主點縮放；不改變相機姿態或焦距。

## 安裝

新重建實驗使用獨立 checkout。安裝程式先檢查兩個固定版本的 repository，才套用修正；
重複執行不會重複套用。若本機修改與 patch 衝突，或基準 commit 不符，會停止安裝。
套用後 Git 會顯示本機修改，原始 checkout 檢查也會回報 dirty。
這組 patch 另由安裝程式的 dry run 與記錄的程式雜湊驗證。

```bash
python scripts/apply_stage1_camera_patch.py \
  --actorshq-root upstream/ActorsHQ-for-Gaussian-Garments
python scripts/apply_stage1_camera_patch.py \
  --actorshq-root upstream/ActorsHQ-for-Gaussian-Garments --apply
```

在現有執行環境 JSON 設定中加入：

```json
{"stage1_projection": "calibrated"}
```

接著使用既有的 `offline initialize`、`offline reconstruct` 指令。
重建計畫會加入 `--GS_calibrated_principal_point`，並在執行前驗證已安裝程式的雜湊。
Gaussian 與雙目中間產物的目錄名稱會帶有 `_calibrated`。
比較實驗仍須設定不同的 `gg_output_root`，因為最終 `stage1/template.obj`
仍依 subject 與輸出根目錄決定位置。

## 權重相容性

既有流程預設維持 `legacy`。標定模式需要重新訓練對應的靜態 Gaussian 權重，
不能只將舊權重改用新相機渲染。渲染器會檢查 `cfg_args` 記錄的投影模式，不一致時停止。
這個選項處理 Stage 1 的靜態 Gaussian 場景，不是 Stage 3 外觀權重或透明度控制器。

此 patch 不修改分割、遮罩侵蝕、TSDF voxel 大小或重新網格化。
渲染變清楚後，仍須檢查 mesh 邊界與拓撲。

## 程式來源

- [ActorsHQ-for-Gaussian-Garments](https://github.com/hlimach/ActorsHQ-for-Gaussian-Garments)，
  commit `46651f55030407ea25463ed070eed9f7e837a9af` 的 `mesh_reconstruction.py`：
  將標定模式參數傳入 Gaussian 訓練。
- 內嵌 GS2Mesh checkout，commit `b08a68f2b896301472d25fe3f5287b5b1bf1b737`：
  `third_party/gaussian-splatting/{arguments,scene,utils}` 傳遞主點；
  `gs2mesh_utils/{argument_utils,renderer_utils,eval_utils}.py`
  處理參數、權重模式驗證及不同的輸出名稱。

完整修改見 [patch bundle](../patches/stage1-camera/)。
實測影像改善與剩餘 mesh 問題見 [驗證紀錄](maintainer/STAGE1_CAMERA_20261110.md)。
