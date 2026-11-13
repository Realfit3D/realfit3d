# 修補衣物的訓練與推論

[English](REPRODUCING_REPAIRED_GARMENTS.md)

Stage 1.1 接收重建 mesh、對應來源人體與 JSON 設定，進行開口重建、平滑、
局部修補與幾何檢查，再產生 UV。審核後的新資產交給 Stage 2 配準，Stage 3
學習外觀，Stage 4 微調動態模型。試穿時的接觸修正另外作用在模擬頂點，
修正後的位置與速度會回饋下一步，最後套用 Stage 3 外觀渲染。

## 準備環境與輸入

使用自己的 checkout 與重建、GG、ContourCraft 私有環境，填寫
[`configs/paths.example.json`](../configs/paths.example.json)。Git 不包含授權的
SMPL-X、RGB、mask、相機、動作及權重；只有 MP4 無法還原訓練輸入。
模型與 pickle 請使用可信來源。

CPU 修補環境已使用 Linux／Python 3.10 測試：

```bash
python3.10 -m venv /your/work/repair-env
/your/work/repair-env/bin/python -m pip install -r requirements/repair.txt
/your/work/repair-env/bin/python -m pip install -e .
```

Open3D 即使只做 CPU 幾何運算仍需 EGL／OpenGL runtime library。
修補依賴版本固定於 `requirements/repair.txt`；`reconstruction_python` 指向此環境。
若也要重建 Stage 1，還需 ActorsHQ 的重建依賴。標定相機修正另見
[Stage 1 指南](STAGE1_CAMERA.zh-TW.md)。

接觸 rollout 的私有 ContourCraft 環境也需 NumPy、SciPy、Open3D。
MP4 使用 MoviePy 2 的頂層 `ImageSequenceClip`，請在私有 GG 環境安裝
`requirements/render-video.txt`，先確認短片編碼成功。這些幾何依賴不會取代
CUDA、PyTorch、rasterizer 與碰撞擴充的安裝。

## 修補與品質檢查

[`configs/repair/`](../configs/repair) 中的 preset 使用該次 capture 的 Y-up、公尺座標。
裁切位置是設定值，不是自動辨識的衣物語意；不同資料可能需要調整。

| Preset | 輸入與處理 |
| --- | --- |
| Actor05 上衣 | 原始四開口上衣；橢圓／ARAP 開口擬合、下襬重建與局部重網格 |
| Actor05 下衣 | 融合後長褲；褲腰／褲腳裁切、保留三開口與局部折返修補 |
| Actor08 上下衣 | 融合 mesh 與來源人體；重建邊帶、局部重網格及人體間距修正；上衣保留立領與彎袖 |
| Actor06 上衣 | mesh、對應女性 SMPL-X 人體與 NPZ 權重；建立人體支撐的無袖拓樸候選後修補開口，遮擋區域是推估表面 |
| Actor06 下衣、Actor07 上下衣 | 既有 mesh 與來源人體；補小洞、限制位移的開口平滑與人體間距修正 |

這些是實驗操作的參數化版本；不同幾何套件版本不保證逐位元相同。
沒有在世界座標強制鏡射彎袖，也不把人體支撐的遮擋區域稱為真實衣物重建。

```bash
python -m realfit3d.quality.repair_template \
  --source /your/input/lower_largest.ply \
  --recipe configs/repair/actor05_lower.json --inspect

python -m realfit3d.quality.repair_template \
  --source /your/input/lower_largest.ply \
  --recipe configs/repair/actor05_lower.json \
  --output /your/work/actor05_lower_candidate
```

需要人體間距的 preset 加上 `--body /your/input/source_body.ply`。
Actor06 上衣另需 `--body-model /your/models/smplx/SMPLX_FEMALE.npz`。
`--reference` 可指定原始上衣供下襬尺寸估計。`--inspect` 只讀；候選資料夾已存在時拒絕覆寫。

輸出包含 `template.obj`、`template_uv.obj`、`uv_diagnostics.npz` 與 `quality.json`。
報告記錄輸入／設定／程式雜湊、套件版本、連通件、開口、非流形、退化面、自交、
三角形品質、尺寸變化與 UV face 對應。失敗時保留診斷報告，不允許安裝。
重網格會改變頂點身份，因此最近頂點距離與真正位移分開記錄。
裁切可能縮短褲子，請查看 `dimension_change_m`。

也可在 paths JSON 加入 `template_repairs`：

```json
{
  "template_repairs": {
    "Actor05_lower_v2": {
      "source": "/your/input/lower_largest.ply",
      "recipe": "repair/actor05_lower.json",
      "output": "/your/work/actor05_lower_candidate"
    }
  }
}
```

路徑相對於該 JSON。執行 `offline repair --subject Actor05_lower_v2` 先看計畫，
加 `--execute` 才會產生候選。

## 交給 Stage 2–4

確認開口與尺寸後，安裝到新的 GG subject；安裝器驗證輸出雜湊並複製相機與
初始化顏色點雲，不覆蓋既有資產：

```bash
python -m realfit3d.quality.install_repaired_template \
  --candidate /your/work/actor05_lower_candidate \
  --destination /your/gg_output/Actor05_lower_v2 \
  --cameras /your/input/cameras.json \
  --point-cloud /your/input/point_cloud.ply --accept-visual-review
```

將對應影像、衣物 mask、`cameras.json` 與 SMPL-X 放在
`<gg_data_root>/Actor05_lower_v2/Sequence1/`。影像、mask、人體與配準 mesh
必須採相同、從零開始的連續影格編號，擷取片段時保留原始影格對照表。
點雲提供初始顏色，需與來源 capture 對應。

依序使用 `offline register-template`、`register-sequence`、`appearance`、`convert`、
`dynamics`，皆帶入相同的 `--subject Actor05_lower_v2 --garment-type lower`；
配準與轉換另帶 `--sequence Sequence1 --frame 0`。完整命令列見
[英文指南](REPRODUCING_REPAIRED_GARMENTS.md)。預設只規劃，加 `--execute` 才執行。
Stage 2/3 保留上游自己的恢復／覆寫行為；先查 checkpoint 與 log，不能因資料夾存在
就判定完成，也不要重跑完成的階段。

Stage 1.1 沒有新增 Stage 2 邊界損失。配準仍可能造成折返或自交，需重新檢查。
`training_frame_rate` 設為實際 capture fps；本輪 Actor 片段為 25。
`validation_frames` 預設 10；`training_updates` 預設 1000、
`training_rollout_steps` 為 8、`training_split_repeat` 為 100、
`training_checkpoint_every` 為 100。

轉換使用指定 checkpoint 做初始放鬆與訓練 restart，下衣在放鬆前設置腰圍 HANDLE，
並驗證索引報告與 UV mesh 雜湊。訓練交替使用配準與 training split 的物理 rollout。
程序內註冊的 `cmu_npz_smplx_preserve` 保留捕捉的姿勢與手部參數，拒絕重取樣及
隨機改身形；因此 dynamics 必須透過本 CLI 啟動。詳見
[訓練轉換](TRAINING_CONVERSION.zh-TW.md)。

Pascal 機器可在私有 ContourCraft checkout 套用
`patches/contourcraft-cpu/cpu-graph.patch`，並設定 `REALFIT_CC_GRAPH_BACKEND=cpu`。
套用前先 `git apply --check`；patch 對應固定版本 `d606432c2f7e96b05664e6a4ffb9d1cb98705baf`。
只替換連通分量計算，碰撞擴充仍需 CUDA。套用後 checkout 檢查會回報預期的 dirty。

## 推論與影片

沿用 `tryon`／`library render`，`checkpoint` 指定預訓練或所選 Stage 4 權重。
paths JSON 加入下列設定啟用新版接觸修正；省略時沿用原推論：

```json
{
  "contact_correction": {
    "schema_version": 1,
    "relaxation_steps": 30,
    "lower_impulse_passes": 2,
    "max_waist_growth_ratio": 1.06,
    "projector": {
      "clearance": 0.003,
      "minimum_clearance": 0.002,
      "max_displacement": 0.035
    }
  }
}
```

此路徑要求兩件衣物依下衣→上衣排列、下衣已有腰圍 HANDLE、使用 SMPL-X 動作與
目標性別模型。流程包括腰圍目標擬合、初始腰部上下衣間距、靜態放鬆、逐步人體
表面投影與下衣 impulse 修正。位置與速度回饋模擬；設定、腰圍及逐步修正量
保存在 trajectory 旁的 `*.contact/`。

整套只載入一份動態權重，各衣物各用自己的 Stage 3 外觀。
若只比較 Stage 4，固定衣物、betas、動作、接觸設定、相機、透明度與外觀權重。
同時更換 mesh 修補與接觸策略，是整條流程的比較。
局部表面接觸限制不保證沒有衣物間碰撞、自交或連續運動中的穿透。

## 測試範圍

`PYTHONPATH=src python -m unittest discover -s tests -v` 執行契約、接觸求解器及
幾何測試；幾何測試需安裝修補環境。網頁播放測試為
`node --test tests/library.test.cjs`。
整合後的 GPU 訓練／rollout 尚需在設定完成的私有環境做 smoke test，才進行下一輪
完整訓練。幾何、碰撞與外觀分開記錄；流程完成不代表畫質通過。
來源與整合狀態見[維護紀錄](maintainer/REPAIR_INTEGRATION_20261113.md)。
