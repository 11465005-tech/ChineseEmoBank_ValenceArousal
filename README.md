# ChineseEmoBank_ValenceArousal
 
以 Pipelined Neural Networks（NNword → NNmod → NNsent）預測中文詞、短語、句子的 Valence-Arousal（CVAW / CVAP / CVAS），詞向量來源為 `hfl/chinese-bert-wwm-ext`。
 
## v8-UniCtx2 更新說明：句子層 NNsent 統一語境、NNword 微調層、jieba 斷詞
 
v8-UniCtx2 是在 v8-UniCtx 上依實驗檢討所做的修改版（程式：`chinese_pipelined_bert_pipeline_v8-UniCtx2.py` 與 Colab 版 `.ipynb`）。v8 系列共有三個獨立版本：
 
| 版本 | 內容 |
|---|---|
| `v8-Simple` | 純簡化 NNsent：不再區分有／無修飾詞（`hit=1` / `hit=0`）兩種路徑 |
| `v8-UniCtx` | 簡化版 + 向量鄰近度 kNN 特徵與 gate + OOV span |
| `v8-UniCtx2` | 本次版本：微調層 NNword、移除 kNN、jieba 斷詞、加強正則化 |
 
所有 v8 版本共通的設計：每個 span 都走同一條路徑——`NNword`（有修飾詞再接 `NNmod` 遞迴鏈）取得先驗 → `BERT(span [SEP] 整句)` 取語境 → 同一個 adapter → 同一個遞迴 `NNsent` step，`hit` 只是一個輸入特徵；整句語境 `max_length=128`（v7 的「整句向量」只用 16，實際只看到句子前約 14 字）。
 
### 改動一：NNword 加入微調層（Stage 1）
- 在 BERT `[CLS]` 向量與 VA 預測頭之間加一層殘差 adapter：`LayerNorm → Linear(768→256) → ReLU → Dropout → Linear(256→768)`，上投影層零初始化，起點等於原本的 v7-2。
- 正則化：dropout（輸入端 0.2、adapter 內 0.2、預測頭 0.3）、L2（AdamW weight decay 0.05）、L1（1e-6，只加在權重矩陣）。
- 從訓練集再切 15% 當驗證集，用驗證 loss 選最佳 epoch，並輸出 train / val / test 的 Pearson r 與落差。
### 改動二：移除 kNN
- 移除鄰近度特徵、`conf`、gate 與 OOV 相似度門檻。每個 span 只剩 6 維 meta：`[v0, a0, cmv, cma, hit, is_lex]`（`is_lex` = 核心詞是否在 CVAW）。
- 看過／沒看過的詞仍走同一條路徑，差別只在 NNword 自身的預測。
- 原因：在 NNword 層單獨比較時，kNN 與原 NNword 幾乎相同（測試詞 r：NNword 0.880 / 0.616，混合後變化 +0.000 / +0.005），且相似度信心與預測誤差相關性很弱（ρ 約 -0.08 / -0.05）。推測 kNN 與 NNword 建立在同一個 BERT 向量空間，資訊大量重疊。
### 改動三：NNsent 斷詞改用 jieba
- v8-UniCtx 逐字子字串比對，會把「強調」切出「強」、「視病如親」切出「親」，並把 `真的`、`極了`、`這種`、`雖然` 等功能詞收成 OOV span。
- 新版使用 jieba 繁體詞典（`dict.txt.big`），並把 CVAW 詞全部加入詞典；以「整個斷詞結果」比對 CVAW，修飾詞只收緊鄰在情緒詞前的詞（最多 3 個 token，不跨標點）。
- `OOV_SPANS` 預設關閉；句中找不到任何 CVAW 詞時，整句當一個 span（fallback）。
### 改動四：NNsent 的 L1 / L2 加強
- L2：1e-4 → 1e-3；L1：1e-5 → 5e-5；另加整句向量 dropout 0.3、隱藏層 dropout 0.2，early stopping patience 15。
- 附選用的 L1 × L2 驗證集掃描（`RUN_REG_SWEEP`），只看驗證集，不用測試集挑參數。
### 效果（CVAS 測試集，單次、單一種子，Colab 實際執行結果）
 
| | v7-3c | v8-UniCtx | v8-UniCtx2 |
|---|---|---|---|
| NNsent Valence r | 0.7292 | 0.7378 | 0.7345 |
| NNsent Arousal r | 0.3954 | 0.4589 | 0.4363 |
| 驗證集 Valence / Arousal r | — | 0.7035 / 0.4345 | 0.6984 / 0.3837 |
| 最佳 epoch | — | 10 | 42 |
| 最佳驗證 loss | — | 0.0188 | 0.0195 |
 
v8-UniCtx2 的 NNsent 訓練／驗證／測試 Pearson r：Valence 0.7793 / 0.6984 / 0.7345，Arousal 0.5783 / 0.3837 / 0.4363；測試集 MAE：Valence 0.0930、Arousal 0.1142。
NNword（測試詞）r：Valence 0.8745、Arousal 0.6320（訓練詞 0.9346 / 0.8328）。
 
**觀察**
- 加強正則化後訓練變穩（最佳 epoch 由 10 延到 42，訓練與驗證 loss 差距很小），但驗證與測試表現沒有因此變好；Arousal 比 v8-UniCtx 低 0.023，比 v7-3c 仍高 0.041。
- 拿掉 kNN 後 Arousal 仍高於 v7-3c，與 NNword 層實驗一致，推測進步主要來自 v8 共通的整句語境（128 長度）與統一路徑，而非 kNN；此點尚未以多種子與消融確認。
- NNword 微調層對測試表現沒有明顯改變（與先前 NNword-only 實驗 0.880 / 0.616 相比，差異在噪音內），Arousal 的訓練－測試落差仍約 0.20。
- 斷詞異常（`強`、`親`、`真的`、`旅行`、`家屬` 等）已消失。
**已知限制**
- 單次、單一種子的差異在 ±0.01 內不宜過度解讀；四項修改同時進行，無法單獨歸因。
- 情緒詞不在 CVAW 時（例如「動容」）只能退回整句 fallback；`完全` 等詞不在修飾詞表內，「完全無法」不會被當成帶修飾詞的 span。
- 每個 span 都帶整句語境，因此 14.6 的 Leave-One-Out「沒作用」標籤在多 span 句子中會偏低，不宜照字面解讀；單一 span 句子較可信。
---
 
## v7 更新說明：Word2Vec → BERT，NNword 架構重構
 
### 改動一：詞向量來源改為 BERT
- 改用 `hfl/chinese-bert-wwm-ext`（哈工大以數億字級中文語料預訓練的模型）
- 透過 self-attention 直接為任意詞產生 768 維語意向量，不再需要 kNN 補值
- 測試集未見詞的預測品質明顯提升
### 改動二：NNword 架構重構
 
**舊版：相似度加權求和**
`Int(w) = W₂ · (sigmoid(W₁·Sw + b₁) ⊙ X) + b₂`
輸入為目標詞與所有種子詞的 cosine similarity 向量 `Sw`。此設計容易收斂到「輸出平均值」的退化解（`W₂ → 0`，無論輸入為何皆輸出固定常數）。
 
**新版：BERT `[CLS]` 向量直接接三層 MLP**
`768 → 256 → 64 → 2`（最後 sigmoid 壓至 `[0,1]`）
不再依賴種子詞相似度，直接從語意向量學習情感強度，避免了退化解。
 
### NNmod：邏輯不變，僅輸入維度改變
沿用殘差移位設計，輸入維度由 128（Word2Vec）改為 768（BERT）：
 
- `delta = tanh(fc₂(BN(fc₁(vec_mod)))) × 0.5`
- `Int(mod w) = sigmoid(logit(Int(w)) + delta × 3)`
`tanh` 讓 delta 可正可負，使「非常」「沒有」等修飾詞能朝相反方向移動情感強度；在 logit（log-odds）空間做位移，確保效果在 `[0,1]` 全範圍內都均勻，不會在邊界附近被壓縮。
 
### 訓練流程：兩階段，NNword 於 Stage 2 凍結
1. **Stage 1**：僅訓練 NNword，於 CVAW 上跑 50 epochs
2. **Stage 2**：凍結 NNword 參數，僅訓練 NNmod，跑 100 epochs
凍結原因：先前嘗試聯合訓練（NNword + NNmod 一起更新）時，NNmod 回傳的梯度會把 NNword 帶壞，使其退化為固定輸出。凍結後 NNword 只做推論、輸出穩定的 `Int(w)`，NNmod 才能正確學到修飾詞的位移效果。
 
