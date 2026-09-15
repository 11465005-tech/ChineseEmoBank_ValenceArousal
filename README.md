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
