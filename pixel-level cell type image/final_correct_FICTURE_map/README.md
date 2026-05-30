# Final Correct FICTURE Map

这个文件夹只保留最终有用文件。

## 规则

只使用这次 FICTURE 输出自己配套的文件：

- source PNG: `pixel-level cell type image/visiumhd_exp1_hex12_k12/hex_12.k12.pixel.png`
- source HTML: `pixel-level cell type image/visiumhd_exp1_hex12_k12/hex_12.k12.pixel.info.html`

这里不放其他 HTML，也不放任何后来解释/推断出来的标签。

## 这个 map 怎么来的

- FICTURE map 位置来自已经修正并通过 `PASS_OFFICIAL` 的 H&E 对齐结果。
- ROI: `[75, 40, 3219, 3367]`，大小 `3144 x 3327`。
- 没有手动 dx/dy 平移，没有按 ROI 强行 resize。

## 保留文件

- `final_ficture_roi_3144x3327.png`: 最终正确 FICTURE ROI map。
- `source_matched_factor_info.html`: 这次 FICTURE 输出自己配套的 HTML。
- `final_factor_color_gene_table_from_ficture_html.csv`: 只从上面 HTML 直接提取的 factor/RGB/gene 表。
- `manifest_final_correct_ficture_map.json`: 输入、输出和生成规则记录。
