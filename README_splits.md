# Splitting without template leakage

Each row has a `template_group_id` (in the `*_meta.csv` sidecar, row-aligned with the main CSV).
Rows sharing a group id were built from the same base problem sentence (trade x language x template),
differing only in prefix / area / wage / typos / filler. Keep every group on ONE side of a split.

```python
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

parts = [pd.read_csv(f"dhaka_trade_dataset_2m_part{i}of4.csv") for i in range(1, 5)]
metas = [pd.read_csv(f"dhaka_trade_dataset_2m_part{i}of4_meta.csv") for i in range(1, 5)]
df = pd.concat([pd.concat(parts, ignore_index=True), pd.concat(metas, ignore_index=True)], axis=1)

# 10 folds -> fold 0 = test (10%), fold 1 = val (10%), rest = train (80%)
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
folds = list(sgkf.split(df, df["trade"], groups=df["template_group_id"]))
test_idx, val_idx = folds[0][1], folds[1][1]
train_idx = sorted(set(range(len(df))) - set(test_idx) - set(val_idx))
```
Notes: group id = trade*1000 + lang_index*100 + template index (hard negatives offset by +50);
`template_groups.csv` maps ids to the template text. `lang` and `is_hard_negative` columns let you report
per-language / boundary-case metrics. Do not use TF-IDF vocab fit on the full data before splitting.
