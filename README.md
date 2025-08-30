# zsc-llm — Notebook‑accurate Hierarchical Zero‑Shot Classification

> Build a **label tree** from your label names/descriptions and classify texts **level‑by‑level** with fast embedding similarity. Clean Python API + simple CLI. Works with **TF‑IDF** (default) or **Sentence‑Transformers**.

---

## ✨ Features

- **Notebook‑accurate**: mirrors the methodology you prototyped (build hierarchy → route per level)
- **Plug & play**: start with just **label names** (descriptions optional, but recommended)
- **Interpretable**: returns the **full decision path** through the tree to the leaf class
- **Flexible embeddings**: TF‑IDF (CPU, fast) or Sentence‑Transformers (GPU‑ready)
- **Readable nodes**: internal clusters named via **keywords** or **child labels**
- **Scoring**: choose **log‑prob** (additive, stable) or **prob** (0..1)
- **CLI & Python**: one command or a few lines of code

---

## 📦 Installation

```bash
# clone your repo (or unzip the folder)
git clone <your-repo-url>.git
cd zsc-llm

# install in editable mode into the current Python environment
pip install -e .
```

> **Colab/Jupyter tip**: run `pip install -e .` **inside a notebook cell** so it installs into the same kernel.
>
> If you choose not to install, you can temporarily add the path:
> ```python
> import sys; sys.path.append("/content/zsc-llm/src")
> ```

### Requirements
- Python ≥ 3.9
- `numpy`, `scikit-learn`
- Optional: `sentence-transformers` (for transformer embeddings & GPU)

---

## 🧠 How it works (one minute)

1. Provide **labels** (and optionally **descriptions**).
2. We embed label descriptions (or label names if descriptions aren’t provided).
3. We **cluster labels** recursively (divisive k‑means) → build a tree: internal **nodes** → **leaves** (labels).
4. For each internal node we compute **prototypes** for children and install a **router** (cosine sim → softmax).
5. At inference:
   - Embed text, start at **Root**, choose best child, **descend** level‑by‑level.
   - Optional **beam search** explores multiple branches per level.
   - Return **top‑k paths** and the **leaf** as the predicted class.

---

## 🚀 Quick start — CLI

### A) Using **label descriptions** (recommended)
`label_descs.json` maps label → richer description.

```bash
zsc-llm predict-notebook \
  --label-descs-file samples/label_descs_dense.json \
  --input samples/stress_texts.txt \
  --embedder tfidf \
  --beam 5 --topk-paths 5 \
  --scores prob \
  --node-names keywords \
  --out paths.csv
```

- `--label-descs-file`: JSON like `{ "LabelName": "longer description", ... }`
- `--scores`: `prob` (0..1) or `log` (sum of log‑probs along the path)
- `--node-names`: `keywords` (TF‑IDF topic), `children` (child label tokens), or `none`
- Use Sentence‑Transformers:
  ```bash
  zsc-llm predict-notebook \
    --label-descs-file samples/label_descs_dense.json \
    --input samples/stress_texts.txt \
    --embedder st --st-model sentence-transformers/all-MiniLM-L6-v2 --device cuda:0 \
    --beam 5 --topk-paths 5 --scores prob
  ```

### B) Using **labels only** (descriptions optional)
TXT file with one label per line, or a JSON list.

```bash
zsc-llm predict-notebook \
  --labels-file samples/labels_min.txt \
  --input samples/long_example.txt \
  --embedder tfidf \
  --beam 3 --topk-paths 3 \
  --scores prob
```

### Input formats
- `.txt` — one text per line
- `.jsonl` — one JSON object per line; use `--text-key` to specify the field (default: `text`)
- `.csv` — supply `--text-col <column name>`

**Output**: a CSV with columns  
`index, text, path, path_prob|log_score, top_leaf`

---

## 🐍 Quick start — Python

### A) With **descriptions**
```python
from zsc_llm.nb_method import ZeroShotHierarchicalClassifier, SimpleTFIDFEmbedder
import json, pathlib

label_texts = json.loads(pathlib.Path("samples/label_descs_dense.json").read_text())

clf = ZeroShotHierarchicalClassifier(
    branching_factor=8,
    min_cluster_size=1,
    embedder=SimpleTFIDFEmbedder(),  # TF-IDF (fast, CPU)
    sim_temperature=0.7,
    internal_naming="keywords",      # or "children" / "none"
).fit(label_texts=label_texts)

text = "ACH to my external bank is pending for days and I also see a foreign transaction fee on a hotel booking."
paths = clf.predict_paths([text], topk_paths=5, beam=5, score_mode="prob")

# Full paths with probabilities
for path, prob in paths[0]:
    print(" > ".join(path), f"| path_prob={prob:.4f}")

# Best class (leaf)
best_path, best_prob = paths[0][0]
best_class = best_path[-1]
print(f"\nBest class: {best_class}  (path_prob={best_prob:.4f})")
```

### B) With **labels only**
```python
from zsc_llm.nb_method import ZeroShotHierarchicalClassifier, SimpleTFIDFEmbedder

labels = ["Billing","Refunds","TechSupport","AppBugs","CardIssues","AccountClosure"]

clf = ZeroShotHierarchicalClassifier(
    embedder=SimpleTFIDFEmbedder(),
    internal_naming="children"  # robust naming even without descriptions
).fit(labels=labels)

print(clf.predict_paths(["App crashed during transfer and card declined at POS."],
                        topk_paths=3, beam=3, score_mode="prob")[0])
```

### C) With **Sentence‑Transformers** (GPU)
```python
from zsc_llm.nb_method import ZeroShotHierarchicalClassifier, SentenceTransformerEmbedder

clf = ZeroShotHierarchicalClassifier(
    embedder=SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", device="cuda:0"),
    internal_naming="children"
).fit(labels=["Billing","Refunds","TechSupport","AppBugs","CardIssues","AccountClosure"])
```

---

## 🔍 Output explained

A path looks like:
```
Root > node[ach pending wire] > node[foreign fee fx] > Fee_ForeignTxn | path_prob=0.42
```
- **Internal nodes** are clusters; named from **TF‑IDF keywords** or **child labels** so you can read them.
- **Leaf** (last segment) is the predicted class.
- `path_prob` multiplies per‑level probabilities (converted from log‑probs for readability).
- Prefer `score_mode="log"` for numerically stable **additive** scores during analysis.

---

## ⚙️ Key parameters

- `branching_factor` — max children per split (default **8**)
- `min_cluster_size` — small clusters merge/fallback (default **1**)
- `beam` — keep top‑K branches per level at inference (default **1**)
- `topk_paths` — how many full paths to return (default **1**)
- `sim_temperature` — softmax temperature at routers (lower → peakier)
- `internal_naming` — `keywords` | `children` | `none`
- `score_mode` — `prob` | `log`

---

## 📂 Samples

- `samples/label_descs_dense.json` — **80** closely‑related classes with overlapping descriptions
- `samples/stress_texts.txt` — **50** ambiguous, multi‑issue test lines to stress the hierarchy
- `samples/labels_min.txt` — tiny labels‑only example
- `samples/long_example.txt` — long multi‑topic customer message

Run the stress set:
```bash
zsc-llm predict-notebook \
  --label-descs-file samples/label_descs_dense.json \
  --input samples/stress_texts.txt \
  --embedder tfidf \
  --beam 5 --topk-paths 5 --scores prob \
  --node-names keywords \
  --out stress_paths.csv
```

---

## 🧪 Dev & testing

```bash
# run tests
pytest -q
```

Project layout:
```
zsc-llm/
├─ src/zsc_llm/
│  ├─ __init__.py
│  ├─ nb_method.py          # main implementation (API)
│  └─ cli.py                # CLI: zsc-llm predict-notebook
├─ samples/
│  ├─ label_descs_dense.json
│  ├─ stress_texts.txt
│  ├─ labels_min.txt
│  └─ long_example.txt
├─ tests/
│  ├─ test_basic.py
│  └─ test_labels_only.py
├─ pyproject.toml
├─ README.md
└─ LICENSE
```

---

## 🛠️ Troubleshooting

- **I still see `node` in paths**  
  Set `internal_naming="children"` (works even with labels‑only), reinstall (`pip install -e .`), and re‑run.

- **Import fails in notebook**  
  Ensure you ran `pip install -e .` **in the same kernel**. Otherwise use the temporary path:  
  `import sys; sys.path.append("/content/zsc-llm/src")`

- **Slow ST encoding**  
  Start with TF‑IDF (fast) → switch to Sentence‑Transformers with `device="cuda:0"` for better semantics.

- **Probabilities look small**  
  They’re **path** probabilities (product across levels). Use `score_mode="log"` for additive analysis.

---

## 📜 License

MIT — see `LICENSE`.

---

## 🙏 Citation

If you use this code, a simple citation in your README or docs is appreciated:

> *zsc‑llm: Notebook‑accurate hierarchical zero‑shot classification (TF‑IDF/ST embeddings, per‑level routing), 2025.*
