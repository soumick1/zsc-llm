# zsc-llm (clean rebuild)

Zero-shot **hierarchical** classification that matches the **notebook methodology**:
1) Build a **tree from label descriptions** (TF‑IDF by default, or Sentence-Transformers)
2) Install **per-node embedding-similarity routers**
3) **Classify level-by-level** with beam search

## Install
```bash
pip install -e .
```

## CLI (notebook method)
```bash
zsc-llm predict-notebook       --label-descs-file samples/label_descs.json       --input samples/long_example.txt       --embedder tfidf       --branching-factor 8 --min-cluster-size 1       --temperature 0.7 --beam 3 --topk-paths 3       --out paths.csv
```

## Python API
```python
from zsc_llm.nb_method import ZeroShotHierarchicalClassifier, SimpleTFIDFEmbedder

label_texts = {
    "Billing": "charges, fees, refunds, adjustments, statements, interest, taxation",
    "Refunds": "request or status of refunds, reversals, chargebacks",
    "TechSupport": "app/website problems, login, 2FA, device issues",
    "AppBugs": "crashes, slow, update problems, notifications not working",
    "CardIssues": "card not working at ATM or POS, declines",
    "AccountClosure": "close account, terminate service, transfer out",
}

clf = ZeroShotHierarchicalClassifier(
    branching_factor=8, min_cluster_size=1, cluster_method="divisive_kmeans",
    embedder=SimpleTFIDFEmbedder(), sim_temperature=0.7
).fit(label_texts=label_texts)

texts = ["My card is declined and the app keeps crashing after update."]
paths = clf.predict_paths(texts, topk_paths=3, beam=3)
print(paths[0])  # list of (path, score)
```



### Labels-only (descriptions optional)
**Python:**
```python
from zsc_llm.nb_method import ZeroShotHierarchicalClassifier, SimpleTFIDFEmbedder

labels = ["Billing","Refunds","TechSupport","AppBugs","CardIssues","AccountClosure"]
clf = ZeroShotHierarchicalClassifier(embedder=SimpleTFIDFEmbedder()).fit(labels=labels)
paths = clf.predict_paths(["App crashed during transfer and card declined at POS."], topk_paths=3, beam=3)
print(paths[0])
```

**CLI:**
```bash
zsc-llm predict-notebook       --labels-file samples/labels_min.txt       --input samples/long_example.txt       --embedder tfidf --beam 3 --topk-paths 3
```
