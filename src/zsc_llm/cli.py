from __future__ import annotations
import argparse, json, csv, sys
from pathlib import Path
from typing import List, Dict, Optional
from . import __version__
from .nb_method import ZeroShotHierarchicalClassifier, SimpleTFIDFEmbedder, SentenceTransformerEmbedder

def _read_lines(p: Path) -> List[str]:
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

def _read_json_map(p: Path) -> Dict[str, str]:
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit("label_descs must be a JSON object mapping label -> description")
    return {str(k): str(v) for k, v in obj.items()}

def _read_texts(path: Path, text_col: Optional[str], text_key: str) -> List[str]:
    if path.suffix.lower() == ".txt":
        return _read_lines(path)
    if path.suffix.lower() == ".jsonl":
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                if text_key not in obj:
                    raise SystemExit(f"Key '{text_key}' not in JSONL record")
                out.append(str(obj[text_key]))
        return out
    if path.suffix.lower() == ".csv":
        if not text_col:
            raise SystemExit("--text-col is required for CSV input")
        out = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if text_col not in reader.fieldnames:
                raise SystemExit(f"Column '{text_col}' not found in CSV header: {reader.fieldnames}")
            for r in reader:
                out.append(str(r[text_col]))
        return out
    raise SystemExit("Unsupported input format. Use .txt, .csv, or .jsonl")

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="zsc-llm", description="Zero-shot hierarchical classification (notebook methodology)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="Show version info")
    p_info.set_defaults(func=lambda a: (print(f"zsc-llm {__version__}") or 0))

    p_nb = sub.add_parser("predict-notebook", help="Build tree from labels/label descriptions and classify level-by-level")
    p_nb.add_argument("--labels-file", type=str, required=False, help="TXT (one label per line) or JSON list of labels")
    p_nb.add_argument("--label-descs-file", type=str, required=False, help="JSON mapping label -> description (optional)")
    p_nb.add_argument("--input", type=str, required=True, help="Path to .txt / .csv / .jsonl with examples")
    p_nb.add_argument("--text-col", type=str, default=None, help="CSV column name containing text")
    p_nb.add_argument("--text-key", type=str, default="text", help="JSONL key for text")
    p_nb.add_argument("--embedder", type=str, default="tfidf", choices=["tfidf","st"], help="Embedder for labels & routing")
    p_nb.add_argument("--st-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="ST model name if --embedder=st")
    p_nb.add_argument("--device", type=str, default=None, help="Device for ST model (e.g., cuda:0)")
    p_nb.add_argument("--branching-factor", type=int, default=8)
    p_nb.add_argument("--min-cluster-size", type=int, default=1)
    p_nb.add_argument("--temperature", type=float, default=0.7)
    p_nb.add_argument("--beam", type=int, default=2)
    p_nb.add_argument("--topk-paths", type=int, default=3)
    p_nb.add_argument("--node-names", type=str, default="keywords", choices=["keywords","children","none"], help="How to name internal nodes in paths")
    p_nb.add_argument("--scores", type=str, default="prob", choices=["prob","log"], help="Output path scores as probabilities (0..1) or log-probabilities")
    p_nb.add_argument("--out", type=str, default=None)

    def _cmd_nb(a) -> int:
        labels_list = None
        label_descs = None

        if a.labels_file:
            p = Path(a.labels_file)
            if p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(obj, list):
                    raise SystemExit("--labels-file JSON must be a list of strings")
                labels_list = [str(x) for x in obj]
            else:
                labels_list = _read_lines(p)

        if a.label_descs_file:
            label_descs = _read_json_map(Path(a.label_descs_file))

        if labels_list is None and label_descs is None:
            raise SystemExit("Provide at least one of --labels-file or --label-descs-file.")

        texts = _read_texts(Path(a.input), a.text_col, a.text_key)

        emb = SimpleTFIDFEmbedder() if a.embedder == "tfidf" else SentenceTransformerEmbedder(a.st_model, device=a.device)
        clf = ZeroShotHierarchicalClassifier(
            branching_factor=a.branching_factor,
            min_cluster_size=a.min_cluster_size,
            cluster_method="divisive_kmeans",
            embedder=emb,
            sim_temperature=a.temperature,
        )
        if label_descs is not None:
            clf.fit(label_texts=label_descs)
        else:
            clf.fit(labels=labels_list)

        paths = clf.predict_paths(texts, topk_paths=a.topk_paths, beam=a.beam, score_mode=a.scores)
        header = ["index","text","path","path_prob" if a.scores=="prob" else "log_score","top_leaf"]
        rows = []
        for i, (t, plist) in enumerate(zip(texts, paths)):
            top_leaf = plist[0][0][-1] if plist else ""
            for pth, sc in plist:
                rows.append([i, t, " > ".join(pth), f"{sc:.6f}", top_leaf])

        if a.out:
            with open(a.out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(header); w.writerows(rows)
            print(f"Wrote {a.out}")
        else:
            print("\t".join(header))
            for r in rows:
                print("\t".join(map(str, r)))
        return 0

    p_nb.set_defaults(func=_cmd_nb)

    args = ap.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
