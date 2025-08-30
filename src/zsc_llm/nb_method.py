# Notebook-accurate hierarchical zero-shot classification
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterable, Tuple, Sequence
import numpy as np
import warnings

# sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional sentence-transformers embedder
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

# ---------------- Embedders ----------------
class BaseEmbedder:
    def fit(self, texts: Sequence[str]) -> "BaseEmbedder":
        return self
    def transform(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

class SimpleTFIDFEmbedder(BaseEmbedder):
    def __init__(self, max_features: int = 20000, ngram_range=(1,2), lowercase: bool=True):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, lowercase=lowercase)
    def fit(self, texts: Sequence[str]) -> "SimpleTFIDFEmbedder":
        self.vectorizer.fit(list(texts))
        return self
    def transform(self, texts: Sequence[str]) -> np.ndarray:
        X = self.vectorizer.transform(list(texts))
        X = X.astype(np.float32)
        X = normalize(X, norm="l2", axis=1)
        return X.toarray()

class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str]=None):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Please `pip install sentence-transformers`.")
        self.model = SentenceTransformer(model_name, device=device)
    def fit(self, texts: Sequence[str]) -> "SentenceTransformerEmbedder":
        _ = texts
        return self
    def transform(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(list(texts), normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        return emb.astype(np.float32)

# ---------------- Tree ----------------
_node_id_counter = 0
def _new_node_id() -> int:
    global _node_id_counter
    _node_id_counter += 1
    return _node_id_counter

@dataclass
class TreeNode:
    node_id: int
    name: str
    children: List["TreeNode"] = field(default_factory=list)
    parent_id: Optional[int] = None
    is_leaf: bool = False
    label: Optional[str] = None     # only for leaves
    description: Optional[str] = None
    # router artifacts
    child_names: List[str] = field(default_factory=list)
    prototypes: Optional[np.ndarray] = None
    temperature: float = 0.7

def iterate_nodes(root: TreeNode) -> Iterable[TreeNode]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for c in node.children:
            stack.append(c)

def _descendant_labels(node: TreeNode) -> List[str]:
    if node.is_leaf and node.label is not None:
        return [node.label]
    out: List[str] = []
    for c in node.children:
        out.extend(_descendant_labels(c))
    return out

# ---------------- Clustering (divisive k-means) ----------------
def _build_tree_divisive_kmeans(labels: List[str], label_vecs: Dict[str, np.ndarray],
                                branching_factor: int, min_cluster_size: int,
                                random_state: int=42) -> TreeNode:
    rng = np.random.default_rng(random_state)

    def build(sub_labels: List[str]) -> TreeNode:
        if len(sub_labels) <= 1:
            l = sub_labels[0]
            return TreeNode(node_id=_new_node_id(), name=l, is_leaf=True, label=l)

        k = min(branching_factor, len(sub_labels))
        V = np.stack([label_vecs[l] for l in sub_labels], axis=0).astype(np.float32)
        # Stabilize:
        V = V + 1e-6 * rng.standard_normal(V.shape).astype(np.float32)

        # If k==1 due to small size, make leaves
        if k <= 1:
            node = TreeNode(node_id=_new_node_id(), name="group", is_leaf=False)
            for l in sub_labels:
                node.children.append(TreeNode(node_id=_new_node_id(), name=l, is_leaf=True, label=l))
            return node

        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        idx = km.fit_predict(V)

        # group by cluster
        clusters: List[List[str]] = [[] for _ in range(k)]
        for i, lab in enumerate(sub_labels):
            clusters[int(idx[i])].append(lab)

        # merge tiny clusters if needed
        clusters = [c for c in clusters if len(c) > 0]
        if any(len(c) < min_cluster_size for c in clusters) and len(sub_labels) > 1:
            # simple fallback: reduce k and retry
            return build(sub_labels[:]) if k == 2 else build(sub_labels)

        node = TreeNode(node_id=_new_node_id(), name="node", is_leaf=False)
        for c in clusters:
            if len(c) == 1:
                node.children.append(TreeNode(node_id=_new_node_id(), name=c[0], is_leaf=True, label=c[0]))
            else:
                node.children.append(build(c))
        return node

    root = TreeNode(node_id=_new_node_id(), name="Root", is_leaf=False)
    # top split
    top = build(labels)
    if top.is_leaf:
        root.children.append(top)
    else:
        root.children = top.children
        for ch in root.children:
            ch.parent_id = root.node_id
    return root

# ---------------- Zero-shot router (EmbSim) ----------------
def _softmax(z: np.ndarray, temperature: float) -> np.ndarray:
    t = max(temperature, 1e-6)
    z = z / t
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (d,), b: (m, d) -> (m,)
    a = a / max(np.linalg.norm(a) + 1e-12, 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return (b @ a).astype(np.float32)

def _simple_tokens(text: str) -> list[str]:
    import re as _re
    toks = [t.lower() for t in _re.findall(r"[A-Za-z0-9]+", text)]
    # drop very short tokens
    return [t for t in toks if len(t) >= 3]

# ---------------- Hierarchical classifier ----------------
class ZeroShotHierarchicalClassifier:
    """
    Build a label tree from label descriptions (TF-IDF or ST embeddings), then route level-by-level.
    internal_naming: 'keywords' (TF-IDF summary), 'children' (label-token summary), or 'none'.
    """
    def __init__(
                     self,
                     branching_factor: int = 8,
                     min_cluster_size: int = 1,
                     cluster_method: str = "divisive_kmeans",
                     random_state: int = 42,
                     embedder: Optional[BaseEmbedder] = None,
                     sim_temperature: float = 0.7,
                     internal_naming: str = "keywords"  # 'keywords' | 'children' | 'none'
                     ):
        if embedder is None:
            embedder = SimpleTFIDFEmbedder()
        self.branching_factor = int(branching_factor)
        self.min_cluster_size = int(min_cluster_size)
        self.cluster_method = str(cluster_method)
        self.random_state = int(random_state)
        self.embedder = embedder
        self.sim_temperature = float(sim_temperature)
        self.internal_naming = internal_naming

        self.root: Optional[TreeNode] = None
        self.label_descriptions: Dict[str, str] = {}
        self.label_vecs: Dict[str, np.ndarray] = {}

    def fit(self, *, labels: Sequence[str] | None = None, label_texts: Dict[str, str] | None = None) -> "ZeroShotHierarchicalClassifier":
        """
        Train the classifier:
        - If `label_texts` is provided (preferred), it's a mapping {label -> description}.
        - Else, provide `labels` (list of label names). Descriptions default to the label names.
        """
        if label_texts is not None:
            labels_list = list(label_texts.keys())
            if not labels_list:
                raise ValueError("label_texts is empty")
            descs = [str(label_texts[l]) for l in labels_list]
            self.label_descriptions = {l: str(label_texts[l]) for l in labels_list}
        else:
            if not labels:
                raise ValueError("Provide either label_texts or labels.")
            labels_list = list(labels)
            descs = [str(l) for l in labels_list]
            self.label_descriptions = {l: str(l) for l in labels_list}

        # 1) Fit embedder on descriptions
        self.embedder.fit(descs)
        V = self.embedder.transform(descs)
        self.label_vecs = {l: V[i] for i, l in enumerate(labels_list)}

        # 2) Build tree (divisive kmeans only for now)
        if self.cluster_method != "divisive_kmeans":
            warnings.warn("Only 'divisive_kmeans' is implemented; falling back to it.")
        self.root = _build_tree_divisive_kmeans(labels_list, self.label_vecs,
                                                branching_factor=self.branching_factor,
                                                min_cluster_size=self.min_cluster_size,
                                                random_state=self.random_state)

        # 3) Install routers (prototypes) at each internal node + assign readable names
        for node in iterate_nodes(self.root):
            if node.is_leaf:
                continue
            child_names = []
            protos = []
            for child in node.children:
                names = _descendant_labels(child)
                child_names.append(child.name if not child.is_leaf else child.label or child.name)
                vecs = np.stack([self.label_vecs[n] for n in names], axis=0)
                proto = vecs.mean(axis=0)
                nrm = np.linalg.norm(proto) + 1e-12
                proto = (proto / nrm).astype(np.float32)
                protos.append(proto)
            node.child_names = child_names
            node.prototypes = np.stack(protos, axis=0).astype(np.float32)
            node.temperature = self.sim_temperature

            # --- Human-friendly internal node names ---
            topic_name: Optional[str] = None
            if self.internal_naming != "none":
                # Strategy A: TF-IDF keyword summary (works if TF-IDF embedder is used)
                try:
                    vec = getattr(self.embedder, "vectorizer", None)
                    if self.internal_naming in ("keywords", "children") and vec is not None and hasattr(vec, "get_feature_names_out"):
                        fnames = vec.get_feature_names_out()
                        desc_texts = [self.label_descriptions[l] for l in _descendant_labels(node)]
                        if desc_texts:
                            M = vec.transform(desc_texts).astype("float32")
                            g = M.mean(axis=0).A.ravel()
                            topk = g.argsort()[::-1][:3]
                            keywords = [fnames[i] for i in topk if g[i] > 0]
                            if keywords:
                                topic_name = "node[" + " ".join(keywords) + "]"
                except Exception:
                    topic_name = None

                # Strategy B: summarize by tokens from descendant leaf labels (robust)
                if (topic_name is None) and (self.internal_naming in ("children", "keywords")):
                    from collections import Counter
                    leaves = _descendant_labels(node)
                    cnt = Counter()
                    for lab in leaves:
                        for t in set(_simple_tokens(lab)):
                            cnt[t] += 1
                    common = [w for w, _ in cnt.most_common(3)]
                    if common:
                        topic_name = "node[" + " ".join(common) + "]"

            # Final fallback: show first few child names so it's never just "node"
            if not topic_name:
                child_labels = []
                for ch in node.children:
                    child_labels.append(ch.label if ch.is_leaf and ch.label else ch.name)
                if child_labels:
                    topic_name = "node[" + " ".join(child_labels[:3]) + "]"

            if topic_name:
                node.name = topic_name

        return self

    def predict_paths(self, texts: Sequence[str], *, topk_paths: int = 1, beam: int = 1, score_mode: str = 'log') -> List[List[Tuple[List[str], float]]]:
        if self.root is None:
            raise RuntimeError("Call fit() first.")

        X = self.embedder.transform(list(texts)).astype(np.float32)
        out: List[List[Tuple[List[str], float]]] = []
        for x in X:
            out.append(self._predict_one(x, topk_paths=topk_paths, beam=beam, score_mode=score_mode))
        return out

    def _predict_one(self, x: np.ndarray, *, topk_paths: int, beam: int, score_mode: str) -> List[Tuple[List[str], float]]:
        # Beam entries: (node, path, score)
        # score is sum of log-probabilities along the path
        import math
        beam_list: List[Tuple[TreeNode, List[str], float]] = [(self.root, ["Root"], 0.0)]
        results: List[Tuple[List[str], float]] = []

        while beam_list:
            new_beam: List[Tuple[TreeNode, List[str], float]] = []
            for node, path, score in beam_list:
                if node.is_leaf or node.prototypes is None or node.prototypes.shape[0] == 0:
                    results.append((path, score))
                    continue
                sims = _cosine(x, node.prototypes)  # (C,)
                probs = _softmax(sims[None, :], node.temperature)[0]  # (C,)
                # pick top children
                idx = np.argsort(-probs)[: max(1, beam)]
                for j in idx:
                    child = node.children[int(j)]
                    new_beam.append((child, path + [child.name if not child.is_leaf else (child.label or child.name)],
                                     score + float(math.log(max(probs[int(j)], 1e-12)))))
            # prune beam
            if not new_beam:
                break
            new_beam.sort(key=lambda t: t[2], reverse=True)
            beam_list = new_beam[: max(1, beam)]

            # Stop if all are leaves (handled via moving to results)
            if all(n.is_leaf for n,_,_ in beam_list):
                results.extend([(p, s) for n,p,s in beam_list])
                break

        # Deduplicate by path and select topk
        acc: Dict[Tuple[str, ...], float] = {}
        for p, s in results:
            key = tuple(p)
            acc[key] = max(acc.get(key, -1e9), s)
        ranked = sorted([(list(k), v) for k,v in acc.items()], key=lambda t: t[1], reverse=True)
        return [(p, (float(np.exp(s)) if score_mode == 'prob' else s)) for p, s in ranked[: max(1, topk_paths)]]
