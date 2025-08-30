def test_import():
    import zsc_llm
    assert hasattr(zsc_llm, "__version__")

def test_fit_predict_paths():
    from zsc_llm.nb_method import ZeroShotHierarchicalClassifier, SimpleTFIDFEmbedder
    label_texts = {"A":"alpha beta", "B":"gamma delta", "C":"epsilon zeta"}
    clf = ZeroShotHierarchicalClassifier(embedder=SimpleTFIDFEmbedder()).fit(label_texts=label_texts)
    paths = clf.predict_paths(["alpha beta text"], topk_paths=2, beam=2)
    assert isinstance(paths, list) and len(paths) == 1
    assert isinstance(paths[0], list) and len(paths[0]) >= 1
