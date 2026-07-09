"""
inference.py  —  APT Attribution Engine
Standalone module. Import predict_top3() into app.py directly.
No FastAPI, no Docker needed.

Usage:
    from inference import predict_top3
    results = predict_top3(["T1059", "T1003", "T1566"])
"""
import pathlib
import warnings
import numpy as np
import joblib

_ROOT          = pathlib.Path(__file__).parent.resolve()
_model         = None
_feature_vocab = None


def _load():
    global _model, _feature_vocab
    if _model is None:
        _model         = joblib.load(_ROOT / "models"    / "model.joblib")
        _feature_vocab = joblib.load(_ROOT / "artifacts" / "feature_vocab.joblib")


def encode_incident(technique_ids, feature_vocab):
    n   = len(feature_vocab)
    vec = np.zeros((1, n), dtype=np.float32)
    unknown, matched = [], 0
    for tid in technique_ids:
        root = tid.split(".")[0] if "." in tid else tid
        if root in feature_vocab:
            vec[0, feature_vocab[root]] = 1.0
            matched += 1
        else:
            unknown.append(tid)
    if unknown:
        warnings.warn(
            f"encode_incident: {len(unknown)} unknown ID(s) skipped: {unknown}",
            UserWarning, stacklevel=2,
        )
    return vec, matched, unknown


def predict_top3(technique_ids):
    """
    Predict top-3 APT groups for a list of ATT&CK technique IDs.

    Parameters
    ----------
    technique_ids : list[str]

    Returns
    -------
    list[dict] — [{rank, group, confidence}, ...]

    Raises
    ------
    ValueError if list is empty or all IDs are unknown.
    """
    _load()

    if not technique_ids:
        raise ValueError("technique_ids must be a non-empty list.")

    vec, matched, unknown = encode_incident(technique_ids, _feature_vocab)

    if matched == 0:
        raise ValueError(f"No known technique IDs provided. Unrecognised: {unknown}")

    proba    = _model.predict_proba(vec)[0]
    top3_idx = np.argsort(proba)[::-1][:3]

    return [
        {
            "rank"      : int(i + 1),
            "group"     : str(_model.classes_[idx]),
            "confidence": float(round(proba[idx], 6)),
        }
        for i, idx in enumerate(top3_idx)
    ]


if __name__ == "__main__":
    test = ["T1059", "T1003", "T1566"]
    print(f"Testing predict_top3({test})\n")
    for r in predict_top3(test):
        bar = "█" * int(r["confidence"] * 40)
        print(f"  #{r['rank']}  {r['group']:<35}  {r['confidence']:.4f}  {bar}")