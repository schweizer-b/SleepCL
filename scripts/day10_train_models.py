# Train classical staging models with participant-separated split validation.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sleepcl.subject_ids import participant_id, validate_split


LABEL_ORDER = ["W", "N1", "N2", "N3", "REM"]
ALLOWED_STAGES = set(LABEL_ORDER)


def load_dataset(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]
    y = d["y"].astype(object)
    subject = d["subject"].astype(object)
    rec_id = d["rec_id"].astype(object)
    return X, y, subject, rec_id


def load_meta_and_spec(npz_path: Path) -> Tuple[Dict, object]:
    meta_path = Path(npz_path.with_suffix("").as_posix() + "_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Resolve the canonical sidecar relative to the actual dataset location.
    # Old meta may contain a path tied to the original working directory;
    # such a path must never select an unrelated, stale feature spec.
    spec_path = Path(npz_path.with_suffix("").as_posix() + "_feature_spec.joblib")
    declared_name = Path(str(meta.get("feature_spec_joblib", ""))).name
    if declared_name and declared_name != spec_path.name:
        raise ValueError(f"Feature sidecar metadata mismatch: {declared_name} vs {spec_path.name}")
    if not spec_path.is_file():
        raise FileNotFoundError(f"Feature sidecar joblib not found beside dataset: {spec_path}")
    feature_spec = joblib.load(spec_path)
    return meta, feature_spec


def ensure_predict_proba(est) -> None:
    if not hasattr(est, "predict_proba"):
        raise TypeError(
            "Model does not implement predict_proba(). "
            "Closed-loop confidence gating typically relies on probabilities. "
            "Use calibration for SVMs (CalibratedClassifierCV)."
        )


def plot_confusion(cm: np.ndarray, out_png: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks(range(len(LABEL_ORDER)), LABEL_ORDER, rotation=45, ha="right")
    plt.yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def per_subject_accuracy(y_true: np.ndarray, y_pred: np.ndarray, subj: np.ndarray) -> pd.DataFrame:
    rows = []
    for s in sorted(set(subj.tolist())):
        m = (subj == s)
        if not np.any(m):
            continue
        acc = float(np.mean((y_pred[m] == y_true[m]).astype(float)))
        rows.append({"subject": s, "n_epochs": int(np.sum(m)), "accuracy": acc})
    return pd.DataFrame(rows)


def build_models(model_names: List[str], seed: int, class_weight: str) -> Dict[str, object]:
    cw = None if class_weight == "none" else "balanced"
    models: Dict[str, object] = {}

    if "lr" in model_names:
        models["lr"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight=cw,
                random_state=seed,
            )),
        ])

    if "svm_lin" in model_names:
        # LinearSVC + calibration => predict_proba
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", LinearSVC(
                class_weight=cw,
                random_state=seed,
                max_iter=5000,
            )),
        ])
        models["svm_lin"] = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)

    if "svm_rbf" in model_names:
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                C=3.0,
                gamma="scale",
                class_weight=cw,
            )),
        ])
        models["svm_rbf"] = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)


    if "rf" in model_names:
        # RF has predict_proba natively
        models["rf"] = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            class_weight=cw,
        )

    if "mlp" in model_names:
        # MLPClassifier doesn't support class_weight; keep it simple.
        models["mlp"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=200,
                early_stopping=True,
                random_state=seed,
            )),
        ])

    return models


def main() -> None:
    ap = argparse.ArgumentParser(description="Day 10: train multiple stager models on cached features (subject-wise split).")
    ap.add_argument("--dataset", required=True, help="Path to .npz from day10_make_dataset.py")
    ap.add_argument("--split_json", required=True, help="New participant-level split JSON with train/validation/test lists. TEST is never evaluated by this trainer.")
    ap.add_argument("--outdir", default="runs/day10_models")
    ap.add_argument("--models", nargs="+", default=["lr", "svm_lin", "rf", "mlp"],
                    choices=["lr", "svm_lin", "svm_rbf", "rf", "mlp"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--class_weight", default="none", choices=["none", "balanced"])
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    out_root = Path(args.outdir)
    # Never silently overwrite historical models or reports.
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Refusing nonempty model output directory: {out_root}")

    X, y, subject, rec_id = load_dataset(dataset_path)
    # Fail early on known historical night-ID caches, before loading a possibly
    # stale/missing sidecar. This preserves the explicit Batch03 safety gate.
    preliminary_meta_path = Path(dataset_path.with_suffix("").as_posix() + "_meta.json")
    if not preliminary_meta_path.is_file():
        raise FileNotFoundError(f"Meta not found: {preliminary_meta_path}")
    if json.loads(preliminary_meta_path.read_text(encoding="utf-8")).get("subject_id_semantics") != "sleepedf-person-v2":
        raise ValueError("Legacy or unknown dataset subject IDs; rebuild cache into a NEW path using corrected Day10 dataset script")
    meta, feature_spec = load_meta_and_spec(dataset_path)
    if not (len(X) == len(y) == len(subject) == len(rec_id)):
        raise ValueError("Dataset row arrays have inconsistent lengths")
    if any(str(s) != participant_id(str(r)) for s, r in zip(subject, rec_id)):
        raise ValueError("Dataset contains a participant/recording mismatch")
    if not np.all(np.isfinite(X)):
        raise ValueError("Feature dataset contains nonfinite values")
    if X.dtype != np.dtype("float32") or meta.get("feature_dtype", "float32") != "float32":
        raise ValueError("Day10 v2 dataset must contain float32 X and matching feature_dtype metadata")

    # Clean labels defensively
    keep = np.array([str(lbl) in ALLOWED_STAGES for lbl in y], dtype=bool)
    X = X[keep]
    y = np.asarray([str(lbl) for lbl in y[keep]], dtype=object)
    subject = subject[keep]
    rec_id = rec_id[keep]

    split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    groups = validate_split(split, available=set(subject.tolist()))
    train_subjects = groups["train_subjects"]
    validation_subjects = groups["validation_subjects"]
    test_subjects = groups["test_subjects"]  # RESERVED; never evaluated here.

    train_mask = np.isin(subject, list(train_subjects))
    val_mask = np.isin(subject, list(validation_subjects))
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    subj_val = subject[val_mask]
    if len(y_train) == 0 or len(y_val) == 0:
        raise RuntimeError("Empty training or validation data")
    if len(set(y_train)) < 2:
        raise ValueError("Training set must contain at least two classes")
    # Strictly no fitting, predictions or inspection on final TEST data.
    out_root.mkdir(parents=True, exist_ok=True)

    print("=== Day 10: train models ===")
    print(f"DATASET: {dataset_path.resolve()}")
    print(f"OUTDIR : {out_root.resolve()}")
    print(f"Train subjects: {len(train_subjects)} | Validation subjects: {len(validation_subjects)} | RESERVED test subjects: {len(test_subjects)}")
    print(f"Train epochs  : {len(y_train)} | Validation epochs: {len(y_val)} | Test epochs: NOT ACCESSED")
    print(f"Models        : {args.models}")

    models = build_models(args.models, seed=int(args.seed), class_weight=str(args.class_weight))
    results_rows = []

    for name, model in models.items():
        model_dir = out_root / name
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Training: {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = float(accuracy_score(y_val, y_pred))
        bacc = float(balanced_accuracy_score(y_val, y_pred))
        macro_f1 = float(f1_score(y_val, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0))

        report = classification_report(y_val, y_pred, labels=LABEL_ORDER, zero_division=0)
        cm = confusion_matrix(y_val, y_pred, labels=LABEL_ORDER)

        (model_dir / "report.txt").write_text(report + "\n\n" + str(cm), encoding="utf-8")
        plot_confusion(cm, model_dir / "confusion_matrix.png", title=f"Confusion matrix ({name}, participant-disjoint validation)")

        df_ps = per_subject_accuracy(y_val, y_pred, subj_val)
        df_ps.to_csv(model_dir / "per_subject_validation.csv", index=False)

        # Ensure runner compatibility
        ensure_predict_proba(model)

        bundle = {
            "model": model,
            "feature_spec": feature_spec,
            "channel": meta["channel"],
            "epoch_sec": float(meta["epoch_sec"]),
            "preproc": meta.get("preproc", {}),
            "n_features": int(X.shape[1]),
            "bundle_metadata_version": 2,
            "feature_dtype": "float32",
            "source_dataset": str(dataset_path),
            "split": {
                "train_subjects": sorted(list(train_subjects)),
                "validation_subjects": sorted(list(validation_subjects)),
                "test_subjects": sorted(list(test_subjects)),
                "id_semantics": "sleepedf-person-v2",
                "test_evaluated": False,
                "seed": int(args.seed),
            },
            "training": {
                "model_name": name,
                "class_weight": str(args.class_weight),
            },
        }

        model_path = model_dir / "stager.joblib"
        joblib.dump(bundle, model_path)

        results_rows.append({
            "model": name,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_f1": macro_f1,
            "model_path": str(model_path),
        })

        print(f"Saved: {model_path.resolve()}")
        print(f"VAL acc={acc:.3f} | bal_acc={bacc:.3f} | macro_f1={macro_f1:.3f}")

    pd.DataFrame(results_rows).sort_values("macro_f1", ascending=False).to_csv(out_root / "staging_summary.csv", index=False)
    print(f"\nSaved summary: {(out_root / 'staging_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
