"""
rerun_medical_classifiers.py
============================
Re-evaluates only the medical finetuned-me5 classifier experiments
(logreg + svm, nested-class structure) that previously failed with:
  "Nested classifier dictionary is missing the 'hospital' model."

Runs with overwrite=True so it replaces the empty/zero rows.
"""

from sklearn.model_selection import train_test_split
from orgpackage.aux import load_dataset
from orgpackage.orchestrator import classifier_orchestrator


def build_splits(data):
    """Exact same splits as run_finetuned_experiments.py."""
    train_full, _ = train_test_split(data, test_size=0.5, random_state=42)
    train, val    = train_test_split(train_full, test_size=0.5, random_state=42)
    trains      = {'medical': train, 'administrative': train, 'education': train}
    validations = {'medical': val,   'administrative': val,   'education': val}
    return trains, validations


if __name__ == "__main__":
    print("Loading dataset ...")
    data = load_dataset()
    trains, validations = build_splits(data)

    # Narrow orchestrator to medical + finetuned-me5 only, force overwrite
    classifier_orchestrator(
        {k: v for k, v in trains.items()      if k == 'medical'},
        {k: v for k, v in validations.items() if k == 'medical'},
        models=['finetuned-me5'],
        overwrite=True,
    )

    print("\nDone. Check results/experiments.csv for updated medical classifier rows.")
