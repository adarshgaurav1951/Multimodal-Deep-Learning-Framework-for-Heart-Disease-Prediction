"""
Unit tests for ECG and Clinical Data Processing and Leakage Prevention.
"""
import pytest
import numpy as np
import pandas as pd
from src.data.ecg_processor import ECGProcessor
from src.data.clinical_processor import ClinicalProcessor
from src.data.dataset import patient_level_split


def test_ecg_validation_valid():
    proc = ECGProcessor()
    dummy_ecg = np.random.randn(12, 1000).astype(np.float32)
    validated = proc.validate(dummy_ecg)
    assert validated.shape == (12, 1000)


def test_ecg_validation_rejects_nan():
    proc = ECGProcessor()
    dummy_ecg = np.random.randn(12, 1000)
    dummy_ecg[0, 50] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        proc.validate(dummy_ecg)


def test_ecg_validation_rejects_short():
    proc = ECGProcessor()
    dummy_ecg = np.random.randn(12, 50)
    with pytest.raises(ValueError, match="at least 100 samples"):
        proc.validate(dummy_ecg)


def test_ecg_filtering_and_resampling():
    proc = ECGProcessor(target_fs=500.0, nominal_samples=2500)
    raw_ecg = np.random.randn(12, 1500)
    processed = proc.process(raw_ecg, orig_fs=250.0)
    assert processed.shape == (12, 2500)
    assert np.isfinite(processed).all()


def test_clinical_processor_train_only_fit():
    train_records = [
        {"age": 50.0, "sex": 1.0, "cholesterol": 200.0, "resting_bp": 120.0},
        {"age": 60.0, "sex": 0.0, "cholesterol": 250.0, "resting_bp": 140.0},
        {"age": 70.0, "sex": 1.0, "cholesterol": None, "resting_bp": 160.0}
    ]
    proc = ClinicalProcessor(feature_names=["age", "sex", "cholesterol", "resting_bp"])
    proc.fit(train_records)

    # Median of [200.0, 250.0] is 225.0
    assert proc.impute_values_["cholesterol"] == 225.0

    # Transform record with missing cholesterol
    test_record = {"age": 55.0, "sex": 1.0, "cholesterol": None, "resting_bp": 130.0}
    transformed = proc.transform(test_record)
    assert transformed.shape == (4,)
    assert np.isfinite(transformed).all()


def test_clinical_processor_serialization():
    train_df = pd.DataFrame({
        "age": [45, 55, 65],
        "sex": [1, 0, 1],
        "resting_bp": [120, 130, 140]
    })
    proc1 = ClinicalProcessor(feature_names=["age", "sex", "resting_bp"])
    proc1.fit(train_df)

    state = proc1.state_dict()
    proc2 = ClinicalProcessor()
    proc2.load_state_dict(state)

    assert proc2.feature_names == proc1.feature_names
    assert proc2.impute_values_ == proc1.impute_values_
    assert proc2.means_ == proc1.means_


def test_patient_level_split_no_leakage():
    # 20 patients with multiple records each
    records = []
    for p in range(20):
        pat_id = f"PAT_{p:02d}"
        for r in range(np.random.randint(2, 5)):
            records.append({"record_id": f"{pat_id}_{r}", "patient_id": pat_id, "target": p % 2})
    df = pd.DataFrame(records)

    train_idx, val_idx, test_idx = patient_level_split(df, val_size=0.15, test_size=0.15, seed=42)

    train_pts = set(df.iloc[train_idx]["patient_id"])
    val_pts = set(df.iloc[val_idx]["patient_id"])
    test_pts = set(df.iloc[test_idx]["patient_id"])

    # Verify zero intersection
    assert len(train_pts.intersection(val_pts)) == 0
    assert len(train_pts.intersection(test_pts)) == 0
    assert len(val_pts.intersection(test_pts)) == 0


def test_save_and_load_cohort_csv(tmp_path):
    from src.data.csv_loader import save_cohort_to_csv, load_cohort_from_csv

    clin_df = pd.DataFrame({
        "record_id": [f"REC_{i:03d}" for i in range(10)],
        "patient_id": [f"PAT_{i:03d}" for i in range(10)],
        "age": [50 + i for i in range(10)],
        "sex": [1.0 if i % 2 == 0 else 0.0 for i in range(10)],
        "target": [i % 2 for i in range(10)]
    })
    ecgs = np.random.randn(10, 12, 500).astype(np.float32)
    saved_files = save_cohort_to_csv(clin_df, ecgs, data_dir=tmp_path)

    assert saved_files["clinical"].exists()
    assert saved_files["multimodal"].exists()

    # Load from directory
    loaded_clin, loaded_ecgs = load_cohort_from_csv(tmp_path, nominal_samples=500)
    assert len(loaded_clin) == len(clin_df)
    assert loaded_ecgs.shape == ecgs.shape
    assert set(loaded_clin["patient_id"]) == set(clin_df["patient_id"])

    # Load from single multimodal CSV
    loaded_clin_single, loaded_ecgs_single = load_cohort_from_csv(saved_files["multimodal"], nominal_samples=500)
    assert len(loaded_clin_single) == len(clin_df)
    assert loaded_ecgs_single.shape == ecgs.shape


def test_csv_loader_rejects_non_csv(tmp_path):
    from src.data.csv_loader import load_cohort_from_csv

    fake_db = tmp_path / "data.db"
    fake_db.write_text("fake binary database content")

    with pytest.raises(ValueError, match="All data must be in CSV format"):
        load_cohort_from_csv(fake_db)

