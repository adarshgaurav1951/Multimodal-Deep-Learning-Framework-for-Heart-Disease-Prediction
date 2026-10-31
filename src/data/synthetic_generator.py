"""
Synthetic Physiological & Clinical Data Generator.
Produces physiologically plausible 12-lead ECG waveforms and correlated clinical variables
for testing, benchmarking, and demonstration purposes.
"""
from __future__ import annotations

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from src.data.contracts import STANDARD_12_LEADS, DEFAULT_CLINICAL_FEATURES


def generate_ecg_beat(
    fs: float = 500.0,
    hr: float = 75.0,
    st_elev_depression: float = 0.0,
    t_invert: bool = False,
    noise_level: float = 0.02
) -> np.ndarray:
    """
    Generate a single synthetic ECG cardiac cycle using sum of Gaussians.
    """
    beat_duration = 60.0 / hr
    t = np.linspace(0, beat_duration, int(fs * beat_duration), endpoint=False)
    center = beat_duration * 0.4
    
    # Baseline
    ecg = np.zeros_like(t)

    # P-wave (atrium depolarization)
    p_center = center - 0.16
    ecg += 0.15 * np.exp(-((t - p_center) ** 2) / (2 * (0.025 ** 2)))

    # Q-wave (septal depolarization)
    q_center = center - 0.04
    ecg -= 0.15 * np.exp(-((t - q_center) ** 2) / (2 * (0.01 ** 2)))

    # R-wave (ventricular depolarization)
    r_center = center
    ecg += 1.2 * np.exp(-((t - r_center) ** 2) / (2 * (0.015 ** 2)))

    # S-wave
    s_center = center + 0.04
    ecg -= 0.25 * np.exp(-((t - s_center) ** 2) / (2 * (0.012 ** 2)))

    # ST segment shift
    st_center = center + 0.12
    ecg += st_elev_depression * np.exp(-((t - st_center) ** 2) / (2 * (0.06 ** 2)))

    # T-wave (ventricular repolarization)
    t_center = center + 0.24
    t_amp = -0.35 if t_invert else 0.35
    ecg += t_amp * np.exp(-((t - t_center) ** 2) / (2 * (0.045 ** 2)))

    # Add Gaussian thermal noise
    if noise_level > 0:
        ecg += np.random.normal(0, noise_level, size=len(t))

    return ecg


def generate_12lead_ecg(
    duration_sec: float = 5.0,
    fs: float = 500.0,
    hr: float = 72.0,
    is_pathological: bool = False
) -> np.ndarray:
    """
    Generate 12-lead ECG waveform array of shape [12, samples].
    Pathological cases introduce ST deviation and T-wave abnormalities in anterior/inferior leads.
    """
    total_samples = int(duration_sec * fs)
    lead_ecgs = np.zeros((12, total_samples), dtype=np.float32)

    # Base beat parameters
    st_dev = np.random.uniform(-0.35, -0.15) if is_pathological else np.random.uniform(-0.02, 0.02)
    t_inv = True if is_pathological and np.random.rand() > 0.4 else False

    # Lead projection scale factors simulating Einthoven and precordial spatial geometry
    # Leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    lead_scales = [0.7, 1.0, 0.5, -0.85, 0.3, 0.75, -0.4, 0.2, 0.8, 1.1, 1.0, 0.8]
    lead_pathology_weights = [0.3, 0.9, 0.8, 0.2, 0.1, 0.85, 0.4, 0.6, 0.9, 1.0, 0.8, 0.5]

    single_beat = generate_ecg_beat(fs=fs, hr=hr, st_elev_depression=st_dev, t_invert=t_inv)
    beat_len = len(single_beat)
    num_beats = int(np.ceil(total_samples / beat_len)) + 1
    repeated = np.tile(single_beat, num_beats)[:total_samples]

    # Respiration baseline wander
    time_axis = np.linspace(0, duration_sec, total_samples)
    respiration = 0.08 * np.sin(2 * np.pi * 0.25 * time_axis)

    for i in range(12):
        scale = lead_scales[i]
        path_weight = lead_pathology_weights[i] if is_pathological else 0.0
        lead_signal = scale * repeated + (path_weight * st_dev) + respiration + np.random.normal(0, 0.015, total_samples)
        lead_ecgs[i] = lead_signal.astype(np.float32)

    return lead_ecgs


def generate_synthetic_multimodal_cohort(
    num_patients: int = 250,
    records_per_patient: int = 2,
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic dataset with realistic correlations between clinical metrics,
    ECG morphologies, and disease status.
    
    Returns:
        clinical_df: DataFrame with record_id, patient_id, clinical features, and target
        ecg_array: np.ndarray of shape [N, 12, 2500] (500Hz x 5s)
    """
    np.random.seed(seed)
    records = []
    ecg_list = []

    for p_idx in range(num_patients):
        patient_id = f"PAT_{p_idx + 1:04d}"
        
        # Latent patient cardiovascular risk factor
        base_risk = np.random.beta(2, 3)  # Risk probability
        
        # Clinical variables correlated with risk
        age = int(np.clip(np.random.normal(55 + 15 * base_risk, 10), 25, 85))
        sex = 1.0 if np.random.rand() < (0.55 + 0.2 * base_risk) else 0.0
        resting_bp = float(np.clip(np.random.normal(120 + 35 * base_risk, 18), 90, 210))
        cholesterol = float(np.clip(np.random.normal(200 + 70 * base_risk, 40), 120, 480))
        def safe_choice(opts, probs):
            p = np.array(probs, dtype=np.float64)
            p = np.maximum(p, 0.01)
            p = p / np.sum(p)
            return float(np.random.choice(opts, p=p))

        fasting_bs = 1.0 if np.random.rand() < (0.10 + 0.35 * base_risk) else 0.0
        resting_ecg = safe_choice([0.0, 1.0, 2.0], [0.7 - 0.5 * base_risk, 0.2 + 0.2 * base_risk, 0.1 + 0.3 * base_risk])
        max_hr = float(np.clip(np.random.normal(170 - 0.7 * age - 25 * base_risk, 20), 70, 210))
        exercise_angina = 1.0 if np.random.rand() < (0.15 + 0.55 * base_risk) else 0.0
        st_depression = float(np.clip(np.random.exponential(0.5 + 2.0 * base_risk), 0.0, 6.5))
        st_slope = safe_choice([0.0, 1.0, 2.0], [0.5 - 0.4 * base_risk, 0.3 + 0.2 * base_risk, 0.2 + 0.2 * base_risk])
        num_major_vessels = safe_choice([0.0, 1.0, 2.0, 3.0], [0.6 - 0.5 * base_risk, 0.2 + 0.1 * base_risk, 0.1 + 0.2 * base_risk, 0.1 + 0.2 * base_risk])
        thal = safe_choice([0.0, 1.0, 2.0], [0.6 - 0.5 * base_risk, 0.1 + 0.1 * base_risk, 0.3 + 0.4 * base_risk])

        # Target label based on joint latent factors
        logit = -2.5 + 2.8 * base_risk + 0.03 * (age - 55) + 0.4 * sex + 0.015 * (resting_bp - 120) + 0.6 * st_depression + 0.8 * exercise_angina + 0.5 * num_major_vessels
        disease_prob = 1.0 / (1.0 + np.exp(-logit))
        target = 1 if np.random.rand() < disease_prob else 0

        # Generate records for this patient
        n_records = np.random.randint(1, records_per_patient + 1)
        for r_sub in range(n_records):
            record_id = f"REC_{p_idx + 1:04d}_{r_sub + 1}"
            
            # ECG for this record (reflects pathology with slight noise/hr fluctuation)
            record_hr = np.clip(max_hr * 0.45 + np.random.normal(0, 4), 50, 115)
            ecg_12 = generate_12lead_ecg(
                duration_sec=5.0,
                fs=500.0,
                hr=record_hr,
                is_pathological=(target == 1)
            )

            # Randomly simulate real-world small missingness in 3% of clinical measurements
            record_features = {
                "record_id": record_id,
                "patient_id": patient_id,
                "age": age,
                "sex": sex,
                "resting_bp": resting_bp if np.random.rand() > 0.03 else None,
                "cholesterol": cholesterol if np.random.rand() > 0.04 else None,
                "fasting_bs": fasting_bs,
                "resting_ecg": resting_ecg,
                "max_hr": max_hr,
                "exercise_angina": exercise_angina,
                "st_depression": st_depression,
                "st_slope": st_slope,
                "num_major_vessels": num_major_vessels if np.random.rand() > 0.02 else None,
                "thal": thal,
                "target": target
            }
            records.append(record_features)
            ecg_list.append(ecg_12)

    clinical_df = pd.DataFrame(records)
    ecg_array = np.stack(ecg_list, axis=0)  # [N, 12, 2500]

    return clinical_df, ecg_array
