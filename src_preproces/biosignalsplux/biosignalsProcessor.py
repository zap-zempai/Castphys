from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import glob
from typing import List

def loadSignals(filename):
    """
    This function loads the signals and the time as
    datetime
    """
    bio_data = np.load(filename)

    ppg = bio_data[:, 0]
    eda = bio_data[:, 1]
    breath = bio_data[:, 2]
    t_timestamp = bio_data[:, -1]
    
    return {
        "ppg": ppg,
        "eda": eda,
        "breath": breath,
        "time": t_timestamp
    }

def getBatchFromSignal(signals, num_samples, offset=0):
    """
    Provides the number of samples from the signals dictionary.
    You may use offset to provide the starting sample.
    """
    return {k: v[offset:offset+num_samples]
            for k, v in signals.items()}

def get_all_biosignal_files(folder: Path) -> List[Path]:
    """
    Get all biosignal files available in base_dir. This files
    are numpy files with extension ".npy"
    """
    csvs = glob.glob("*.npy", root_dir=folder)

    if len(csvs) < 1:
        raise Exception(f"No biosignals files were found in {folder}")
    
    return [(folder / f) for f in csvs]


def open_signals_npy(folder) -> pd.DataFrame:
        csvs = get_all_biosignal_files(folder)

        signals_dict = {
            "ppg": np.array([]),
            "eda": np.array([]),
            "breath": np.array([]),
            "time": []
        }

        for csv in csvs:
            signals = loadSignals(csv)

            ppg = signals["ppg"]
            eda = signals["eda"]
            breath = signals["breath"]
            ttime = signals["time"]

            signals_dict["ppg"] = np.concatenate([signals_dict["ppg"], ppg])
            signals_dict["eda"] = np.concatenate([signals_dict["eda"], eda])
            signals_dict["breath"] = np.concatenate([signals_dict["breath"], breath])
            signals_dict["time"] = np.concatenate([signals_dict["time"], ttime])
        
        # Change time to Datetime
        signals_dict["time"] = [datetime.fromtimestamp(t) for t in signals_dict["time"]]
        
        return pd.DataFrame.from_dict(signals_dict, orient="columns")
