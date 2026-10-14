"""
smart_home_utils.py
--------------------------------------------------------------------------
Custom module for the "Smart Home Analytics" college PBL project.

This module is imported by app.py and demonstrates, in one place, almost
every syllabus topic required for the project:

    1.  Python core concepts (functions, loops, conditionals, f-strings)
    2.  Data structures        -> list, tuple, set, dictionary
    3.  Python libraries       -> numpy, pandas, scipy, sklearn, requests, bs4
    4.  Custom modules         -> this file itself
    5.  Exception handling     -> custom exceptions + try/except everywhere
    6.  Web scraping           -> Requests + BeautifulSoup
    7.  File handling (CSV)    -> save_data / load_data
    8.  CRUD operations        -> create_record / update_record / delete_record
    9.  NumPy                  -> numpy_analysis()
    10. SciPy                  -> scipy_analysis(), hypothesis_test_two_devices()
    11. Pandas                 -> used throughout (DataFrame operations)
    12. Data preprocessing     -> preprocess_data()
    13. Data visualization     -> chart-ready aggregations (charts built in app.py)
    14. Statistical analysis   -> descriptive_statistics()
    15. Probability            -> probability_device_on(), normal_probability_density()
    16. Sampling & inference   -> random_sample_with_ci()
    17. Hypothesis testing     -> hypothesis_test_two_devices()
    18. Machine learning       -> train_energy_model(), predict_energy()
--------------------------------------------------------------------------
"""

import os
import random
import statistics
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# ===========================================================================
# DATA STRUCTURES  (Syllabus Topic #2: list, tuple, set, dictionary)
# ===========================================================================

ROOMS = ["Kitchen", "Living Room", "Bedroom", "Bathroom",
         "Garage", "Office", "Laundry Room"]                     # list

DEVICE_TYPES = (                                                 # tuple (immutable)
    "Refrigerator", "Air Conditioner", "Washing Machine", "Television",
    "Water Heater", "Microwave", "Lighting", "Space Heater",
    "Ceiling Fan", "Wi-Fi Router",
)

DEVICE_STATUSES = {"On", "Off", "Standby"}                       # set (unique values)

# dictionary: approximate base power draw (kWh per hour) per device type
DEVICE_BASE_ENERGY = {
    "Refrigerator": 0.15, "Air Conditioner": 1.50, "Washing Machine": 0.60,
    "Television": 0.12, "Water Heater": 2.00, "Microwave": 0.35,
    "Lighting": 0.06, "Space Heater": 1.80, "Ceiling Fan": 0.07,
    "Wi-Fi Router": 0.01,
}

CSV_FILE = "smart_home_data.csv"
ELECTRICITY_RATE_PER_KWH = 8.0     # currency units per kWh, used for cost KPI


# ===========================================================================
# CUSTOM EXCEPTIONS  (Syllabus Topic #5)
# ===========================================================================

class DataValidationError(Exception):
    """Raised when a record fails validation rules before CRUD write."""
    pass


class RecordNotFoundError(Exception):
    """Raised when a requested Device_ID does not exist in the dataset."""
    pass


# ===========================================================================
# SAMPLE DATA GENERATION
# ===========================================================================

def generate_sample_data(n_records: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic smart-home dataset using NumPy."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    rows = []
    start_time = datetime.now() - timedelta(days=30)

    for i in range(n_records):                                   # core Python loop
        device = random.choice(DEVICE_TYPES)
        room = random.choice(ROOMS)
        status = random.choice(list(DEVICE_STATUSES))

        temperature = round(float(rng.normal(24, 4)), 1)          # deg C
        humidity = float(np.clip(rng.normal(50, 12), 20, 90))
        humidity = round(humidity, 1)
        usage_hours = float(np.clip(abs(rng.normal(4, 2.5)), 0, 24))
        usage_hours = round(usage_hours, 2)

        base = DEVICE_BASE_ENERGY.get(device, 0.5)
        noise = float(rng.normal(0, 0.15))
        energy = max(0.01, round(base * usage_hours + noise, 3))

        timestamp = start_time + timedelta(minutes=int(rng.integers(0, 30 * 24 * 60)))

        rows.append({
            "Device_ID": i + 1,
            "Device": device,
            "Room": room,
            "Temperature": temperature,
            "Humidity": humidity,
            "Energy_kWh": energy,
            "Usage_Hours": usage_hours,
            "Status": status,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        })

    return pd.DataFrame(rows)


def ensure_dataset_exists(path: str = CSV_FILE) -> pd.DataFrame:
    """Load the dataset from CSV. If it is missing/empty/corrupt, auto-generate
    a fresh sample dataset and save it (Syllabus: File Handling + Exceptions)."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV file is empty.")
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError):
        df = generate_sample_data()
        save_data(df, path)
        return df
    except Exception as e:                                        # catch-all safety net
        print(f"Unexpected error loading dataset, regenerating. Details: {e}")
        df = generate_sample_data()
        save_data(df, path)
        return df


# ===========================================================================
# FILE HANDLING + CRUD OPERATIONS  (Syllabus Topics #7, #8)
# ===========================================================================

def save_data(df: pd.DataFrame, path: str = CSV_FILE) -> None:
    """Persist the DataFrame to a CSV file on disk."""
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        raise IOError(f"Failed to save CSV file '{path}': {e}")


def load_data(path: str = CSV_FILE) -> pd.DataFrame:
    """Read a DataFrame from CSV, raising clear errors on failure."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at '{path}'.")
    except pd.errors.ParserError:
        raise ValueError("CSV file appears corrupted and cannot be parsed.")


def validate_record(record: dict) -> None:
    """Validate a record dict before it is inserted or updated."""
    required = ["Device", "Room", "Temperature", "Humidity",
                "Energy_kWh", "Usage_Hours", "Status", "Timestamp"]
    for field in required:
        if field not in record or record[field] in (None, ""):
            raise DataValidationError(f"Missing required field: {field}")
    if record["Status"] not in DEVICE_STATUSES:
        raise DataValidationError(f"Invalid status '{record['Status']}'.")
    try:
        float(record["Energy_kWh"])
        float(record["Temperature"])
        float(record["Humidity"])
        float(record["Usage_Hours"])
    except (ValueError, TypeError):
        raise DataValidationError("Numeric fields must contain valid numbers.")


def create_record(df: pd.DataFrame, record: dict) -> pd.DataFrame:
    """CREATE - add a new device record and return the updated DataFrame."""
    validate_record(record)
    new_id = int(df["Device_ID"].max()) + 1 if not df.empty else 1
    record = dict(record)
    record["Device_ID"] = new_id
    new_row = pd.DataFrame([record])
    return pd.concat([df, new_row], ignore_index=True)


def update_record(df: pd.DataFrame, device_id: int, updates: dict) -> pd.DataFrame:
    """UPDATE - edit an existing record identified by Device_ID."""
    if device_id not in df["Device_ID"].values:
        raise RecordNotFoundError(f"Device_ID {device_id} not found.")
    for key, value in updates.items():
        if key in df.columns:
            df.loc[df["Device_ID"] == device_id, key] = value
    return df


def delete_record(df: pd.DataFrame, device_id: int) -> pd.DataFrame:
    """DELETE - remove a record identified by Device_ID."""
    if device_id not in df["Device_ID"].values:
        raise RecordNotFoundError(f"Device_ID {device_id} not found.")
    return df[df["Device_ID"] != device_id].reset_index(drop=True)


def search_records(df: pd.DataFrame, device=None, room=None, status=None,
                    keyword: str = None) -> pd.DataFrame:
    """SEARCH / FILTER records by optional criteria."""
    result = df.copy()
    try:
        if device and device != "All":
            result = result[result["Device"] == device]
        if room and room != "All":
            result = result[result["Room"] == room]
        if status and status != "All":
            result = result[result["Status"] == status]
        if keyword:
            mask = (result["Device"].str.contains(keyword, case=False, na=False) |
                    result["Room"].str.contains(keyword, case=False, na=False))
            result = result[mask]
    except KeyError as e:
        raise KeyError(f"Column not found while filtering: {e}")
    return result


# ===========================================================================
# DATA PREPROCESSING  (Syllabus Topic #12)
# ===========================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset: handle missing values, duplicates, outliers,
    data types, and engineer a couple of useful derived features."""
    df = df.copy()
    numeric_cols = ["Temperature", "Humidity", "Energy_kWh", "Usage_Hours"]

    # 1. Handle missing values
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    df["Device"] = df["Device"].fillna("Unknown")
    df["Room"] = df["Room"].fillna("Unknown")
    df["Status"] = df["Status"].fillna("Off")

    # 2. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 3. Outlier capping using the IQR method
    for col in numeric_cols:
        if col in df.columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            df[col] = df[col].clip(lower, upper)

    # 4. Feature engineering
    df["Efficiency_kWh_per_hr"] = df["Energy_kWh"] / df["Usage_Hours"].replace(0, np.nan)
    df["Efficiency_kWh_per_hr"] = df["Efficiency_kWh_per_hr"].fillna(0)

    # 5. Encode status as a numeric code (useful for EDA / ML)
    status_map = {"Off": 0, "Standby": 1, "On": 2}
    df["Status_Code"] = df["Status"].map(status_map).fillna(0).astype(int)

    return df.reset_index(drop=True)


# ===========================================================================
# NUMPY ANALYSIS  (Syllabus Topic #9)
# ===========================================================================

def numpy_analysis(df: pd.DataFrame) -> dict:
    """Perform NumPy-based numerical analysis on the Energy_kWh column."""
    arr = df["Energy_kWh"].to_numpy()
    return {
        "sum": float(np.sum(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "percentile_25": float(np.percentile(arr, 25)),
        "percentile_75": float(np.percentile(arr, 75)),
        "cumulative_sum_tail": np.cumsum(arr)[-5:].tolist(),
    }


# ===========================================================================
# SCIPY STATISTICAL ANALYSIS  (Syllabus Topic #10)
# ===========================================================================

def scipy_analysis(df: pd.DataFrame) -> dict:
    """SciPy-based descriptive statistics: variance, skewness, kurtosis, mode."""
    arr = df["Energy_kWh"].to_numpy()
    desc = scipy_stats.describe(arr)
    mode_result = scipy_stats.mode(np.round(arr, 1), keepdims=True)
    return {
        "n": int(desc.nobs),
        "variance": float(desc.variance),
        "skewness": float(desc.skewness),
        "kurtosis": float(desc.kurtosis),
        "mode_value": float(mode_result.mode[0]),
        "mode_count": int(mode_result.count[0]),
    }


# ===========================================================================
# STATISTICAL ANALYSIS  (Syllabus Topic #14)
# ===========================================================================

def descriptive_statistics(series: pd.Series) -> dict:
    """Compute mean, median, mode, variance, and standard deviation."""
    values = series.dropna().tolist()
    if not values:
        return {"mean": 0, "median": 0, "mode": 0, "variance": 0, "std_dev": 0}
    try:
        mode_val = statistics.mode(values)
    except statistics.StatisticsError:
        mode_val = values[0]

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "mode": mode_val,
        "variance": statistics.variance(values) if len(values) > 1 else 0,
        "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
    }


# ===========================================================================
# PROBABILITY  (Syllabus Topic #15)
# ===========================================================================

def probability_device_on(df: pd.DataFrame) -> float:
    """Empirical probability that a randomly chosen record has Status == 'On'."""
    if len(df) == 0:
        return 0.0
    return float((df["Status"] == "On").sum() / len(df))


def probability_energy_above_threshold(df: pd.DataFrame, threshold: float) -> float:
    """Empirical probability that Energy_kWh exceeds a given threshold."""
    if len(df) == 0:
        return 0.0
    return float((df["Energy_kWh"] > threshold).sum() / len(df))


def normal_probability_density(x, mean, std):
    """Normal distribution PDF value at x (demonstrates probability theory)."""
    return scipy_stats.norm.pdf(x, loc=mean, scale=std)


# ===========================================================================
# SAMPLING & STATISTICAL INFERENCE  (Syllabus Topic #16)
# ===========================================================================

def random_sample_with_ci(df: pd.DataFrame, sample_size: int = 30,
                           confidence: float = 0.95, seed: int = 1) -> dict:
    """Draw a random sample of Energy_kWh values and compute a confidence
    interval for the population mean using the t-distribution."""
    sample_size = max(2, min(sample_size, len(df)))
    sample = df["Energy_kWh"].sample(n=sample_size, random_state=seed)

    mean = sample.mean()
    sem = scipy_stats.sem(sample)  # standard error of the mean
    ci_low, ci_high = scipy_stats.t.interval(
        confidence, df=len(sample) - 1, loc=mean, scale=sem
    )
    return {
        "sample": sample,
        "sample_mean": float(mean),
        "sample_std": float(sample.std()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence": confidence,
    }


# ===========================================================================
# HYPOTHESIS TESTING  (Syllabus Topic #17)
# ===========================================================================

def hypothesis_test_two_devices(df: pd.DataFrame, device_a: str, device_b: str) -> dict:
    """Independent two-sample t-test comparing Energy_kWh between two devices.

    H0: mean energy of device_a == mean energy of device_b
    H1: mean energy of device_a != mean energy of device_b
    """
    group_a = df.loc[df["Device"] == device_a, "Energy_kWh"].dropna()
    group_b = df.loc[df["Device"] == device_b, "Energy_kWh"].dropna()

    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("Not enough data points for the selected devices.")

    t_stat, p_value = scipy_stats.ttest_ind(group_a, group_b, equal_var=False)
    alpha = 0.05
    reject_null = bool(p_value < alpha)

    conclusion = (
        f"Reject H0: '{device_a}' and '{device_b}' have significantly different "
        f"mean energy consumption (p={p_value:.4f} < {alpha})."
        if reject_null else
        f"Fail to reject H0: no significant difference in mean energy "
        f"consumption between '{device_a}' and '{device_b}' (p={p_value:.4f} >= {alpha})."
    )

    return {
        "device_a": device_a, "device_b": device_b,
        "mean_a": float(group_a.mean()), "mean_b": float(group_b.mean()),
        "t_statistic": float(t_stat), "p_value": float(p_value),
        "alpha": alpha, "reject_null": reject_null, "conclusion": conclusion,
    }


# ===========================================================================
# WEB SCRAPING  (Syllabus Topic #6)
# ===========================================================================

def scrape_electricity_info() -> dict:
    """Scrape a public Wikipedia page about electricity pricing using
    Requests + BeautifulSoup. Falls back to static data on any error so the
    dashboard never crashes if the machine is offline."""
    url = "https://en.wikipedia.org/wiki/Electricity_pricing"
    headers = {"User-Agent": "Mozilla/5.0 (SmartHomeAnalyticsProject/1.0)"}

    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else "N/A"

        first_paragraph = ""
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if text and len(text) > 60:
                first_paragraph = text
                break

        headings = [h.get_text(strip=True) for h in soup.find_all("h2")][:8]

        return {
            "source_url": url,
            "page_title": title,
            "summary": first_paragraph[:600],
            "section_headings": headings,
            "status": "success",
        }
    except requests.exceptions.RequestException as e:
        return {
            "source_url": url,
            "page_title": "Electricity Pricing (offline fallback data)",
            "summary": ("Could not reach the internet, so fallback data is shown. "
                        "Electricity pricing generally depends on generation cost, "
                        "transmission and distribution charges, and government tariffs."),
            "section_headings": ["Overview", "Tariff structures", "Regional pricing"],
            "status": f"failed: {e}",
        }
    except Exception as e:
        return {
            "source_url": url, "page_title": "Error",
            "summary": f"Unexpected scraping error: {e}",
            "section_headings": [], "status": "error",
        }


# ===========================================================================
# MACHINE LEARNING  (Syllabus Topic #18)
# ===========================================================================

def train_energy_model(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> dict:
    """Train a Linear Regression model that predicts Energy_kWh from
    Temperature, Humidity, and Usage_Hours, with a train/test split and
    feature scaling."""
    features = ["Temperature", "Humidity", "Usage_Hours"]
    target = "Energy_kWh"

    data = df.dropna(subset=features + [target])
    if len(data) < 10:
        raise ValueError("Not enough data to train a reliable model.")

    X = data[features].values
    y = data[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model, "scaler": scaler, "features": features,
        "mae": float(mae), "r2": float(r2),
        "coefficients": dict(zip(features, model.coef_.tolist())),
        "intercept": float(model.intercept_),
        "y_test": y_test, "y_pred": y_pred,
    }


def predict_energy(model, scaler, temperature: float, humidity: float,
                    usage_hours: float) -> float:
    """Predict energy consumption (kWh) for new input values."""
    try:
        x_new = np.array([[temperature, humidity, usage_hours]])
        x_new_scaled = scaler.transform(x_new)
        prediction = model.predict(x_new_scaled)[0]
        return float(max(0.0, prediction))
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")


# ===========================================================================
# KPI HELPERS
# ===========================================================================

def compute_kpis(df: pd.DataFrame, rate_per_kwh: float = ELECTRICITY_RATE_PER_KWH) -> dict:
    """Compute the KPI values shown on the Dashboard page."""
    total_energy = df["Energy_kWh"].sum()
    avg_energy = df["Energy_kWh"].mean()
    estimated_cost = total_energy * rate_per_kwh

    device_totals = df.groupby("Device")["Energy_kWh"].sum().sort_values(ascending=False)
    top_device = device_totals.index[0] if not device_totals.empty else "N/A"
    top_device_value = float(device_totals.iloc[0]) if not device_totals.empty else 0.0

    return {
        "total_energy": float(total_energy),
        "avg_energy": float(avg_energy),
        "estimated_cost": float(estimated_cost),
        "top_device": top_device,
        "top_device_value": top_device_value,
        "total_records": len(df),
    }
