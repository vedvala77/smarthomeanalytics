# 🏠 Smart Home Analytics — Python PBL Project

A college-level, professional **Streamlit** dashboard that analyzes smart-home
device data (Device, Room, Temperature, Humidity, Energy consumption, Usage
hours, Status, Timestamp). Built as a Project-Based Learning (PBL) submission
that demonstrates the **entire Python + Data Science syllabus** in one working
application.

---

## 1. Project Structure

```
smart_home_analytics/
├── app.py                  # Main Streamlit application (10 pages)
├── smart_home_utils.py     # Custom module: all core logic
├── requirements.txt        # Python dependencies
├── README.md                # This file
└── smart_home_data.csv     # Auto-generated on first run (not required upfront)
```

---

## 2. Syllabus Coverage Map

| # | Topic | Implemented In |
|---|-------|-----------------|
| 1 | Python Core Concepts | `smart_home_utils.py` (functions, loops, conditionals) |
| 2 | Data Structures (list, tuple, set, dict) | `ROOMS` (list), `DEVICE_TYPES` (tuple), `DEVICE_STATUSES` (set), `DEVICE_BASE_ENERGY` (dict) |
| 3 | Python Libraries | numpy, pandas, scipy, scikit-learn, requests, bs4, matplotlib, plotly |
| 4 | Custom Modules | `smart_home_utils.py` imported into `app.py` |
| 5 | Exception Handling | Custom exceptions (`DataValidationError`, `RecordNotFoundError`) + try/except throughout |
| 6 | Web Scraping | `scrape_electricity_info()` — Requests + BeautifulSoup |
| 7 | File Handling (CSV) | `save_data()`, `load_data()`, `ensure_dataset_exists()` |
| 8 | CRUD Operations | `create_record()`, `update_record()`, `delete_record()`, `search_records()` |
| 9 | NumPy | `numpy_analysis()` |
| 10 | SciPy | `scipy_analysis()`, `hypothesis_test_two_devices()` |
| 11 | Pandas | Used throughout for all tabular operations |
| 12 | Data Preprocessing | `preprocess_data()` — missing values, duplicates, outliers, encoding |
| 13 | Data Visualization | Visualization page — Matplotlib + Plotly charts |
| 14 | Statistical Analysis | `descriptive_statistics()` — mean, median, mode, variance, std dev |
| 15 | Probability | `probability_device_on()`, `probability_energy_above_threshold()`, `normal_probability_density()` |
| 16 | Sampling & Inference | `random_sample_with_ci()` — 95% confidence interval |
| 17 | Hypothesis Testing | `hypothesis_test_two_devices()` — independent t-test |
| 18 | Machine Learning | `train_energy_model()`, `predict_energy()` — scikit-learn Linear Regression |

---

## 3. Setup Instructions (VS Code, step by step)

### Step 1 — Create the project folder
Open VS Code → Terminal → New Terminal, then:
```bash
mkdir smart_home_analytics
cd smart_home_analytics
```

### Step 2 — Create the files
Inside this folder, create these four files (copy the code given to you into
each one exactly as provided):
- `app.py`
- `smart_home_utils.py`
- `requirements.txt`
- `README.md`

You can create them in VS Code via **File → New File**, or from the terminal:
```bash
# Windows (PowerShell)
New-Item app.py, smart_home_utils.py, requirements.txt, README.md

# macOS / Linux
touch app.py smart_home_utils.py requirements.txt README.md
```
Then paste the corresponding code into each file and save.

### Step 3 — Create and activate a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```
VS Code should auto-detect the `venv` — if not, select it via
**Ctrl+Shift+P → Python: Select Interpreter**.

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Run the Streamlit app
```bash
streamlit run app.py
```
This opens the dashboard automatically at `http://localhost:8501` in your
browser. On first run, if `smart_home_data.csv` doesn't exist, the app
**automatically generates** a realistic sample dataset for you — no manual
setup needed.

### Step 6 — Explore the app
Use the sidebar to navigate between the 10 pages: Dashboard, Dataset/CRUD,
Data Preprocessing, Statistical Analysis, Probability & Sampling, Hypothesis
Testing, Visualization, Web Scraping, Machine Learning, and About Project.

---

## 4. Notes for Your Presentation / Viva

- Every page has an **info box** at the top stating exactly which syllabus
  topics it demonstrates — point to these directly during evaluation.
- The **Dataset/CRUD** page writes every change back to `smart_home_data.csv`
  immediately, so you can show file persistence live.
- The **Web Scraping** page requires internet access; if offline, it
  gracefully falls back to static data via exception handling (it will not
  crash the app) — a good talking point about robust error handling.
- The **Machine Learning** page lets you train the model live and then enter
  your own Temperature/Humidity/Usage Hours values to get a real-time
  prediction.

---

## 5. Likely Viva Questions & Short Answers

1. **Q: Why did you use Streamlit instead of Flask/Django?**
   A: Streamlit lets you build interactive, data-focused dashboards with pure
   Python — no HTML/CSS/JS needed — making it ideal for rapid data-science UI
   prototyping.

2. **Q: What is a custom module and why did you create `smart_home_utils.py`?**
   A: A custom module is a separate `.py` file containing reusable functions
   imported elsewhere with `import`. It separates business logic from the UI,
   keeping code organized and testable.

3. **Q: Where have you used all four core data structures?**
   A: `ROOMS` is a list, `DEVICE_TYPES` is a tuple (immutable device names),
   `DEVICE_STATUSES` is a set (unique statuses), and `DEVICE_BASE_ENERGY` is a
   dictionary mapping each device to its base power draw.

4. **Q: How does your app handle exceptions?**
   A: Custom exceptions (`DataValidationError`, `RecordNotFoundError`) are
   raised for invalid CRUD input, and try/except blocks wrap file I/O,
   network calls, and statistical computations so the app never crashes and
   always shows a helpful message.

5. **Q: How does web scraping work in your project?**
   A: `requests.get()` fetches an HTML page, and `BeautifulSoup` parses the
   HTML to extract the title, a summary paragraph, and section headings. If
   the request fails (e.g., no internet), a fallback dictionary is returned.

6. **Q: What CRUD operations does the Dataset page support?**
   A: Create (add new device record), Read/Search (filter by device, room,
   status, keyword), Update (edit energy, hours, status of a record), and
   Delete (remove a record by Device_ID) — all persisted to CSV.

7. **Q: What is the difference between NumPy and Pandas in your project?**
   A: NumPy is used for fast numerical array operations (sum, mean, std,
   percentiles, cumulative sum) on raw arrays, while Pandas manages the whole
   tabular dataset (filtering, grouping, merging, CSV I/O).

8. **Q: What preprocessing steps did you apply?**
   A: Missing value imputation (median fill), duplicate removal, outlier
   capping using the IQR method, feature engineering (`Efficiency_kWh_per_hr`),
   and encoding `Status` into a numeric `Status_Code`.

9. **Q: Explain mean, median, mode, variance, and standard deviation as used here.**
   A: Mean is the average energy value; median is the middle value when
   sorted; mode is the most frequent value; variance measures the average
   squared deviation from the mean; standard deviation is the square root of
   variance, showing typical spread in the same units as the data.

10. **Q: How did you calculate probability in this project?**
    A: Empirically — e.g., P(Status = 'On') is the count of 'On' records
    divided by total records. We also plotted a theoretical Normal
    distribution (SciPy `norm.pdf`) fitted to the energy data's mean and
    standard deviation.

11. **Q: What is a confidence interval, and how did you compute it?**
    A: A confidence interval is a range likely to contain the true population
    parameter. We used `scipy.stats.t.interval()` on a random sample's mean
    and standard error to build a 95% CI for average energy consumption.

12. **Q: What hypothesis test did you use and why?**
    A: An independent two-sample t-test (`scipy.stats.ttest_ind`) with
    Welch's correction (`equal_var=False`), comparing mean energy consumption
    between two devices, since we don't assume the two groups have equal
    variance.

13. **Q: What are the null and alternative hypotheses in your test?**
    A: H0: the two devices have equal mean energy consumption. H1: their mean
    energy consumption differs. We reject H0 if p-value < 0.05.

14. **Q: What machine learning algorithm did you use and why?**
    A: Linear Regression, because the target (Energy_kWh) is continuous and
    we assume an approximately linear relationship with Temperature,
    Humidity, and Usage Hours — it's also simple to interpret via
    coefficients.

15. **Q: Why did you scale features before training?**
    A: `StandardScaler` standardizes features to zero mean and unit variance
    so that no single feature dominates due to differing units/scales,
    which especially helps gradient-based and distance-based learning.

16. **Q: What do MAE and R² mean in your model evaluation?**
    A: MAE (Mean Absolute Error) is the average absolute difference between
    predicted and actual energy values, in kWh. R² (coefficient of
    determination) shows the proportion of variance in energy consumption
    explained by the model (closer to 1 is better).

17. **Q: How is the train/test split performed?**
    A: `train_test_split()` from scikit-learn randomly divides the dataset
    (default 80/20) so the model is evaluated on unseen data, preventing
    overly optimistic performance estimates.

18. **Q: How does the prediction form work?**
    A: The user enters Temperature, Humidity, and Usage Hours; these are
    scaled with the same fitted `StandardScaler`, then passed to the trained
    `LinearRegression` model's `.predict()` method to output a kWh estimate.

19. **Q: How is the estimated electricity cost KPI calculated?**
    A: Total Energy (kWh) is multiplied by a fixed electricity rate per kWh
    (`ELECTRICITY_RATE_PER_KWH`), giving an estimated total cost.

20. **Q: What would you improve if you had more time?**
    A: Add more ML models (Random Forest, XGBoost) for comparison, real
    device API integration instead of synthetic data, user authentication for
    multi-household support, and time-series forecasting (e.g., ARIMA/Prophet)
    for future energy usage.

---

## 6. Tech Stack

Python · Streamlit · Pandas · NumPy · SciPy · Matplotlib · Plotly · Requests ·
BeautifulSoup · Scikit-learn
