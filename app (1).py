"""
app.py
--------------------------------------------------------------------------
Smart Home Analytics - Streamlit Dashboard
A college-level Python PBL project.

Run with:
    streamlit run app.py
--------------------------------------------------------------------------
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import smart_home_utils as shu   # custom module (Syllabus Topic #4)

# ===========================================================================
# PAGE CONFIG + STYLING
# ===========================================================================

st.set_page_config(
    page_title="Smart Home Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #0f172a; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .metric-label { color: #94a3b8; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase; }
    .metric-value { color: #f8fafc; font-size: 1.7rem; font-weight: 700; margin-top: 4px; }
    .section-title {
        font-size: 1.35rem; font-weight: 700; color: #f8fafc;
        border-left: 5px solid #38bdf8; padding-left: 12px; margin: 1.2rem 0 0.8rem 0;
    }
    .info-box {
        background-color: #1e293b; border-left: 4px solid #38bdf8;
        padding: 12px 16px; border-radius: 8px; color: #e2e8f0; font-size: 0.92rem;
    }
    div[data-testid="stSidebar"] { background-color: #0b1220; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_title(text: str):
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)


def kpi_card(label: str, value: str):
    st.markdown(
        f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def info_box(text: str):
    st.markdown(f"<div class='info-box'>{text}</div>", unsafe_allow_html=True)


# ===========================================================================
# SESSION STATE / DATA LOADING (File Handling + Exception Handling)
# ===========================================================================

if "df" not in st.session_state:
    try:
        st.session_state.df = shu.ensure_dataset_exists()
    except Exception as e:
        st.error(f"Fatal error initializing dataset: {e}")
        st.session_state.df = shu.generate_sample_data()

if "ml_result" not in st.session_state:
    st.session_state.ml_result = None


# ===========================================================================
# SIDEBAR NAVIGATION
# ===========================================================================

st.sidebar.markdown("## 🏠 Smart Home Analytics")
st.sidebar.caption("College PBL Project — Python + Streamlit")

PAGES = [
    "Dashboard", "Dataset / CRUD", "Data Preprocessing", "Statistical Analysis",
    "Probability & Sampling", "Hypothesis Testing", "Visualization",
    "Web Scraping", "Machine Learning", "About Project",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Upload your own CSV", type=["csv"])
if uploaded_file is not None:
    try:
        new_df = pd.read_csv(uploaded_file)
        required_cols = {"Device", "Room", "Temperature", "Humidity",
                          "Energy_kWh", "Usage_Hours", "Status", "Timestamp"}
        if not required_cols.issubset(set(new_df.columns)):
            st.sidebar.error(f"CSV missing required columns: {required_cols - set(new_df.columns)}")
        else:
            if "Device_ID" not in new_df.columns:
                new_df.insert(0, "Device_ID", range(1, len(new_df) + 1))
            st.session_state.df = new_df
            shu.save_data(new_df)
            st.sidebar.success(f"Loaded {len(new_df)} records from uploaded file.")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")

if st.sidebar.button("🔄 Reset to fresh sample data"):
    st.session_state.df = shu.generate_sample_data()
    shu.save_data(st.session_state.df)
    st.sidebar.success("Dataset regenerated.")

st.sidebar.markdown("---")
st.sidebar.caption("Syllabus topics are labeled throughout the app so every "
                    "concept is easy to point out during your viva.")

df = st.session_state.df

# ===========================================================================
# PAGE 1: DASHBOARD
# ===========================================================================

if page == "Dashboard":
    st.title("🏠 Smart Home Analytics Dashboard")
    st.caption("Real-time overview of household device energy usage")

    try:
        kpis = shu.compute_kpis(df)
    except Exception as e:
        st.error(f"Error computing KPIs: {e}")
        kpis = {"total_energy": 0, "avg_energy": 0, "estimated_cost": 0,
                "top_device": "N/A", "top_device_value": 0, "total_records": 0}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Energy", f"{kpis['total_energy']:.2f} kWh")
    with c2:
        kpi_card("Average Energy / Record", f"{kpis['avg_energy']:.3f} kWh")
    with c3:
        kpi_card("Estimated Cost", f"₹ {kpis['estimated_cost']:.2f}")
    with c4:
        kpi_card("Top Consuming Device", f"{kpis['top_device']}")

    st.write("")
    col_a, col_b = st.columns([1.3, 1])

    with col_a:
        section_title("Energy Consumption by Device")
        device_energy = df.groupby("Device")["Energy_kWh"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(device_energy, x="Device", y="Energy_kWh", color="Energy_kWh",
                     color_continuous_scale="Blues", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section_title("Device Status Split")
        status_counts = df["Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig2 = px.pie(status_counts, names="Status", values="Count", hole=0.45,
                      template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    section_title("Energy Consumption by Room")
    room_energy = df.groupby("Room")["Energy_kWh"].sum().sort_values(ascending=False).reset_index()
    fig3 = px.bar(room_energy, x="Room", y="Energy_kWh", color="Room", template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📄 View raw dataset"):
        st.dataframe(df, use_container_width=True)


# ===========================================================================
# PAGE 2: DATASET / CRUD
# ===========================================================================

elif page == "Dataset / CRUD":
    st.title("📋 Dataset & CRUD Operations")
    info_box("Syllabus: <b>File Handling (CSV)</b>, <b>CRUD Operations</b>, "
             "<b>Exception Handling</b>, <b>Data Structures</b>.")

    tabs = st.tabs(["🔍 Search / Filter", "➕ Create", "✏️ Update", "🗑️ Delete"])

    # ---- SEARCH / FILTER ----
    with tabs[0]:
        section_title("Search & Filter Records")
        c1, c2, c3, c4 = st.columns(4)
        device_filter = c1.selectbox("Device", ["All"] + sorted(df["Device"].unique().tolist()))
        room_filter = c2.selectbox("Room", ["All"] + sorted(df["Room"].unique().tolist()))
        status_filter = c3.selectbox("Status", ["All"] + sorted(df["Status"].unique().tolist()))
        keyword = c4.text_input("Keyword search")

        try:
            filtered = shu.search_records(df, device_filter, room_filter, status_filter, keyword)
            st.write(f"**{len(filtered)}** matching records")
            st.dataframe(filtered, use_container_width=True)
        except Exception as e:
            st.error(f"Search error: {e}")

    # ---- CREATE ----
    with tabs[1]:
        section_title("Add a New Device Record")
        with st.form("create_form"):
            c1, c2 = st.columns(2)
            device = c1.selectbox("Device", shu.DEVICE_TYPES)
            room = c2.selectbox("Room", shu.ROOMS)
            c3, c4 = st.columns(2)
            temperature = c3.number_input("Temperature (°C)", value=24.0)
            humidity = c4.number_input("Humidity (%)", value=50.0)
            c5, c6 = st.columns(2)
            energy = c5.number_input("Energy (kWh)", value=1.0, min_value=0.0)
            usage_hours = c6.number_input("Usage Hours", value=2.0, min_value=0.0)
            status = st.selectbox("Status", sorted(shu.DEVICE_STATUSES))
            timestamp = st.text_input("Timestamp", value=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
            submitted = st.form_submit_button("Add Record")

        if submitted:
            record = {"Device": device, "Room": room, "Temperature": temperature,
                      "Humidity": humidity, "Energy_kWh": energy, "Usage_Hours": usage_hours,
                      "Status": status, "Timestamp": timestamp}
            try:
                st.session_state.df = shu.create_record(df, record)
                shu.save_data(st.session_state.df)
                st.success("Record added and saved to CSV.")
            except shu.DataValidationError as e:
                st.error(f"Validation error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    # ---- UPDATE ----
    with tabs[2]:
        section_title("Update an Existing Record")
        if df.empty:
            st.warning("No records available.")
        else:
            device_id = st.selectbox("Select Device_ID", df["Device_ID"].tolist())
            row = df[df["Device_ID"] == device_id].iloc[0]
            with st.form("update_form"):
                c1, c2 = st.columns(2)
                new_energy = c1.number_input("Energy (kWh)", value=float(row["Energy_kWh"]))
                new_hours = c2.number_input("Usage Hours", value=float(row["Usage_Hours"]))
                new_status = st.selectbox("Status", sorted(shu.DEVICE_STATUSES),
                                           index=sorted(shu.DEVICE_STATUSES).index(row["Status"])
                                           if row["Status"] in shu.DEVICE_STATUSES else 0)
                update_submit = st.form_submit_button("Update Record")

            if update_submit:
                try:
                    st.session_state.df = shu.update_record(
                        df, device_id,
                        {"Energy_kWh": new_energy, "Usage_Hours": new_hours, "Status": new_status},
                    )
                    shu.save_data(st.session_state.df)
                    st.success(f"Device_ID {device_id} updated and saved.")
                except shu.RecordNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    # ---- DELETE ----
    with tabs[3]:
        section_title("Delete a Record")
        if df.empty:
            st.warning("No records available.")
        else:
            del_id = st.selectbox("Select Device_ID to delete", df["Device_ID"].tolist(), key="del_id")
            if st.button("Delete Record", type="primary"):
                try:
                    st.session_state.df = shu.delete_record(df, del_id)
                    shu.save_data(st.session_state.df)
                    st.success(f"Device_ID {del_id} deleted and CSV updated.")
                except shu.RecordNotFoundError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected error: {e}")


# ===========================================================================
# PAGE 3: DATA PREPROCESSING
# ===========================================================================

elif page == "Data Preprocessing":
    st.title("🧹 Data Preprocessing")
    info_box("Syllabus: <b>Data Preprocessing</b>, <b>Pandas</b>, <b>NumPy</b>.")

    section_title("Before Preprocessing")
    c1, c2 = st.columns(2)
    c1.metric("Missing values", int(df.isna().sum().sum()))
    c2.metric("Duplicate rows", int(df.duplicated().sum()))
    st.dataframe(df.describe(), use_container_width=True)

    try:
        cleaned = shu.preprocess_data(df)
        st.session_state.cleaned_df = cleaned

        section_title("After Preprocessing")
        c3, c4 = st.columns(2)
        c3.metric("Missing values", int(cleaned.isna().sum().sum()))
        c4.metric("Duplicate rows", int(cleaned.duplicated().sum()))
        st.write("Outliers capped with the **IQR method**, and two new engineered "
                 "features were added: `Efficiency_kWh_per_hr` and `Status_Code`.")
        st.dataframe(cleaned.head(15), use_container_width=True)

        buf = io.StringIO()
        cleaned.to_csv(buf, index=False)
        st.download_button("⬇️ Download cleaned dataset (CSV)", buf.getvalue(),
                            file_name="smart_home_cleaned.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Preprocessing error: {e}")


# ===========================================================================
# PAGE 4: STATISTICAL ANALYSIS
# ===========================================================================

elif page == "Statistical Analysis":
    st.title("📊 Statistical Analysis")
    info_box("Syllabus: <b>Statistical Analysis</b> (mean, median, mode, variance, std), "
             "<b>NumPy</b>, <b>SciPy</b>.")

    column = st.selectbox("Select numeric column", ["Energy_kWh", "Temperature", "Humidity", "Usage_Hours"])

    try:
        desc = shu.descriptive_statistics(df[column])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean", f"{desc['mean']:.3f}")
        c2.metric("Median", f"{desc['median']:.3f}")
        c3.metric("Mode", f"{desc['mode']}")
        c4.metric("Variance", f"{desc['variance']:.3f}")
        c5.metric("Std Dev", f"{desc['std_dev']:.3f}")
    except Exception as e:
        st.error(f"Statistics error: {e}")

    section_title("NumPy Analysis (Energy_kWh)")
    try:
        np_res = shu.numpy_analysis(df)
        st.json(np_res)
    except Exception as e:
        st.error(f"NumPy analysis error: {e}")

    section_title("SciPy Analysis (Energy_kWh)")
    try:
        sp_res = shu.scipy_analysis(df)
        st.json(sp_res)
    except Exception as e:
        st.error(f"SciPy analysis error: {e}")

    section_title("Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[column].dropna(), bins=25, color="#38bdf8", edgecolor="black")
    ax.set_title(f"Distribution of {column} (Matplotlib)")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


# ===========================================================================
# PAGE 5: PROBABILITY & SAMPLING
# ===========================================================================

elif page == "Probability & Sampling":
    st.title("🎲 Probability & Sampling")
    info_box("Syllabus: <b>Probability</b>, <b>Sampling & Statistical Inference</b>.")

    section_title("Probability Calculations")
    c1, c2 = st.columns(2)
    with c1:
        try:
            p_on = shu.probability_device_on(df)
            st.metric("P(Device Status = 'On')", f"{p_on:.3f}")
        except Exception as e:
            st.error(f"Error: {e}")
    with c2:
        threshold = st.number_input("Energy threshold (kWh)", value=float(df["Energy_kWh"].mean()))
        try:
            p_above = shu.probability_energy_above_threshold(df, threshold)
            st.metric(f"P(Energy > {threshold:.2f} kWh)", f"{p_above:.3f}")
        except Exception as e:
            st.error(f"Error: {e}")

    section_title("Normal Distribution (Probability Density)")
    mean_e, std_e = df["Energy_kWh"].mean(), df["Energy_kWh"].std()
    x_vals = np.linspace(df["Energy_kWh"].min(), df["Energy_kWh"].max(), 200)
    y_vals = shu.normal_probability_density(x_vals, mean_e, std_e)
    fig_pdf = go.Figure()
    fig_pdf.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", line=dict(color="#38bdf8")))
    fig_pdf.update_layout(template="plotly_dark", title="Normal PDF fitted to Energy_kWh",
                           xaxis_title="Energy (kWh)", yaxis_title="Probability Density")
    st.plotly_chart(fig_pdf, use_container_width=True)

    section_title("Random Sampling & 95% Confidence Interval")
    sample_size = st.slider("Sample size", min_value=5, max_value=min(200, len(df)), value=30)
    try:
        result = shu.random_sample_with_ci(df, sample_size=sample_size, confidence=0.95)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sample Mean", f"{result['sample_mean']:.3f} kWh")
        c2.metric("95% CI Lower", f"{result['ci_low']:.3f}")
        c3.metric("95% CI Upper", f"{result['ci_high']:.3f}")
        st.write("Interpretation: We are 95% confident the true population mean "
                 "energy consumption lies within this interval.")
        st.bar_chart(result["sample"].reset_index(drop=True))
    except Exception as e:
        st.error(f"Sampling error: {e}")


# ===========================================================================
# PAGE 6: HYPOTHESIS TESTING
# ===========================================================================

elif page == "Hypothesis Testing":
    st.title("🧪 Hypothesis Testing")
    info_box("Syllabus: <b>Hypothesis Testing</b> (independent two-sample t-test via SciPy).")

    devices = sorted(df["Device"].unique().tolist())
    c1, c2 = st.columns(2)
    device_a = c1.selectbox("Device A", devices, index=0)
    device_b = c2.selectbox("Device B", devices, index=min(1, len(devices) - 1))

    st.write("**H0:** The mean energy consumption of Device A equals that of Device B.")
    st.write("**H1:** The mean energy consumption of Device A differs from that of Device B.")

    if st.button("Run Hypothesis Test"):
        try:
            result = shu.hypothesis_test_two_devices(df, device_a, device_b)
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Mean ({device_a})", f"{result['mean_a']:.3f}")
            c2.metric(f"Mean ({device_b})", f"{result['mean_b']:.3f}")
            c3.metric("p-value", f"{result['p_value']:.4f}")
            st.metric("t-statistic", f"{result['t_statistic']:.4f}")

            if result["reject_null"]:
                st.success(result["conclusion"])
            else:
                st.info(result["conclusion"])
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"Hypothesis test error: {e}")


# ===========================================================================
# PAGE 7: VISUALIZATION
# ===========================================================================

elif page == "Visualization":
    st.title("📈 Interactive Data Visualization")
    info_box("Syllabus: <b>Data Visualization</b> using Matplotlib and Plotly.")

    section_title("Energy vs Usage Hours (Plotly Scatter)")
    fig1 = px.scatter(df, x="Usage_Hours", y="Energy_kWh", color="Device",
                       size="Energy_kWh", hover_data=["Room", "Status"],
                       template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    section_title("Temperature vs Humidity (Plotly)")
    fig2 = px.scatter(df, x="Temperature", y="Humidity", color="Room", template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    section_title("Average Energy per Device (Matplotlib)")
    avg_energy = df.groupby("Device")["Energy_kWh"].mean().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(9, 4.5))
    avg_energy.plot(kind="bar", ax=ax3, color="#0ea5e9")
    ax3.set_ylabel("Average Energy (kWh)")
    ax3.set_title("Average Energy Consumption by Device")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig3)

    section_title("Energy Trend Over Time (Plotly Line)")
    try:
        ts_df = df.copy()
        ts_df["Timestamp"] = pd.to_datetime(ts_df["Timestamp"], errors="coerce")
        trend = ts_df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
        trend = trend.set_index("Timestamp").resample("D")["Energy_kWh"].sum().reset_index()
        fig4 = px.line(trend, x="Timestamp", y="Energy_kWh", template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)
    except Exception as e:
        st.error(f"Trend chart error: {e}")

    section_title("Correlation Heatmap")
    numeric_df = df[["Temperature", "Humidity", "Energy_kWh", "Usage_Hours"]]
    corr = numeric_df.corr()
    fig5 = px.imshow(corr, text_auto=True, color_continuous_scale="Blues", template="plotly_dark")
    st.plotly_chart(fig5, use_container_width=True)


# ===========================================================================
# PAGE 8: WEB SCRAPING
# ===========================================================================

elif page == "Web Scraping":
    st.title("🌐 Web Scraping Demonstration")
    info_box("Syllabus: <b>Web Scraping</b> using Requests + BeautifulSoup.")

    st.write("This page scrapes a public webpage about electricity pricing to "
             "contextualize the dashboard's cost estimation. A safe fallback is "
             "shown automatically if there is no internet connection.")

    if st.button("🔎 Scrape Electricity Pricing Info"):
        with st.spinner("Fetching page with Requests + parsing with BeautifulSoup..."):
            try:
                data = shu.scrape_electricity_info()
                st.success(f"Status: {data['status']}")
                st.write(f"**Source URL:** {data['source_url']}")
                st.write(f"**Page Title:** {data['page_title']}")
                st.write("**Summary:**")
                st.write(data["summary"])
                if data["section_headings"]:
                    st.write("**Section Headings Found:**")
                    for h in data["section_headings"]:
                        st.write(f"- {h}")
            except Exception as e:
                st.error(f"Scraping failed unexpectedly: {e}")

    st.markdown("---")
    st.caption("Note: Web scraping requires an active internet connection. If the "
               "sandbox/machine running Streamlit is offline, fallback data is displayed "
               "automatically via exception handling — the app will not crash.")


# ===========================================================================
# PAGE 9: MACHINE LEARNING
# ===========================================================================

elif page == "Machine Learning":
    st.title("🤖 Machine Learning — Energy Prediction")
    info_box("Syllabus: <b>Machine Learning</b> (scikit-learn Linear Regression), "
             "train/test split, feature scaling, MAE, R² score.")

    section_title("Train the Model")
    test_size = st.slider("Test set proportion", 0.1, 0.4, 0.2, 0.05)

    if st.button("🚀 Train Linear Regression Model"):
        try:
            result = shu.train_energy_model(df, test_size=test_size)
            st.session_state.ml_result = result
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")

    if st.session_state.ml_result is not None:
        result = st.session_state.ml_result
        c1, c2 = st.columns(2)
        c1.metric("Mean Absolute Error (MAE)", f"{result['mae']:.4f}")
        c2.metric("R² Score", f"{result['r2']:.4f}")

        section_title("Model Coefficients")
        coef_df = pd.DataFrame({
            "Feature": list(result["coefficients"].keys()),
            "Coefficient": list(result["coefficients"].values()),
        })
        st.dataframe(coef_df, use_container_width=True)
        st.write(f"Intercept: `{result['intercept']:.4f}`")

        section_title("Predicted vs Actual (Test Set)")
        pred_df = pd.DataFrame({"Actual": result["y_test"], "Predicted": result["y_pred"]})
        fig = px.scatter(pred_df, x="Actual", y="Predicted", template="plotly_dark",
                          title="Predicted vs Actual Energy Consumption")
        fig.add_shape(type="line", x0=pred_df["Actual"].min(), y0=pred_df["Actual"].min(),
                      x1=pred_df["Actual"].max(), y1=pred_df["Actual"].max(),
                      line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        section_title("🔮 Predict Energy Consumption")
        c1, c2, c3 = st.columns(3)
        temp_in = c1.number_input("Temperature (°C)", value=25.0)
        hum_in = c2.number_input("Humidity (%)", value=50.0)
        hours_in = c3.number_input("Usage Hours", value=3.0, min_value=0.0)

        if st.button("Predict Energy Consumption"):
            try:
                prediction = shu.predict_energy(result["model"], result["scaler"],
                                                 temp_in, hum_in, hours_in)
                st.success(f"⚡ Predicted Energy Consumption: **{prediction:.3f} kWh**")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.info("Train the model above to unlock predictions.")


# ===========================================================================
# PAGE 10: ABOUT PROJECT
# ===========================================================================

elif page == "About Project":
    st.title("ℹ️ About This Project")
    st.markdown("""
### Smart Home Analytics
A college-level Python Project-Based Learning (PBL) application built with
**Streamlit**, analyzing smart-home device data (device, room, temperature,
humidity, energy consumption, usage hours, status, and timestamp).

#### Syllabus Topics Demonstrated
| # | Topic | Where in the App |
|---|-------|-------------------|
| 1 | Python Core Concepts | Throughout `smart_home_utils.py` |
| 2 | Data Structures (list, tuple, set, dict) | `ROOMS`, `DEVICE_TYPES`, `DEVICE_STATUSES`, `DEVICE_BASE_ENERGY` |
| 3 | Python Libraries | numpy, pandas, scipy, sklearn, requests, bs4, plotly, matplotlib |
| 4 | Custom Modules | `smart_home_utils.py` |
| 5 | Exception Handling | try/except + custom exceptions everywhere |
| 6 | Web Scraping | Web Scraping page (Requests + BeautifulSoup) |
| 7 | File Handling (CSV) | Auto dataset generation, upload, save/load |
| 8 | CRUD Operations | Dataset / CRUD page |
| 9 | NumPy | Statistical Analysis page |
| 10 | SciPy | Statistical Analysis, Hypothesis Testing pages |
| 11 | Pandas | Used throughout |
| 12 | Data Preprocessing | Data Preprocessing page |
| 13 | Data Visualization | Visualization page (Matplotlib + Plotly) |
| 14 | Statistical Analysis | Statistical Analysis page |
| 15 | Probability | Probability & Sampling page |
| 16 | Sampling & Inference | Probability & Sampling page |
| 17 | Hypothesis Testing | Hypothesis Testing page |
| 18 | Machine Learning | Machine Learning page (Linear Regression) |

#### Tech Stack
Python, Streamlit, Pandas, NumPy, SciPy, Matplotlib, Plotly, Requests,
BeautifulSoup, Scikit-learn.

#### Author
Add your name, roll number, class, and college here before submission.
    """)
