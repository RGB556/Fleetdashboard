import streamlit as st
import pandas as pd
from functools import reduce
from datetime import timedelta

st.set_page_config(page_title="Fleet Profit & Lane Performance Dashboard", layout="wide")
st.title("Fleet Profit & Lane Performance Dashboard")

# ---------- Helpers ----------
def load_any(file):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def thursday_week_start(d):
    # Week = Thu 00:00:00 through Wed 23:59:59
    # d.weekday(): Mon=0 ... Sun=6; Thu=3
    if pd.isna(d):
        return pd.NaT
    offset = (d.weekday() - 3) % 7  # days since last Thursday
    return d - timedelta(days=offset)

# ---------- Sidebar uploads ----------
st.sidebar.header("Upload Data Files")

driver_file       = st.sidebar.file_uploader("Driver Master (Drivers.xlsx)", type=["xlsx", "csv"])
fuel_file_tcs     = st.sidebar.file_uploader("TCS Fuel Data", type=["csv", "xlsx"])
fuel_file_irving  = st.sidebar.file_uploader("Irving Fuel Data", type=["csv", "xlsx"])
toll_file         = st.sidebar.file_uploader("EZPass Toll Data", type=["csv", "xlsx"])
loads_file        = st.sidebar.file_uploader("Trip Revenue / Lane Report", type=["csv", "xlsx"])
driver_miles_file = st.sidebar.file_uploader("Driver Mileage Report (Thu→Wed)", type=["xlsx", "csv"])

st.sidebar.markdown("---")
view = st.sidebar.radio(
    "Select View",
    [
        "Overview",
        "Lane Profit",
        "Driver Fuel & Tolls",
        "Weekly / Monthly",
        "Lane Toll Comparison",
        "Home Time & Miles",
    ]
)

# ---------- Load raw data ----------
driver_df       = load_any(driver_file)
fuel_df_tcs     = load_any(fuel_file_tcs)
fuel_df_irving  = load_any(fuel_file_irving)
tolls_df        = load_any(toll_file)
loads_df        = load_any(loads_file)
driver_miles_df = load_any(driver_miles_file)

# If ALL of the main dataframes are still None, stop and show message
if all(x is None for x in [fuel_df_tcs, fuel_df_irving, tolls_df, loads_df, driver_miles_df]):
    st.info("Upload at least some data on the left to begin.")
    st.stop()

# ======================================================
# 1) DRIVER MASTER MAP (cards + EZPass → driver/truck)
# ======================================================
driver_map_irving = None
driver_map_ezpass = None

if driver_df is not None:
    df = driver_df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Adjust these if your driver file header names change
    df = df.rename(columns={
        "Column1": "Driver",
        "Column3": "Truck",
        "Column42": "IrvingFuelCard",
        "Column4": "EZPASS"
    })

    # Drop header row stored as data
    df = df[df["Driver"] != "Driver Name"].copy()

    # Irving card mapping (last digits -> integer)
    df["IrvingFuelCard"] = df["IrvingFuelCard"].astype(str).str.extract(r"(\d+)", expand=False)
    df["IrvingCardInt"] = pd.to_numeric(df["IrvingFuelCard"], errors="coerce")
    driver_map_irving = df[["Driver", "Truck", "IrvingCardInt"]].dropna(subset=["IrvingCardInt"])

    # EZPass mapping
    df["EZPASS"] = df["EZPASS"].astype(str).str.strip()
    driver_map_ezpass = df[["Driver", "Truck", "EZPASS"]]

# =====================================
# 2) FUEL DATA (TCS + Irving combined)
# =====================================
fuel_frames = []

# --- TCS fuel ---
if fuel_df_tcs is not None:
    f = fuel_df_tcs.copy()
    f.columns = [c.strip() for c in f.columns]

    # Adjust these keys if TCS headers differ
    f = f.rename(columns={
        "TX Date": "date",
        "Driver Name": "driver",
        "Unit #": "truck",
        "Total": "fuel_cost"
    })

    for col in ["date", "driver", "truck", "fuel_cost"]:
        if col not in f.columns:
            f[col] = 0 if col == "fuel_cost" else None

    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f["fuel_cost"] = pd.to_numeric(f["fuel_cost"], errors="coerce").fillna(0)
    f["fuel_source"] = "TCS"

    fuel_frames.append(f[["date", "driver", "truck", "fuel_cost", "fuel_source"]])

# --- Irving fuel with card → driver mapping ---
if fuel_df_irving is not None:
    irv = fuel_df_irving.copy()
    irv.columns = [c.strip() for c in irv.columns]

    irv = irv.rename(columns={
        "Tran Date": "date",
        "Amt": "fuel_cost"
    })

    if driver_map_irving is not None and "Card #" in irv.columns:
        irv["Card #"] = pd.to_numeric(irv["Card #"], errors="coerce")
        irv = irv.merge(
            driver_map_irving,
            left_on="Card #",
            right_on="IrvingCardInt",
            how="left"
        )
        # From driver master
        irv["driver"] = irv["Driver"]
        irv["truck"] = irv["Truck"]
    else:
        # Fallback if we have driver/unit in file
        irv = irv.rename(columns={
            "Driver Name": "driver",
            "Unit": "truck"
        })

    for col in ["date", "driver", "truck", "fuel_cost"]:
        if col not in irv.columns:
            irv[col] = 0 if col == "fuel_cost" else None

    irv["date"] = pd.to_datetime(irv["date"], errors="coerce")
    irv["fuel_cost"] = pd.to_numeric(irv["fuel_cost"], errors="coerce").fillna(0)
    irv["fuel_source"] = "Irving"

    fuel_frames.append(irv[["date", "driver", "truck", "fuel_cost", "fuel_source"]])

fuel_df = None
if fuel_frames:
    fuel_df = pd.concat(fuel_frames, ignore_index=True)

# ==========================
# 3) TOLL DATA (EZPass)
# ==========================
tolls_agg = None
if tolls_df is not None:
    t = tolls_df.copy()
    t.columns = [c.strip() for c in t.columns]

    # Adjust these keys if your EZPass export differs
    if "Transaction Date" in t.columns:
        t = t.rename(columns={
            "Transaction Date": "date",
            "Tag/Vehicle Reg.": "tag",
            "Toll": "toll_cost"
        })
    else:
        # Try some generic alternatives
        if "date" not in t.columns:
            for c in t.columns:
                if "date" in c.lower():
                    t = t.rename(columns={c: "date"})
                    break
        if "Tag/Vehicle Reg." in t.columns:
            t = t.rename(columns={"Tag/Vehicle Reg.": "tag"})
        if "Toll" in t.columns:
            t = t.rename(columns={"Toll": "toll_cost"})

    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["toll_cost"] = (
        t["toll_cost"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )

    # Merge tag -> driver/truck using driver_map_ezpass
    if driver_map_ezpass is not None and "tag" in t.columns:
        t["tag"] = t["tag"].astype(str).str.strip()
        t = t.merge(
            driver_map_ezpass,
            left_on="tag",
            right_on="EZPASS",
            how="left"
        )
        t["driver"] = t["Driver"]
        t["truck"] = t["Truck"]
    else:
        if "driver" not in t.columns:
            t["driver"] = None
        if "truck" not in t.columns:
            t["truck"] = None

    tolls_agg = t.groupby(["date", "driver", "truck"], dropna=False)["toll_cost"].sum().reset_index()

# ==========================
# 4) LOAD / REVENUE DATA
# ==========================
loads_agg = None
if loads_df is not None:
    ld = loads_df.copy()
    ld.columns = [c.strip() for c in ld.columns]

    # Adjust these keys to your Trip Rev/Mile export
    ld = ld.rename(columns={
        "Driver": "driver",
        "Truck": "truck",
        "Trip #": "trip_id",
        "Actual DL": "date",  # or "Actual PU" if you prefer
        "Trip Total Alloc.Rev.(USD)": "revenue",
        "Total Miles": "miles",
        "Origin": "origin",
        "Destination": "destination"
    })

    for col in ["date", "driver", "truck", "revenue"]:
        if col not in ld.columns:
            ld[col] = 0 if col == "revenue" else None

    ld["date"] = pd.to_datetime(ld["date"], errors="coerce")
    ld["revenue"] = pd.to_numeric(ld["revenue"], errors="coerce").fillna(0)
    if "miles" in ld.columns:
        ld["miles"] = pd.to_numeric(ld["miles"], errors="coerce").fillna(0)

    # Lane: Origin → Destination
    if "origin" in ld.columns and "destination" in ld.columns:
        ld["lane"] = (
            ld["origin"].astype(str).str.strip()
            + " → "
            + ld["destination"].astype(str).str.strip()
        )
    else:
        ld["lane"] = "Unknown"

    loads_agg = ld

# ==========================
# 5) DRIVER MILEAGE REPORT
# ==========================
driver_miles_summary = None
if driver_miles_df is not None:
    dm = driver_miles_df.copy()
    dm.columns = [c.strip() for c in dm.columns]

    dm = dm.rename(columns={
        "Driver": "driver",
        "Mileage (mi)": "period_miles",
        "Days Worked": "days_worked"
    })

    dm["period_miles"] = pd.to_numeric(dm["period_miles"], errors="coerce").fillna(0)
    dm["days_worked"] = pd.to_numeric(dm["days_worked"], errors="coerce").fillna(0)

    # Pay = $0.54 per mile on ALL miles in that Thu→Wed period
    dm["period_driver_pay"] = dm["period_miles"] * 0.54
    dm["home_days_est"] = 7 - dm["days_worked"].clip(0, 7)

    driver_miles_summary = dm[["driver", "period_miles", "days_worked", "home_days_est", "period_driver_pay"]]
# ==========================
# 6) MERGE EVERYTHING (trip/date level)
# ==========================
frames = []

if loads_agg is not None:
    # Only use the columns that actually exist in the loads file
    base_cols = ["date", "driver", "truck", "revenue", "miles", "lane"]
    existing_cols = [c for c in base_cols if c in loads_agg.columns]
    if existing_cols:
        frames.append(loads_agg[existing_cols])

if fuel_df is not None:
    fuel_agg = fuel_df.groupby(["date", "driver", "truck"], dropna=False)["fuel_cost"].sum().reset_index()
    frames.append(fuel_agg)

if tolls_agg is not None:
    frames.append(tolls_agg)

if not frames:
    st.error("No usable data after processing. Check your uploads and column mappings.")
    st.stop()

def merge_dfs(dfs):
    return reduce(
        lambda left, right: pd.merge(left, right, on=["date", "driver", "truck"], how="outer"),
        dfs
    )

combined = merge_dfs(frames)

for col in ["revenue", "fuel_cost", "toll_cost", "miles"]:
    if col in combined.columns:
        combined[col] = combined[col].fillna(0)

if "lane" not in combined.columns:
    combined["lane"] = "Unknown"

# Driver pay: use load miles as proxy for allocating pay to lanes
if "miles" in combined.columns:
    combined["driver_pay"] = combined.get("driver_pay", combined["miles"] * 0.54)
else:
    combined["driver_pay"] = 0

# Profit BEFORE driver pay (gross lane profit)
combined["profit_before_pay"] = (
    combined.get("revenue", 0)
    - combined.get("fuel_cost", 0)
    - combined.get("toll_cost", 0)
)

# Profit AFTER driver pay (true net)
combined["profit"] = combined["profit_before_pay"] - combined.get("driver_pay", 0)

combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
combined["week"] = combined["date"].apply(thursday_week_start)
combined["month"] = combined["date"] combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
combined["week"] = combined["date"].apply(thursday_week_start)
combined["month"] = combined["date"].dt.to_period("M").dt.to_timestamp()
(lambda r: r.start_time)

# ==========================
# 7) FILTERS
# ==========================
min_date = combined["date"].min()
max_date = combined["date"].max()

with st.sidebar:
    st.markdown("### Filters")
    date_range = st.date_input("Date range", [min_date, max_date])

    drivers = sorted([d for d in combined["driver"].dropna().unique()])
    trucks  = sorted([t for t in combined["truck"].dropna().unique()])
    lanes   = sorted([l for l in combined["lane"].dropna().unique()])

    driver_filter = st.multiselect("Driver(s)", drivers, default=drivers)
    truck_filter  = st.multiselect("Truck(s)", trucks, default=trucks)
    lane_filter   = st.multiselect("Lane(s)", lanes, default=lanes)

mask = (
    (combined["date"] >= pd.to_datetime(date_range[0])) &
    (combined["date"] <= pd.to_datetime(date_range[-1])) &
    (combined["driver"].isin(driver_filter)) &
    (combined["truck"].isin(truck_filter)) &
    (combined["lane"].isin(lane_filter))
)

f = combined[mask].copy()

# ==========================
# 8) TOP SUMMARY KPIs
# ==========================
total_rev   = f["revenue"].sum() if "revenue" in f.columns else 0
total_fuel  = f["fuel_cost"].sum() if "fuel_cost" in f.columns else 0
total_tolls = f["toll_cost"].sum() if "toll_cost" in f.columns else 0
total_pay   = f["driver_pay"].sum() if "driver_pay" in f.columns else 0
total_prof  = f["profit"].sum()
total_miles = f["miles"].sum() if "miles" in f.columns else 0

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Revenue",      f"${total_rev:,.2f}")
col2.metric("Fuel Cost",    f"${total_fuel:,.2f}")
col3.metric("Tolls",        f"${total_tolls:,.2f}")
col4.metric("Driver Pay",   f"${total_pay:,.2f}")
col5.metric("Profit (net)", f"${total_prof:,.2f}")
col6.metric("Miles",        f"{total_miles:,.0f}")

st.markdown("---")
# ==========================
# 9) VIEWS
# ==========================

# ---- Overview ----
if view == "Overview":
    st.subheader("Driver Profitability")

    driver_summary = f.groupby("driver", dropna=False)[
        ["revenue", "fuel_cost", "toll_cost", "driver_pay", "profit_before_pay", "profit", "miles"]
    ].sum()

    if "miles" in driver_summary.columns:
        driver_summary["profit_per_mile_net"] = (
            driver_summary["profit"] / driver_summary["miles"].replace(0, pd.NA)
        )
        driver_summary["profit_per_mile_before_pay"] = (
            driver_summary["profit_before_pay"] / driver_summary["miles"].replace(0, pd.NA)
        )

    st.dataframe(
        driver_summary.sort_values("profit", ascending=False)
        .style.format("${:,.2f}")
    )

    st.subheader("Truck Profitability")
    truck_summary = f.groupby("truck", dropna=False)[
        ["revenue", "fuel_cost", "toll_cost", "driver_pay", "profit_before_pay", "profit", "miles"]
    ].sum()

    if "miles" in truck_summary.columns:
        truck_summary["profit_per_mile_net"] = (
            truck_summary["profit"] / truck_summary["miles"].replace(0, pd.NA)
        )
        truck_summary["profit_per_mile_before_pay"] = (
            truck_summary["profit_before_pay"] / truck_summary["miles"].replace(0, pd.NA)
        )

    st.dataframe(
        truck_summary.sort_values("profit", ascending=False)
        .style.format("${:,.2f}")
    )

# ---- Lane Profit ----
elif view == "Lane Profit":
    st.subheader("Lane Profitability (Origin → Destination)")

    profit_mode = st.radio(
        "Profit metric",
        ["After driver pay (net)", "Before driver pay (no labor)"],
        horizontal=True
    )

    lane_summary = f.groupby("lane", dropna=False)[
        ["revenue", "fuel_cost", "toll_cost", "driver_pay", "profit_before_pay", "profit", "miles"]
    ].sum()

    if "miles" in lane_summary.columns:
        lane_summary["rev_per_mile"] = (
            lane_summary["revenue"] / lane_summary["miles"].replace(0, pd.NA)
        )
        lane_summary["profit_per_mile_net"] = (
            lane_summary["profit"] / lane_summary["miles"].replace(0, pd.NA)
        )
        lane_summary["profit_per_mile_before_pay"] = (
            lane_summary["profit_before_pay"] / lane_summary["miles"].replace(0, pd.NA)
        )

    if profit_mode.startswith("After"):
        lane_summary["profit_used"] = lane_summary["profit"]
        lane_summary["profit_per_mile_used"] = lane_summary["profit_per_mile_net"]
    else:
        lane_summary["profit_used"] = lane_summary["profit_before_pay"]
        lane_summary["profit_per_mile_used"] = lane_summary["profit_per_mile_before_pay"]

    lane_summary = lane_summary.sort_values("profit_used", ascending=False)

    st.markdown("**Lane table (sorted by selected profit metric)**")
    st.dataframe(
        lane_summary[
            [
                "revenue",
                "fuel_cost",
                "toll_cost",
                "driver_pay",
                "profit_before_pay",
                "profit",
                "miles",
                "rev_per_mile",
                "profit_per_mile_used",
            ]
        ]
        .rename(columns={"profit_per_mile_used": "profit_per_mile_selected"})
        .style.format("${:,.2f}")
    )

    st.markdown("**Profit by lane (selected metric)**")
    st.bar_chart(lane_summary["profit_used"])

# ---- Driver Fuel & Tolls ----
elif view == "Driver Fuel & Tolls":
    st.subheader("Fuel & Toll Usage by Driver")

    driver_profit_mode = st.radio(
        "Profit metric",
        ["After driver pay (net)", "Before driver pay (no labor)"],
        horizontal=True
    )

    dsum = f.groupby("driver", dropna=False)[
        ["revenue", "fuel_cost", "toll_cost", "driver_pay", "profit_before_pay", "profit", "miles"]
    ].sum()

    if "miles" in dsum.columns:
        dsum["fuel_per_mile"] = dsum["fuel_cost"] / dsum["miles"].replace(0, pd.NA)
        dsum["toll_per_mile"] = dsum["toll_cost"] / dsum["miles"].replace(0, pd.NA)
        dsum["rev_per_mile"]  = dsum["revenue"] / dsum["miles"].replace(0, pd.NA)

    if driver_profit_mode.startswith("After"):
        dsum["profit_used"] = dsum["profit"]
        dsum["profit_per_mile_used"] = (
            dsum["profit"] / dsum["miles"].replace(0, pd.NA)
            if "miles" in dsum.columns else dsum["profit"]
        )
    else:
        dsum["profit_used"] = dsum["profit_before_pay"]
        dsum["profit_per_mile_used"] = (
            dsum["profit_before_pay"] / dsum["miles"].replace(0, pd.NA)
            if "miles" in dsum.columns else dsum["profit_before_pay"]
        )

    # Merge pay-period miles/pay if present
    if driver_miles_summary is not None:
        dsum = dsum.merge(driver_miles_summary, on="driver", how="left")
        dsum["mile_gap"] = dsum["period_miles"] - dsum["miles"]

    st.markdown("**Driver summary (sorted by selected profit metric)**")
    st.dataframe(
        dsum.sort_values("profit_used", ascending=False)
            .style.format({
                "revenue": "${:,.2f}",
                "fuel_cost": "${:,.2f}",
                "toll_cost": "${:,.2f}",
                "driver_pay": "${:,.2f}",
                "profit_before_pay": "${:,.2f}",
                "profit": "${:,.2f}",
                "profit_used": "${:,.2f}",
                "fuel_per_mile": "${:,.4f}",
                "toll_per_mile": "${:,.4f}",
                "rev_per_mile": "${:,.4f}",
                "profit_per_mile_used": "${:,.4f}",
                "period_miles": "{:,.0f}",
                "period_driver_pay": "${:,.2f}",
                "mile_gap": "{:,.0f}",
            })
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("Fuel Cost by Driver")
        st.bar_chart(dsum["fuel_cost"])
    with col_b:
        st.subheader("Toll Cost by Driver")
        st.bar_chart(dsum["toll_cost"])
    with col_c:
        st.subheader("Selected Profit by Driver")
        st.bar_chart(dsum["profit_used"])

# ---- Weekly / Monthly ----
elif view == "Weekly / Monthly":
    period_type = st.radio("Group by", ["Weekly (Thu→Wed)", "Monthly"], horizontal=True)

    if period_type.startswith("Weekly"):
        grp_col = "week"
        label = "Week (Thu start)"
    else:
        grp_col = "month"
        label = "Month"

    st.subheader(f"{label} Summary")

    per = f.groupby(grp_col)[
        ["revenue", "fuel_cost", "toll_cost", "driver_pay", "profit_before_pay", "profit", "miles"]
    ].sum()

    if "miles" in per.columns:
        per["rev_per_mile"] = per["revenue"] / per["miles"].replace(0, pd.NA)
        per["profit_per_mile"] = per["profit"] / per["miles"].replace(0, pd.NA)

    st.dataframe(per.style.format("${:,.2f}"))
    st.line_chart(per["profit"])

# ---- Lane Toll Comparison ----
elif view == "Lane Toll Comparison":
    st.subheader("Toll Spend by Driver on the Same Lane")

    lane_driver = f.groupby(["lane", "driver"], dropna=False)[
        ["toll_cost", "miles", "revenue", "profit"]
    ].sum()

    if "miles" in lane_driver.columns:
        lane_driver["toll_per_mile"] = lane_driver["toll_cost"] / lane_driver["miles"].replace(0, pd.NA)

    st.markdown("#### Lane × Driver Toll Comparison")
    st.dataframe(
        lane_driver.sort_values("toll_cost", ascending=False)
            .style.format("${:,.2f}")
    )

    st.markdown("#### Pivot: Toll per mile by lane & driver")
    pivot = lane_driver.reset_index().pivot_table(
        index="lane",
        columns="driver",
        values="toll_per_mile",
        aggfunc="mean"
    )
    st.dataframe(pivot)

# ---- Home Time & Miles ----
elif view == "Home Time & Miles":
    st.subheader("Home Time & Period Miles (Thu→Wed)")

    if driver_miles_summary is None:
        st.info("Upload a Driver Mileage Report (Thu→Wed) to see this view.")
    else:
        home_df = driver_miles_summary.copy()
        st.dataframe(
            home_df[["driver", "days_worked", "home_days_est", "period_miles", "period_driver_pay"]]
            .sort_values("home_days_est", ascending=False)  # most home days at the top
            .style.format({
                "period_miles": "{:,.0f}",
                "period_driver_pay": "${:,.2f}",
            })
        )
