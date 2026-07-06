"""
RMS Alarm History Comparator
=============================

A Streamlit application that compares "Room Temperature" alarm events against
"Mains Fail" alarm events for the same Site (or Site Alias), based on
overlapping time intervals rather than exact timestamp matches.

Author: Generated for RMS Alarm Analysis
"""

import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Rms Station",
    "Site",
    "Site Alias",
    "Zone",
    "Cluster",
    "Tenant",
    "Tag",
    "Start Time",
    "End Time",
    "Elapsed Time",
    "Duration (Hr.)",
    "Acknowledged Time",
    "Acknowledged By",
    "Alarm KPI Group",
]

# Define column name variations (with and without spaces)
COLUMN_VARIATIONS = {
    "Rms Station": ["RmsStation", "RMS Station", "RMSStation"],
    "Site": ["Site"],
    "Site Alias": ["SiteAlias", "Site Alias", "Site_Alias"],
    "Zone": ["Zone"],
    "Cluster": ["Cluster"],
    "Tenant": ["Tenant"],
    "Tag": ["Tag"],
    "Start Time": ["StartTime", "Start Time", "Start_Time"],
    "End Time": ["EndTime", "End Time", "End_Time"],
    "Elapsed Time": ["ElapsedTime", "Elapsed Time", "Elapsed_Time"],
    "Duration (Hr.)": ["Duration(Hr.)", "Duration (Hr.)", "Duration_Hr"],
    "Acknowledged Time": ["AcknowledgedTime", "Acknowledged Time", "Acknowledged_Time"],
    "Acknowledged By": ["AcknowledgedBy", "Acknowledged By", "Acknowledged_By"],
    "Alarm KPI Group": ["AlarmKPI Group", "Alarm KPI Group", "KPIGroup", "KPI Group"],
}

DATETIME_COLUMNS = ["Start Time", "End Time", "Acknowledged Time"]

st.set_page_config(
    page_title="RMS Alarm Comparator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Data loading with robust column name handling
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_excel(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Load an Excel file (as raw bytes, for cache-friendliness) into a DataFrame.
    
    Handles:
    - Column names starting from row 0 or row 1
    - Extra spaces in column names
    - Case-insensitive column matching
    """
    buffer = io.BytesIO(file_bytes)
    is_legacy_xls = file_name.lower().endswith(".xls") and not file_name.lower().endswith(".xlsx")
    engine = "xlrd" if is_legacy_xls else "openpyxl"

    try:
        # First, try reading with header in row 0 (default)
        df = pd.read_excel(buffer, engine=engine, header=0)
        
        # Check if first row looks like data (not column names)
        # If the first row contains numeric values or dates, it might be data
        first_row = df.iloc[0] if len(df) > 0 else pd.Series()
        
        # If the first row contains mostly non-string values, the header might be in row 1
        string_count = sum(isinstance(val, str) for val in first_row.values if pd.notna(val))
        total_non_empty = sum(pd.notna(val) for val in first_row.values)
        
        if total_non_empty > 0 and string_count / total_non_empty < 0.3:
            # First row looks like data, try reading with header in row 1
            buffer.seek(0)
            df = pd.read_excel(buffer, engine=engine, header=1)
            
    except Exception as exc:
        st.error(
            f"❌ Error reading Excel file: {exc}\n\n"
            f"Please ensure the file is a valid Excel file with column names in the first row."
        )
        st.stop()

    # Clean column names: strip whitespace, normalize spaces
    df.columns = [
        str(c).strip().replace("  ", " ").replace("\n", " ")
        for c in df.columns
    ]
    
    return df


# --------------------------------------------------------------------------
# Column name normalization helper (improved)
# --------------------------------------------------------------------------

def normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Map actual column names to the standard required column names.
    Handles variations with and without spaces, case differences, etc.
    
    Returns:
    - df: DataFrame with columns renamed to match required names (where found)
    - mapping: Dict mapping required_name -> actual_name found in the file
    """
    # Create reverse mapping: variation -> standard name
    variation_to_standard = {}
    for standard, variations in COLUMN_VARIATIONS.items():
        for var in variations:
            variation_to_standard[var.lower()] = standard
    
    # Map actual columns to standard names
    mapping = {}
    for col in df.columns:
        col_clean = col.lower().strip()
        if col_clean in variation_to_standard:
            standard_name = variation_to_standard[col_clean]
            mapping[standard_name] = col
    
    # Rename columns to standard names
    df_renamed = df.rename(columns={v: k for k, v in mapping.items()})
    
    return df_renamed, mapping


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    """Return the list of required columns that are missing from df."""
    present = set(df.columns)
    return [c for c in required_columns if c not in present]


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def to_hours(series: pd.Series) -> pd.Series:
    """Best-effort conversion of a 'Duration' column to numeric hours.

    Handles:
      - already-numeric hour values
      - timedelta strings like 'HH:MM:SS' or 'D days HH:MM:SS'
      - pandas Timedelta objects
    Unparseable values become NaN.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        # Overwhelmingly numeric already -> treat as hours directly
        return numeric

    # Fall back to timedelta parsing
    td = pd.to_timedelta(series.astype(str), errors="coerce")
    hours = td.dt.total_seconds() / 3600.0
    # Keep any values that were numeric but timedelta parse failed
    hours = hours.fillna(numeric)
    return hours


def preprocess_dataframe(df: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Convert datetime and duration columns; return cleaned df plus a report
    of how many rows failed to parse in each column (for user warnings).
    """
    df = df.copy()
    parse_issues: Dict[str, int] = {}

    for col in DATETIME_COLUMNS:
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = pd.to_datetime(df[col], errors="coerce")
            after_na = df[col].isna().sum()
            parse_issues[col] = int(after_na - before_na)

    if "Duration (Hr.)" in df.columns:
        before_na = df["Duration (Hr.)"].isna().sum()
        df["Duration (Hr.)"] = to_hours(df["Duration (Hr.)"])
        after_na = df["Duration (Hr.)"].isna().sum()
        parse_issues["Duration (Hr.)"] = int(after_na - before_na)

    # Drop rows where Start Time or End Time could not be parsed at all,
    # since they cannot participate in overlap comparison.
    if "Start Time" in df.columns and "End Time" in df.columns:
        invalid_mask = df["Start Time"].isna() | df["End Time"].isna()
        parse_issues[f"{label}_dropped_rows"] = int(invalid_mask.sum())
        df = df.loc[~invalid_mask].reset_index(drop=True)

    return df, parse_issues


# --------------------------------------------------------------------------
# Overlap comparison (vectorized per-site, no nested O(n*m) loops)
# --------------------------------------------------------------------------

def build_mf_lookup(mf_df: pd.DataFrame, site_key: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Pre-group Mains Fail events by site key and build a per-site IntervalIndex
    so overlap queries per Room Temperature event are cheap and vectorized
    (no python-level nested loop over the full cross product of both files).
    """
    lookup: Dict[str, Dict[str, np.ndarray]] = {}

    for site_value, grp in mf_df.groupby(site_key, dropna=False):
        grp = grp.loc[grp["Start Time"] <= grp["End Time"]].reset_index(drop=True)
        if grp.empty:
            continue

        starts = grp["Start Time"].values
        ends = grp["End Time"].values
        interval_index = pd.IntervalIndex.from_arrays(starts, ends, closed="both")

        lookup[site_value] = {
            "interval_index": interval_index,
            "starts": starts,
            "ends": ends,
            "durations": grp["Duration (Hr.)"].values if "Duration (Hr.)" in grp.columns else np.full(len(grp), np.nan),
        }

    return lookup


def _merge_intervals(intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge overlapping (start, end) intervals to avoid double-counting
    overlap duration when several Mains Fail events overlap one another.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def compare_alarms(
    rt_df: pd.DataFrame,
    mf_df: pd.DataFrame,
    site_key: str = "Site",
    progress_callback=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Core comparison engine.

    For each Room Temperature (RT) event, find all Mains Fail (MF) events for
    the same site whose interval overlaps the RT event's interval.

    Returns
    -------
    summary_df : one row per RT event with aggregate overlap info
    detail_df  : one row per (RT event, overlapping MF event) pair
    """
    mf_lookup = build_mf_lookup(mf_df, site_key)

    summary_rows = []
    detail_rows = []

    total = len(rt_df)
    step = max(total // 100, 1)

    rt_records = rt_df.to_dict("records")

    for i, rt in enumerate(rt_records):
        site_value = rt.get(site_key)
        rt_start = rt["Start Time"]
        rt_end = rt["End Time"]

        base_info = {
            "Rms Station": rt.get("Rms Station"),
            "Site": rt.get("Site"),
            "Site Alias": rt.get("Site Alias"),
            "Zone": rt.get("Zone"),
            "Cluster": rt.get("Cluster"),
            "Tenant": rt.get("Tenant"),
            "Room Temperature Start Time": rt_start,
            "Room Temperature End Time": rt_end,
            "Room Temperature Duration (Hr.)": rt.get("Duration (Hr.)"),
        }

        mf_group = mf_lookup.get(site_value)
        overlap_intervals = []
        count = 0

        if mf_group is not None and rt_start <= rt_end:
            query_interval = pd.Interval(rt_start, rt_end, closed="both")
            overlap_mask = mf_group["interval_index"].overlaps(query_interval)

            if overlap_mask.any():
                idxs = np.nonzero(overlap_mask)[0]
                count = len(idxs)

                for idx in idxs:
                    mf_start = pd.Timestamp(mf_group["starts"][idx])
                    mf_end = pd.Timestamp(mf_group["ends"][idx])
                    mf_duration = mf_group["durations"][idx]

                    clipped_start = max(rt_start, mf_start)
                    clipped_end = min(rt_end, mf_end)
                    overlap_intervals.append((clipped_start, clipped_end))

                    detail_rows.append(
                        {
                            **base_info,
                            "Mains Fail Start Time": mf_start,
                            "Mains Fail End Time": mf_end,
                            "Mains Fail Duration (Hr.)": mf_duration,
                            "Overlap Start Time": clipped_start,
                            "Overlap End Time": clipped_end,
                            "Overlap Duration (Hr.)": (clipped_end - clipped_start).total_seconds() / 3600.0,
                        }
                    )

        merged = _merge_intervals(overlap_intervals)
        total_overlap_hours = sum((e - s).total_seconds() for s, e in merged) / 3600.0

        rt_duration_hours = rt.get("Duration (Hr.)")
        if rt_duration_hours is None or (isinstance(rt_duration_hours, float) and np.isnan(rt_duration_hours)):
            rt_duration_hours = (rt_end - rt_start).total_seconds() / 3600.0

        pct_covered = (
            (total_overlap_hours / rt_duration_hours * 100.0)
            if rt_duration_hours and rt_duration_hours > 0
            else np.nan
        )
        pct_covered = min(pct_covered, 100.0) if pd.notna(pct_covered) else pct_covered

        summary_rows.append(
            {
                **base_info,
                "Mains Fail Occurred": "Yes" if count > 0 else "No",
                "Mains Fail Event Count": count,
                "Total Overlapping Mains Fail Duration (Hr.)": round(total_overlap_hours, 4),
                "Percentage of RT Duration Covered (%)": round(pct_covered, 2) if pd.notna(pct_covered) else np.nan,
            }
        )

        if progress_callback and (i % step == 0 or i == total - 1):
            progress_callback((i + 1) / total)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    return summary_df, detail_df


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    zones: List[str],
    clusters: List[str],
    tenants: List[str],
    sites: List[str],
    search_text: str,
) -> pd.DataFrame:
    filtered = df.copy()

    if zones:
        filtered = filtered[filtered["Zone"].isin(zones)]
    if clusters:
        filtered = filtered[filtered["Cluster"].isin(clusters)]
    if tenants:
        filtered = filtered[filtered["Tenant"].isin(tenants)]
    if sites:
        filtered = filtered[filtered["Site"].isin(sites)]
    if search_text:
        text = search_text.strip().lower()
        mask = (
            filtered["Site"].astype(str).str.lower().str.contains(text, na=False)
            | filtered["Site Alias"].astype(str).str.lower().str.contains(text, na=False)
        )
        filtered = filtered[mask]

    return filtered


# --------------------------------------------------------------------------
# Excel export
# --------------------------------------------------------------------------

def to_excel_bytes(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> bytes:
    """Export the summary and detail tables to a multi-sheet Excel workbook,
    with light formatting for readability.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_export = summary_df.copy()
        for col in ["Room Temperature Start Time", "Room Temperature End Time"]:
            if col in summary_export.columns:
                parsed = pd.to_datetime(summary_export[col])
                if getattr(parsed.dt, "tz", None) is not None:
                    parsed = parsed.dt.tz_localize(None)
                summary_export[col] = parsed
        summary_export.to_excel(writer, sheet_name="Summary", index=False)

        if not detail_df.empty:
            detail_export = detail_df.copy()
            detail_export.to_excel(writer, sheet_name="Overlap Details", index=False)
        else:
            pd.DataFrame(
                columns=[
                    "Rms Station", "Site", "Site Alias", "Zone", "Cluster", "Tenant",
                    "Room Temperature Start Time", "Room Temperature End Time",
                    "Mains Fail Start Time", "Mains Fail End Time", "Mains Fail Duration (Hr.)",
                ]
            ).to_excel(writer, sheet_name="Overlap Details", index=False)

        # Auto-fit column widths (approximate)
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            df_for_width = summary_export if sheet_name == "Summary" else detail_df
            for i, col in enumerate(df_for_width.columns, start=1):
                max_len = max(
                    df_for_width[col].astype(str).map(len).max() if not df_for_width.empty else 0,
                    len(str(col)),
                ) + 2
                worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = min(max_len, 40)

    return output.getvalue()


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------

def render_upload_section() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 File 1 — Room Temperature Alarms")
        rt_file = st.file_uploader("Upload Room Temperature Excel file", type=["xlsx", "xls"], key="rt_file")

    with col2:
        st.subheader("📁 File 2 — Mains Fail Alarms")
        mf_file = st.file_uploader("Upload Mains Fail Excel file", type=["xlsx", "xls"], key="mf_file")

    rt_df, mf_df = None, None

    if rt_file is not None:
        rt_df = load_excel(rt_file.getvalue(), rt_file.name)

    if mf_file is not None:
        mf_df = load_excel(mf_file.getvalue(), mf_file.name)

    return rt_df, mf_df


def render_validation(rt_df: pd.DataFrame, mf_df: pd.DataFrame) -> bool:
    ok = True
    
    # Normalize column names
    rt_df_norm, rt_mapping = normalize_column_names(rt_df)
    mf_df_norm, mf_mapping = normalize_column_names(mf_df)
    
    # Check for missing columns after normalization
    missing_rt = validate_columns(rt_df_norm, REQUIRED_COLUMNS)
    missing_mf = validate_columns(mf_df_norm, REQUIRED_COLUMNS)

    if missing_rt:
        # Show helpful message with detected columns
        detected_cols = list(rt_df.columns)
        st.error(f"❌ Room Temperature file is missing required columns: {', '.join(missing_rt)}")
        st.info(f"📋 Detected columns in RT file: {', '.join(detected_cols)}")
        ok = False
    else:
        st.success(f"✅ Room Temperature file contains all required columns")
        # Show mapping
        mapping_str = ", ".join([f"{k} ← {v}" for k, v in rt_mapping.items()])
        st.caption(f"📋 Column mapping: {mapping_str}")
    
    if missing_mf:
        detected_cols = list(mf_df.columns)
        st.error(f"❌ Mains Fail file is missing required columns: {', '.join(missing_mf)}")
        st.info(f"📋 Detected columns in MF file: {', '.join(detected_cols)}")
        ok = False
    else:
        st.success(f"✅ Mains Fail file contains all required columns")
        mapping_str = ", ".join([f"{k} ← {v}" for k, v in mf_mapping.items()])
        st.caption(f"📋 Column mapping: {mapping_str}")

    # Return normalized dataframes via session state for later use
    if ok:
        st.session_state['rt_df_norm'] = rt_df_norm
        st.session_state['mf_df_norm'] = mf_df_norm
    
    return ok


def render_dashboard(summary_df: pd.DataFrame):
    total_events = len(summary_df)
    with_mf = int((summary_df["Mains Fail Occurred"] == "Yes").sum())
    without_mf = total_events - with_mf
    pct_with_mf = (with_mf / total_events * 100.0) if total_events > 0 else 0.0

    st.subheader("📊 Summary Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Room Temperature Events", f"{total_events:,}")
    c2.metric("Events with Mains Fail", f"{with_mf:,}")
    c3.metric("Events without Mains Fail", f"{without_mf:,}")
    c4.metric("% with Mains Fail", f"{pct_with_mf:.1f}%")


def render_filters(summary_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🔍 Filters")
    f1, f2, f3, f4 = st.columns(4)

    with f1:
        zones = st.multiselect("Zone", sorted(summary_df["Zone"].dropna().unique().tolist()))
    with f2:
        clusters = st.multiselect("Cluster", sorted(summary_df["Cluster"].dropna().unique().tolist()))
    with f3:
        tenants = st.multiselect("Tenant", sorted(summary_df["Tenant"].dropna().unique().tolist()))
    with f4:
        sites = st.multiselect("Site", sorted(summary_df["Site"].dropna().unique().tolist()))

    search_text = st.text_input("Search Site / Site Alias", placeholder="Type to search...")

    return apply_filters(summary_df, zones, clusters, tenants, sites, search_text)


def main():
    st.title("🛰️ RMS Alarm History Comparator")
    st.caption(
        "Compare Room Temperature alarm events against Mains Fail alarm events "
        "using overlapping time-interval logic."
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        site_key_choice = st.radio(
            "Match events by:",
            options=["Site", "Site Alias"],
            index=0,
            help="Choose whether to match Room Temperature and Mains Fail events using the Site or the Site Alias column.",
        )
        st.markdown("---")
        st.markdown(
            "**How it works**\n\n"
            "Two events overlap when:\n\n"
            "`MF Start Time <= RT End Time`\n\n"
            "`AND`\n\n"
            "`MF End Time >= RT Start Time`"
        )

    rt_df_raw, mf_df_raw = render_upload_section()

    if rt_df_raw is None or mf_df_raw is None:
        st.info("⬆️ Please upload both the Room Temperature and Mains Fail Excel files to begin.")
        return

    st.markdown("---")
    st.subheader("✅ Validation")
    if not render_validation(rt_df_raw, mf_df_raw):
        st.stop()
        return

    # Use normalized dataframes from session state
    rt_df = st.session_state['rt_df_norm']
    mf_df = st.session_state['mf_df_norm']

    with st.spinner("Converting date/time columns..."):
        rt_df, rt_issues = preprocess_dataframe(rt_df, "RT")
        mf_df, mf_issues = preprocess_dataframe(mf_df, "MF")

    dropped_rt = rt_issues.get("RT_dropped_rows", 0)
    dropped_mf = mf_issues.get("MF_dropped_rows", 0)
    if dropped_rt or dropped_mf:
        st.warning(
            f"⚠️ Dropped {dropped_rt:,} Room Temperature row(s) and {dropped_mf:,} Mains Fail row(s) "
            "due to unparseable Start/End Time values."
        )

    if rt_df.empty:
        st.error("No valid Room Temperature rows remain after cleaning. Please check the file.")
        return

    st.markdown("---")
    st.subheader("⏳ Processing")
    progress_bar = st.progress(0.0, text="Comparing events...")

    def update_progress(fraction: float):
        progress_bar.progress(min(fraction, 1.0), text=f"Comparing events... {fraction * 100:.0f}%")

    summary_df, detail_df = compare_alarms(
        rt_df, mf_df, site_key=site_key_choice, progress_callback=update_progress
    )
    progress_bar.progress(1.0, text="Comparison complete ✅")

    st.markdown("---")
    render_dashboard(summary_df)

    st.markdown("---")
    filtered_summary = render_filters(summary_df)

    st.markdown("---")
    st.subheader(f"📋 Comparison Report ({len(filtered_summary):,} of {len(summary_df):,} events)")
    st.dataframe(filtered_summary, use_container_width=True, height=420)

    # Filter detail rows to match the filtered summary (by RT start/site combo)
    if not detail_df.empty:
        key_cols = ["Site", "Room Temperature Start Time", "Room Temperature End Time"]
        keys = filtered_summary[key_cols].drop_duplicates()
        filtered_detail = detail_df.merge(keys, on=key_cols, how="inner")
    else:
        filtered_detail = detail_df

    with st.expander(f"🔎 View Overlapping Mains Fail Event Details ({len(filtered_detail):,} pairs)"):
        st.dataframe(filtered_detail, use_container_width=True, height=350)

    st.markdown("---")
    st.subheader("⬇️ Export")
    excel_bytes = to_excel_bytes(filtered_summary, filtered_detail)
    st.download_button(
        label="Download Comparison Report (Excel)",
        data=excel_bytes,
        file_name=f"rms_alarm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
