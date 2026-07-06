import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import datetime

# Constants
REQUIRED_COLS_RT = ['Rms Station', 'Site', 'Site Alias', 'Zone', 'Cluster', 'Tenant', 'Tag', 'Start Time', 'End Time', 'Elapsed Time', 'Duration (Hr.)', 'Acknowledged Time', 'Acknowledged By', 'Alarm KPI Group']
REQUIRED_COLS_MF = same? Actually both have same columns, so we can use same list for validation.

def load_excel(uploaded_file):
    return pd.read_excel(uploaded_file)

def validate_columns(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def convert_datetimes(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def compute_union_overlap(intervals):
    # intervals: list of (start, end) as Timestamp
    if not intervals:
        return pd.Timedelta(0)
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for start, end in sorted_intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    total = sum((end - start).total_seconds() for start, end in merged) / 3600.0  # hours
    return total

def perform_overlap(rt_df, mf_df, match_col):
    # rt_df and mf_df already have datetime columns converted
    # We'll create a copy to avoid modifying original
    rt = rt_df.copy()
    mf = mf_df.copy()
    
    # We'll assign a unique index to RT events
    rt['rt_id'] = range(len(rt))
    # For grouping, we'll use the match_col
    # Ensure both have that column
    if match_col not in rt.columns or match_col not in mf.columns:
        raise ValueError(f"Column '{match_col}' not found in both dataframes.")
    
    # Group by match_col
    groups_rt = rt.groupby(match_col)
    groups_mf = mf.groupby(match_col)
    
    # Prepare lists to accumulate summary rows and detail rows
    summary_rows = []
    detail_rows = []
    
    # Get all match_col values present in RT
    rt_values = set(rt[match_col].unique())
    # We'll process each value
    total_groups = len(rt_values)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, value in enumerate(rt_values):
        status_text.text(f"Processing group {idx+1}/{total_groups}: {value}")
        rt_group = groups_rt.get_group(value) if value in groups_rt.groups else pd.DataFrame()
        mf_group = groups_mf.get_group(value) if value in groups_mf.groups else pd.DataFrame()
        
        if rt_group.empty:
            continue
        
        # For RT events in this group, we need to determine overlaps
        if mf_group.empty:
            # No MF events for this site, all RT events have no overlap
            for _, row in rt_group.iterrows():
                summary_rows.append({
                    'rt_id': row['rt_id'],
                    'MF Occurred': 'No',
                    'MF Count': 0,
                    'Total Overlap Duration (Hr.)': 0,
                    'Coverage %': 0,
                })
            # No detail rows
        else:
            # Perform cross join
            # We'll add a temporary key
            rt_group_temp = rt_group.copy()
            mf_group_temp = mf_group.copy()
            rt_group_temp['_key'] = 1
            mf_group_temp['_key'] = 1
            merged = pd.merge(rt_group_temp, mf_group_temp, on='_key', suffixes=('_rt', '_mf'))
            merged.drop('_key', axis=1, inplace=True)
            
            # Filter by overlap condition
            merged = merged[
                (merged['Start Time_mf'] <= merged['End Time_rt']) &
                (merged['End Time_mf'] >= merged['Start Time_rt'])
            ]
            
            # If no overlaps
            if merged.empty:
                for _, row in rt_group.iterrows():
                    summary_rows.append({
                        'rt_id': row['rt_id'],
                        'MF Occurred': 'No',
                        'MF Count': 0,
                        'Total Overlap Duration (Hr.)': 0,
                        'Coverage %': 0,
                    })
            else:
                # Now we need to aggregate per rt_id
                # For each rt_id, collect list of (start_mf, end_mf)
                # Also compute union overlap duration
                # We'll group by rt_id
                for rt_id, group in merged.groupby('rt_id'):
                    # Get the RT row info (we can get first row of group which has RT info)
                    rt_row = group.iloc[0]  # all have same RT info
                    # Collect MF intervals
                    intervals = list(zip(group['Start Time_mf'], group['End Time_mf']))
                    # Compute union overlap
                    overlap_hours = compute_union_overlap(intervals)
                    # Compute RT duration (in hours)
                    rt_duration = (rt_row['End Time_rt'] - rt_row['Start Time_rt']).total_seconds() / 3600.0
                    coverage_pct = (overlap_hours / rt_duration) * 100 if rt_duration > 0 else 0
                    
                    # Add summary row
                    summary_rows.append({
                        'rt_id': rt_id,
                        'MF Occurred': 'Yes',
                        'MF Count': len(group),
                        'Total Overlap Duration (Hr.)': overlap_hours,
                        'Coverage %': coverage_pct,
                    })
                    
                    # Add detail rows for each overlapping MF event
                    for _, mf_row in group.iterrows():
                        detail_rows.append({
                            'rt_id': rt_id,
                            'MF Start Time': mf_row['Start Time_mf'],
                            'MF End Time': mf_row['End Time_mf'],
                            'MF Duration (Hr.)': mf_row['Duration (Hr.)_mf'],
                        })
                
                # For RT events that have no overlap, they won't appear in merged.
                # We need to add summary rows for those with count 0.
                all_rt_ids = set(rt_group['rt_id'])
                overlapped_ids = set(merged['rt_id'])
                no_overlap_ids = all_rt_ids - overlapped_ids
                for rid in no_overlap_ids:
                    summary_rows.append({
                        'rt_id': rid,
                        'MF Occurred': 'No',
                        'MF Count': 0,
                        'Total Overlap Duration (Hr.)': 0,
                        'Coverage %': 0,
                    })
        
        progress_bar.progress((idx+1)/total_groups)
    
    status_text.empty()
    progress_bar.empty()
    
    # Create summary dataframe from summary_rows
    summary_df = pd.DataFrame(summary_rows)
    # Merge with original RT to get all columns
    # We need to merge on rt_id
    rt_with_id = rt.reset_index(drop=True)  # ensure rt_id is present
    summary_df = pd.merge(rt_with_id, summary_df, on='rt_id', how='left')
    # Fill missing if any (shouldn't)
    summary_df['MF Occurred'] = summary_df['MF Occurred'].fillna('No')
    summary_df['MF Count'] = summary_df['MF Count'].fillna(0)
    summary_df['Total Overlap Duration (Hr.)'] = summary_df['Total Overlap Duration (Hr.)'].fillna(0)
    summary_df['Coverage %'] = summary_df['Coverage %'].fillna(0)
    
    # Create details dataframe
    details_df = pd.DataFrame(detail_rows)
    
    return summary_df, details_df
