import csv
import os
import numpy as np

# File paths (assumes the files are in the same directory as this script)
data = os.path.join(os.path.dirname(__file__), 'OriginalDataset.csv')
output = os.path.join(os.path.dirname(__file__), 'TrainingDataAdaptedOutput2.csv')


# -----------------------------
# 1. Read original CSV and group by time and SCATS
# -----------------------------
# We assume that the header row contains the 15-minute time columns in positions 10 to -3.
with open(data, 'r', newline='') as f_in:
    reader = csv.reader(f_in)
    # Read the first row (header)
    header_row = next(reader)
    # The time-of-day columns (e.g. "00:00", "00:15", "00:30", ...)
    times_15min = header_row[10:-3]

    # Dictionary to store summed flows
    # key = (date_str, scats_id, time_str), value = sum_of_flows
    summed_flows = {}

    for row in reader:
        scats_id = row[0]
        date_str = row[5]  # e.g. "1/10/2006"

        # Loop through each 15-min column in this row
        for i, time_str in enumerate(times_15min):
            try:
                flow_val = float(row[10 + i])  # read the flow
            except ValueError:
                flow_val = 0.0  # or skip if needed

            key = (date_str, scats_id, time_str)
            summed_flows[key] = summed_flows.get(key, 0.0) + flow_val


# ------------------------------------------------------------------
# 3. Upsample each (date, SCATS) set of 15-min flows to 5-min intervals
# ------------------------------------------------------------------
def time_to_minutes(t_str):
    """Convert 'HH:MM' to an integer minute-of-day (0–1439)."""
    hh, mm = t_str.split(':')
    return int(hh) * 60 + int(mm)


def minutes_to_time(m):
    """Convert integer minute-of-day back to 'HH:MM'."""
    hh = m // 60
    mm = m % 60
    return f"{hh:02d}:{mm:02d}"


# Group the summed flows by (date, SCATS), each storing a list of (minute_of_day, flow).
grouped_by_date_scats = {}
for (date_str, scats_id, time_str), flow_val in summed_flows.items():
    minute_of_day = time_to_minutes(time_str)
    grouped_by_date_scats.setdefault((date_str, scats_id), []).append((minute_of_day, flow_val))

# The new 5-min time grid (0, 5, 10, ..., 1435)
time_grid_5min = np.arange(0, 1440, 5, dtype=int)

# We'll collect all upsampled rows here to write later
upsampled_rows = []

for (date_str, scats_id), flows in grouped_by_date_scats.items():
    # Sort flows by minute_of_day
    flows.sort(key=lambda x: x[0])  # (minute_of_day, flow_val)

    x_orig = np.array([pt[0] for pt in flows], dtype=float)  # original times
    y_orig = np.array([pt[1] for pt in flows], dtype=float)  # original flows

    # Use np.interp for linear interpolation onto the 5-min grid
    y_new = np.interp(time_grid_5min, x_orig, y_orig)

    # Add each upsampled point as a row: [ "date time", flow, #points, %observed, scats_id ]
    for m_of_day, flow_val in zip(time_grid_5min, y_new):
        t_str = minutes_to_time(m_of_day)
        # The first column is "5 Minutes" in your final file, but we can store date+time if you prefer
        date_time_str = f"{date_str} {t_str}"
        upsampled_rows.append([date_time_str, flow_val, 1, 100, scats_id])

# ------------------------------------------------------------------
# 4. Write the upsampled data to output CSV
# ------------------------------------------------------------------
# We'll mimic your original header style
header = ["5 Minutes", "Lane 1 Flow (Veh/5 Minutes)", "# Lane Points", "% Observed", "SCATS"]

with open(output, 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(header)
    for row_vals in upsampled_rows:
        writer.writerow(row_vals)

# ------------------------------------------------------------------
# 5. Split into train and test sets
# ------------------------------------------------------------------
with open(output, 'r') as fid:
    lines = fid.readlines()

test_data_size = 72000
row_count = len(lines)

# If you want to ensure the header is repeated, do:
# train_lines.insert(0, ",".join(header) + "\n")
# test_lines.insert(0, ",".join(header) + "\n")
# But since we already wrote the header in output, the first line is already the header.

#header = "5 Minutes,Lane 1 Flow (Veh/5 Minutes),# Lane Points,% Observed,SCATS\n"


print("Upsampled and grouped data saved. Train and test files are ready.")
