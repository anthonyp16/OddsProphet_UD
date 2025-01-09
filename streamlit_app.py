import math
from itertools import combinations
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pygsheets
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.set_page_config(
    page_title="OddsProphet | Dashboard",
    page_icon="💰",
    layout="wide"
)
st.title("Odds:blue[Prophet]")
st.divider()

def get_sheet_update_time() -> str:

    gc = pygsheets.authorize(service_file="oddstool-fdc41ddfa8e3.json")
    sh = gc.open("Odds_Tool")
    updated_time_utc = datetime.fromisoformat(sh.updated.replace("Z", "+00:00"))
    eastern_time = updated_time_utc.astimezone(ZoneInfo("America/New_York"))
    return eastern_time.strftime("%-m/%-d/%y %I:%M %p %Z")

@st.cache_data(ttl="4m")
def load_data(sheets_url: str):

    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    df = pd.read_csv(csv_url)
    #last_updated_str = get_sheet_update_time()
    last_updated_str = "placeholder"
    return df, last_updated_str

# --------------------
# Highlighting Helpers
# --------------------
def highlight_row(row, columns):

    bg_colors = [''] * len(columns)
    pairs = len(columns) // 2
    for i in range(pairs):
        over_val = row[2*i]
        under_val = row[2*i + 1]
        if '.1' in str(over_val) or '.1' in str(under_val):
            bg_colors[2*i:2*i+2] = ['background-color: orange']*2
        elif over_val < under_val:
            bg_colors[2*i:2*i+2] = ['background-color: #009177']*2
        elif over_val > under_val:
            bg_colors[2*i:2*i+2] = ['background-color: #644EC7']*2
    return bg_colors

def highlight_above_val(val):
    return "color: springgreen; font-weight: bold;" if val >= 53.7 else ""

# --------------------
# Data Filtering
# --------------------
def filter_dataframe(df: pd.DataFrame, enable_filter: bool) -> pd.DataFrame:
    st.subheader("Underdog Fantasy Props")

    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    if not enable_filter:
        return df

    filter_cols = ['Sport', 'Favorite', 'Probability', 'Best Odds', 'Start']
    to_filter = st.multiselect("Choose columns to filter", df[filter_cols].columns)

    filtered_df = df.copy()
    for column in to_filter:
        if is_categorical_dtype(filtered_df[column]) or filtered_df[column].nunique() < 10:
            user_cat_input = st.multiselect(
                f"{column}",
                filtered_df[column].unique(),
                default=list(filtered_df[column].unique())
            )
            filtered_df = filtered_df[filtered_df[column].isin(user_cat_input)]
        elif is_numeric_dtype(filtered_df[column]):
            _min, _max = float(filtered_df[column].min()), float(filtered_df[column].max())
            step = (_max - _min) / 100 if _max != _min else 1
            user_num_input = st.slider(
                f"{column}",
                min_value=_min,
                max_value=_max,
                value=(_min, _max),
                step=step,
            )
            filtered_df = filtered_df[filtered_df[column].between(*user_num_input)]
        elif is_datetime64_any_dtype(filtered_df[column]):
            user_date_input = st.date_input(
                f"{column}",
                value=(filtered_df[column].min(), filtered_df[column].max()),
            )
            if len(user_date_input) == 2:
                start_date, end_date = map(pd.to_datetime, user_date_input)
                filtered_df = filtered_df.loc[filtered_df[column].between(start_date, end_date)]
        else:
            user_text_input = st.text_input(f"Search in {column} (Case sensitive)")
            if user_text_input:
                filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(user_text_input)]
    return filtered_df

# ---------------------------
# Main Layout & Data Retrieval
# ---------------------------
col1, col2, col3 = st.columns(3)
r_col1, r_col2 = st.columns([1, 1])
with r_col1:
    modify = st.checkbox("Create filters")
with r_col2:
    show_sportsbooks = st.checkbox("Show Sportsbook Columns", value=False)

df, last_updated_str = load_data(st.secrets["public_gsheets_url"])
df.columns = df.columns.str.replace('Over', 'O')
df.columns = df.columns.str.replace('Under', 'U')
sportsbooks = df.columns[df.columns.get_loc('Best Odds')+1:-1].to_list()
df['Start'] = pd.to_datetime(df['Start'])
df['Start'] = df['Start'].dt.strftime("%a %I:%M %p")

# -------------
# Filter & Show
# -------------
filtered_df = filter_dataframe(df, enable_filter=modify)
displayed_df = filtered_df.copy()
displayed_sportsbooks = sportsbooks.copy()


if not show_sportsbooks:
    displayed_df.drop(columns=sportsbooks, inplace=True)
    displayed_sportsbooks = []


format_dict = {'Line': '{:.1f}', 'Probability': '{:.1f}%', 'Best Odds': '{:.0f}'}
format_dict.update({col: '{:.0f}' for col in displayed_sportsbooks})
styled_df = displayed_df.style.format(format_dict)

if displayed_sportsbooks:
    styled_df = styled_df.apply(
        lambda row: highlight_row(row, displayed_sportsbooks),
        axis=1,
        subset=displayed_sportsbooks
    )
if 'Probability' in displayed_df.columns:
    styled_df = styled_df.applymap(highlight_above_val, subset=['Probability'])

event = st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",      
    selection_mode="multi-row" 
)
selected_rows = event.selection.rows if event.selection is not None else []
selected_bets = displayed_df.iloc[selected_rows, :7] if len(selected_rows) > 0 else pd.DataFrame()
st.caption(f":gray[Last Updated:] ")
st.divider()

# -----------------------------
# EV Calculator & Probability
# -----------------------------
st.subheader("Expected Value Calculator")
bet_amount = st.number_input("Bet Amount ($)", min_value=5, value=10, format="%d")

sim_col1, sim_col2 = st.columns(2)
with sim_col1:
    simulate_free_square = st.checkbox("Simulate Free Square", value=False)
with sim_col2:
    simulate_dp = st.checkbox("Simulate Discounted Pick", value=False)

max_picks = 6
current_picks = len(selected_bets)
available_spots = max_picks - current_picks

if available_spots <= 0:
    st.warning("You already have 6 picks. Cannot add more.")
else:
    if simulate_free_square and available_spots > 0:
        new_row = pd.DataFrame([{"Sport": "SIMULATED FREE SQUARE", "Probability": 99.0}])
        selected_bets = pd.concat([selected_bets, new_row], ignore_index=True)
        available_spots -= 1
    if simulate_dp and available_spots > 0:
        new_row = pd.DataFrame([{"Sport": "SIMULATED DISCOUNTED PICK", "Probability": 63.0}])
        selected_bets = pd.concat([selected_bets, new_row], ignore_index=True)
        available_spots -= 1

num_picks = len(selected_bets)

if num_picks > 0:
    probs = (selected_bets['Probability'] / 100).values
    p_all = math.prod(probs) 

    def prob_exact_k(prbs, k):
        total = 0
        for combo in combinations(range(len(prbs)), k):
            p = 1
            for i in range(len(prbs)):
                p *= prbs[i] if i in combo else (1 - prbs[i])
            total += p
        return total

    power_payouts = {2: 3, 3: 5, 4: 10, 5: 20, 6: 37.5}
    flex_payouts = {
        3: {3: 2.25, 2: 1.25},
        4: {4: 5,    3: 1.5},
        5: {5: 10,   4: 2,    3: 0.4},
        6: {6: 25,   5: 2,    4: 0.4}
    }

    power_ev = bet_amount * (power_payouts[num_picks] * p_all - 1) if num_picks in power_payouts else None

    if num_picks in flex_payouts:
        flex_expected_mult = 0
        for k, mult in flex_payouts[num_picks].items():
            flex_expected_mult += prob_exact_k(probs, k) * mult
        flex_ev = bet_amount * (flex_expected_mult - 1)
    else:
        flex_ev = None

    st.text("Selected Bets:")
    st.dataframe(selected_bets, use_container_width=True, hide_index=True)

    container = st.container()

    if power_ev is not None:
        container.write(
            f"**Power Play EV:** {' :blue[{:.2f}]'.format(power_ev) if power_ev > 0 else f'{power_ev:.2f}'}"
        )
    else:
        container.write("**Power Play** not available for this number of picks.")

    if flex_ev is not None:
        container.write(
            f"**Flex Play EV:** {' :blue[{:.2f}]'.format(flex_ev) if flex_ev > 0 else f'{flex_ev:.2f}'}"
        )
    else:
        container.write("**Flex Play** not available for this number of picks.")
else:
    st.write("No bets selected. Please select rows above to calculate Expected Value.")
