import streamlit as st
import pandas as pd
import math
import json
from itertools import combinations
import pygsheets
from datetime import datetime
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

def get_time():
    service_account_info = dict(st.secrets["gcp_service_account"])
    
    gc = pygsheets.authorize(service_account_info=service_account_info)
    sh = gc.open('UD_OddsProphet')
    updated_time = datetime.fromisoformat(sh.updated.replace("Z", "+00:00"))
    return NULL

col1, col2, col3 = st.columns(3)
st.title("Odds:blue[Prophet]")
st.divider()

r_col1, r_col2 = st.columns([1,1])

with r_col1:
    modify = st.checkbox("Create filters")
with r_col2:
    show_sportsbooks = st.checkbox("Show Sportsbook Columns", value=False)

@st.cache_data(ttl="4m")
def load_data(sheets_url):
    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    return pd.read_csv(csv_url), 

df = load_data(st.secrets['public_gsheets_url'])[0]

df.columns = df.columns.str.replace('Over', 'O')
df.columns = df.columns.str.replace('Under', 'U')
df['Start'] = pd.to_datetime(df['Start'])
df['Start'] = df['Start'].dt.strftime("%a %I:%M %p")
sportsbooks =  df.columns[df.columns.get_loc('Best Odds')+1:-1].to_list()

dataset = st.container()

def highlight_row(row, columns):
    bg_colors = [''] * len(columns)
    pairs = len(columns) // 2
    for i in range(pairs):
        over_val = row[2*i]
        under_val = row[2*i+1]
        if '.1' in str(over_val) or '.1' in str(under_val):
            bg_colors[2*i:2*i+2] = ['background-color: orange']*2
        elif over_val < under_val:
            bg_colors[2*i:2*i+2] = ['background-color: #009177']*2
        elif over_val > under_val:
            bg_colors[2*i:2*i+2] = ['background-color: #644EC7']*2
    return bg_colors

def highlight_above_val(val):
    """
    Highlights values >= 53.7% in Probability column.
    """
    return "color: springgreen; font-weight: bold;" if val >= 53.7 else ""


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    
    st.subheader("Underdog Fantasy Props")
    
    # Convert datetimes once
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    if not modify:
        return df

    modification_container = st.container()
    with modification_container:
        filter_cols = ['Sport', 'Favorite', 'Probability', 'Best Odds', 'Start']
        to_filter_columns = st.multiselect("Choose a column to filter on", df[filter_cols].columns)
        
        filtered_df = df.copy()
        for column in to_filter_columns:
            # Treat columns with <10 unique as categorical
            if is_categorical_dtype(filtered_df[column]) or filtered_df[column].nunique() < 10:
                user_cat_input = st.multiselect(
                    f"Values for {column}",
                    filtered_df[column].unique(),
                    default=list(filtered_df[column].unique())
                )
                filtered_df = filtered_df[filtered_df[column].isin(user_cat_input)]
            elif is_numeric_dtype(filtered_df[column]):
                _min = float(filtered_df[column].min())
                _max = float(filtered_df[column].max())
                step = (_max - _min) / 100 if _max != _min else 1
                user_num_input = st.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                filtered_df = filtered_df[filtered_df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(filtered_df[column]):
                user_date_input = st.date_input(
                    f"Values for {column}",
                    value=(filtered_df[column].min(), filtered_df[column].max()),
                )
                if len(user_date_input) == 2:
                    start_date, end_date = map(pd.to_datetime, user_date_input)
                    filtered_df = filtered_df.loc[filtered_df[column].between(start_date, end_date)]
            else:
                user_text_input = st.text_input(
                    f"Search for {column} (Case sensitive)",
                )
                if user_text_input:
                    filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(user_text_input)]

        return filtered_df


with dataset:
    final = filter_dataframe(df)

    # Toggle to show/hide sportsbook columns


    # If user decides to hide them, drop sportsbook columns
    displayed_df = final.copy()
    displayed_sportsbooks = sportsbooks.copy()

    if not show_sportsbooks:
        displayed_df = displayed_df.drop(columns=sportsbooks)
        displayed_sportsbooks = []

    # Format dictionary
    format_dict = {'Line': '{:.1f}', 'Probability': '{:.1f}%', 'Best Odds': '{:.0f}'}
    format_dict.update({col: '{:.0f}' for col in displayed_sportsbooks})

    styled_df = displayed_df.style.format(format_dict)

    if displayed_sportsbooks:
        styled_df = styled_df.apply(
            lambda row: highlight_row(row, displayed_sportsbooks),
            axis=1,
            subset=displayed_sportsbooks
        )

    # Apply top 10% probability highlight if Probability column is available
    if 'Probability' in displayed_df.columns:
        styled_df = styled_df.applymap(highlight_top_10, subset=['Probability'])

    # Create the selectable dataframe event
    event = st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",      # Re-run app on selection
        selection_mode="multi-row"  # Enable multi-row selection
    )

    # Get selected rows from the event
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
    simulate_taco = st.checkbox("Simulate Taco :taco:", value=False)

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
    if simulate_taco and available_spots > 0:
        new_row = pd.DataFrame([{"Sport": "SIMULATED TACO", "Probability": 63.0}])
        selected_bets = pd.concat([selected_bets, new_row], ignore_index=True)
        available_spots -= 1

num_picks = len(selected_bets)

if num_picks > 0:
    # Convert probabilities to fractions
    probs = (selected_bets['Probability'] / 100).values
    p_all = math.prod(probs)  # Probability of all correct

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

    # Show Power EV
    if power_ev is not None:
        container.write(
            f"**Power Play EV:** {' :blue[{:.2f}]'.format(power_ev) if power_ev > 0 else f'{power_ev:.2f}'}"
        )
    else:
        container.write("**Power Play** not available for this number of picks.")

    # Show Flex EV
    if flex_ev is not None:
        container.write(
            f"**Flex Play EV:** {' :blue[{:.2f}]'.format(flex_ev) if flex_ev > 0 else f'{flex_ev:.2f}'}"
        )
    else:
        container.write("**Flex Play** not available for this number of picks.")
else:
    st.write("No bets selected. Please select rows above to calculate Expected Value.")
