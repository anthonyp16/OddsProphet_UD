import streamlit as st
import pandas as pd
import math
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
    gc = pygsheets.authorize(service_file='oddstool-fdc41ddfa8e3.json')
    sh = gc.open('Odds_Tool')
    updated_time = datetime.fromisoformat(sh.updated.replace("Z", "+00:00"))
    return updated_time.strftime("%-m/%-d/%y %I:%M %p %Z")

col1, col2, col3 = st.columns(3)
st.title("Odds:blue[Prophet]")
st.divider()

r_col1, r_col2 = st.columns([1,1])

with r_col1:
    modify = st.checkbox("Create filters")
with r_col2:
    show_sportsbooks = st.checkbox("Show Sportsbook Columns", value=False)

@st.cache_data(ttl="5m")
def load_data(sheets_url):
    csv_url = sheets_url.replace("/edit#gid=", "/export?format=csv&gid=")
    return pd.read_csv(csv_url), 

df = load_data(st.secrets["public_gsheets_url"])[0]


df.columns = df.columns.str.replace('Over', 'O')
df.columns = df.columns.str.replace('Under', 'U')

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



def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    
    st.subheader("PrizePicks Props")
    
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

    # Determine top 10% threshold for Probability column (if it exists)
    if 'Probability' in displayed_df.columns and pd.api.types.is_numeric_dtype(displayed_df['Probability']):
        top_10_thresh = displayed_df['Probability'].quantile(0.89)
    else:
        top_10_thresh = None

    def highlight_top_10(val):
        if top_10_thresh is not None and val >= top_10_thresh:
            return "color: springgreen; font-weight: bold;"
        return ""

    # Format dictionary
    format_dict = {'Line': '{:.1f}', 'Probability': '{:.1f}%', 'Best Odds': '{:.0f}'}
    format_dict.update({col: '{:.0f}' for col in displayed_sportsbooks})

    styled_df = displayed_df.style.format(format_dict)

    # Apply sportsbook highlights only if sportsbooks are visible
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
    selected_bets = displayed_df.iloc[selected_rows] if len(selected_rows) > 0 else pd.DataFrame()
    st.caption(f":gray[Last Updated:] {get_time()}")
    st.divider()


# EV Calculator Section
    st.subheader("Expected Value Calculator")
    num_picks = len(selected_bets)
    bet_amount = st.number_input("Bet Amount ($)", min_value=5, value=10, format="%d")
    if num_picks > 0:
        probs = (selected_bets['Probability'] / 100).values
        p_all = math.prod(probs)

        # Helper for probability of exactly k correct
        def prob_exact_k(probs, k):
            total = 0
            for combo in combinations(range(len(probs)), k):
                p = 1
                for i in range(len(probs)):
                    if i in combo:
                        p *= probs[i]
                    else:
                        p *= (1 - probs[i])
                total += p
            return total

        # Payout structures
        power_payouts = {
            2: 3, 3: 5, 4: 10, 5: 20, 6: 37.5
        }

        flex_payouts = {
            3: {3:2.25, 2:1.25},
            4: {4:5, 3:1.5},
            5: {5:10, 4:2, 3:0.4},
            6: {6:25, 5:2, 4:0.4}
        }

        # Calculate EV for power play
        if num_picks in power_payouts:
            power_ev = bet_amount * (power_payouts[num_picks]*p_all - 1)
        else:
            power_ev = None

        # Calculate EV for flex play (if defined)
        if num_picks in flex_payouts:
            flex_expected_mult = 0
            for k, mult in flex_payouts[num_picks].items():
                flex_expected_mult += prob_exact_k(probs, k)*mult
            # EV = sum(prob(k)*mult(k)*100) - 100
            flex_ev = bet_amount * (flex_expected_mult - 1)
        else:
            flex_ev = None

        st.text("Selected Bets:")
        st.dataframe(selected_bets, use_container_width=True, hide_index=True)
        container = st.container(border=True)


        # Display results
        if power_ev is not None:
            container.write(f"**Power Play EV:** {power_ev:.2f}")
        else:
            container.write("**Power Play** not available for this number of picks.")

        if flex_ev is not None:
            container.write(f"**Flex Play EV:** {flex_ev:.2f}")
        else:
            container.write("**Flex Play** not available for this number of picks.")

    else:
        st.write("No bets selected. Please select rows above to calculate Expected Value.")
