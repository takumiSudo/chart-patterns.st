import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
# App title and description
st.title("Algorithms for Automated Chart Pattern Trading")
st.markdown("""
In the world of technical analysis, identifying key chart patterns is crucial for making informed trading decisions. Algorithms for automated chart pattern trading offer an edge by providing precise support and resistance levels, along with marking significant highs and lows in the market. These algorithms are designed to streamline the process of chart reading, making them indispensable tools for modern traders.

In this blog post, we dive into three powerful algorithms, as explained in Neurotrader's video: "3 Must-Know Algorithms for Automating Chart Pattern Trading in Python." We’ll explore how these algorithms help traders identify critical patterns, including local tops and bottoms, which form the foundation of well-known patterns such as the Head and Shoulders. These local turning points are key markers in a time series, often used to predict future price movements. Join us as we break down these essential tools for automated chart pattern trading.

Algorithms that provide support and resistance lines for Technical Analysis as well as highs and lows. 
In this markdown, we explore 3 different algorithms that are explained in this video created by **Neurotrader's** : 
[3 Must Know algorithms for Automating Chart Pattern Trading in Python](https://www.youtube.com/watch?v=X31hyMhB-3s&list=PLMOOFzjddzEtOiP7WhQ9uQy7G_9dt3yKF&index=5).
""")

# Local Tops/Bottoms Section
st.subheader("Local Tops/Bottoms")

st.markdown("""
Many of the technical analysis instruments that traders use are made from `local tops and bottoms`.
Such as the Heads and shoulders pattern that identifies 5 local turning points in a time series data, most likely a stock chart. 
Many other chart patterns are based on the local patterns:
""")
st.image("img/chart-patterns.png", caption="Chart Patterns")

# Rolling Window Section
st.subheader("1. Rolling Window 🪟")
st.markdown("""
Conceive local tops and bottoms by **comparing it with near points**.
We can recognize a local top if the point is `higher` or `lower` than the N adjacent points.
The parameter for this algorithm is dependent on how many adjacent points we are going to consider for its comparison. 
As the number of `order` for comparison increases, the local tops will decrease.

## Rolling Window for local tops and bottoms

@data _top
top[0]: confirmation index
top[1]: top index
top[2]: price @ index

bottom[0]: confirmation index
bottom[1]: bottom index
bottom[2]: price @ index
           
## Rolling Window for local tops and bottoms
@data _top
top[0]: confirmation index
top[1]: top index
top[2]: price @ index

bottom[0]: confirmation index
bottom[1]: bottom index
bottom[2]: price @ index
""")
st.image("img/rolling-window-confirmation-point.png", caption="Rolling Window Confirmation Point")

# Code snippet for Rolling Window
st.code("""

def rw_top(data: np.array, current_index: int, order: int)-> bool:
    if current_index < order * 2 + 1:
        return False

    top = True
    k = current_index - order # subtract for the lag
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] > v or data[k - i] > v:
            top = False
            break

    return top

def rw_bottom(data: np.array, current_index: int, order: int)-> bool:
    if current_index < order * 2 + 1:
        return False

    top = True
    k = current_index - order
    v = data[k]
    for i in range(1, order + 1):
        if data[k + i] < v or data[k - i] < v:
            top = False
            break

    return top

def rw_extremes(data: np.array, order: int):

    tops = []
    bottoms = []

    for i in range(len(data)):
        if rw_top(data, i, order):
            top = [i, i - order, data[i - order]]
            tops.append(top)
        if rw_bottom(data, i, order):
            bottom = [i, i - order, data[i - order]]
            bottoms.append(bottom)

    return tops, bottoms

    """, language='python')


# Directional Change Section
st.subheader("2. Directional Change 📈")
st.markdown("""
A directional change algorithm is a local `maxima` and `minima` calculating algorithm that sets thresholds. 
When the direction change exceeds that given threshold, it allocates a local high or low at the highest point of that candlestick.
""")
st.code("""

def directional_change(close: np.array, high: np.array, low: np.array, sigma:float): 
    up_zig = True # Set last extreme is a bottom. Next is high
    tmp_max = high[0]
    tmp_min = low[0]
    tmp_max_i = 0
    tmp_min_i = 0

    tops = []
    bottoms = []

    for i in range(len(close)):
        if up_zig: # if the last extreme was low -> next is high
            if high[i] > tmp_max:
                tmp_max = high[i]
                tmp_max_i = i
            elif close[i] < tmp_max - tmp_max * sigma: # if the close[i] price was lower than the tmp_max - (tmp_max * sigma)
                top = [i, tmp_max_i, tmp_max]
                tops.append(top)
        else: # last extreme was a high -> next is a low
            if low[i] > tmp_min:
                tmp_min = low[i]
                tmp_min_i = i
            elif close[i] > tmp_min - tmp_min * sigma:
                bottom = [i, tmp_min_i, tmp_min]
                bottoms.append(bottom)

                # Setup for next top
                up_zig = True
                tmp_max = high[i]
                tmp_max_i = i

    return tops, bottoms

def get_extremes(ohlc: pd.DataFrame, sigma: float):
    tops, bottoms = directional_change(ohlc['close'], ohlc['high'], ohlc['low'], sigma)
    tops = pd.DataFrame(tops, columns=['conf_i', 'ext_i', 'ext_p'])
    bottoms = pd.DataFrame(bottoms, columns=['conf_i', 'ext_i', 'ext_p'])
    tops['type'] = 1
    bottoms['type'] = -1
    extremes = pd.concat([tops, bottoms])
    extremes = extremes.set_index('conf_i')
    extremes = extremes.sort_index()
""", language='python')

# Perceptually Important Points Section
st.subheader("3. Perceptually Important Points 🔭")
st.markdown("""
Perceptually Important Points (PIPs) in technical analysis are key price levels on a chart that represent significant changes in market direction...
""")
st.image("img/pip-support-line.png", caption="PIP Support Line")
st.image("img/pip-support-line-2.png", caption="PIP Support Line Example")

st.code("""
def find_pips(data: np.array, n_pips: int, dist_measure: int):
    

    pips_x = [0, len(data) - 1]  # Index
    pips_y = [data[0], data[-1]] # Price

    for curr_point in range(2, n_pips):

        md = 0.0 # Max distance
        md_i = -1 # Max distance index
        insert_index = -1

        for k in range(0, curr_point - 1):

            # Left adjacent, right adjacent indices
            left_adj = k
            right_adj = k + 1

            time_diff = pips_x[right_adj] - pips_x[left_adj]
            price_diff = pips_y[right_adj] - pips_y[left_adj]
            slope = price_diff / time_diff
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope;

            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                
                d = 0.0 # Distance
                if dist_measure == 1: # Euclidean distance
                    d =  ( (pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2 ) ** 0.5
                    d += ( (pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2 ) ** 0.5
                elif dist_measure == 2: # Perpindicular distance
                    d = abs( (slope * i + intercept) - data[i] ) / (slope ** 2 + 1) ** 0.5
                else: # Vertical distance    
                    d = abs( (slope * i + intercept) - data[i] )

                if d > md:
                    md = d
                    md_i = i
                    insert_index = right_adj

        pips_x.insert(insert_index, md_i)
        pips_y.insert(insert_index, data[md_i])

    return pips_x, pips_y

""", language='python')

# Closing Remarks
st.markdown("""
This application demonstrates the implementation of three critical algorithms for automated chart pattern trading. 
You can tweak parameters and visualize the results to better understand their behavior in real-time.
""")


# Importing the functions from your script
from algos import rw_extremes, directional_change, find_pips

# Load the dataset (assuming it's in the same directory)
data = pd.read_csv('BTCUSDT86400.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')

# Select Algorithm
with st.sidebar:
    st.title("🚀 Traditional Pattern Mining")
    linkedin_url = "https://www.linkedin.com/in/takumisudo"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Takumi Sudo`</a>', unsafe_allow_html=True)
    st.sidebar.title("Algorithm Settings")
    algo_option = st.sidebar.selectbox(
        "Choose an Algorithm",
        ("Rolling Window", "Directional Change", "Perceptually Important Points")
    )

if algo_option == "Rolling Window":
    st.header("Rolling Window Algorithm")

    # Rolling Window Parameters
    order = st.sidebar.slider('Select Rolling Window Order', 1, 20, 10)

    st.write(f"Rolling Window Order: {order}")

    # Compute Rolling Window Extremes
    tops, bottoms = rw_extremes(data['close'].to_numpy(), order)

    # Plotting the results
    fig, ax = plt.subplots()
    ax.plot(data['close'], label='Close Price')
    idx = data.index
    for top in tops:
        ax.plot(idx[top[1]], top[2], marker='o', color='green', label='Top' if top == tops[0] else "")
    for bottom in bottoms:
        ax.plot(idx[bottom[1]], bottom[2], marker='o', color='red', label='Bottom' if bottom == bottoms[0] else "")
    ax.legend()
    st.pyplot(fig)

elif algo_option == "Directional Change":
    st.header("Directional Change Algorithm")

    # Directional Change Parameters
    sigma = st.sidebar.slider('Select Sigma (Threshold Percentage)', 0.01, 0.1, 0.02)

    st.write(f"Directional Change Sigma: {sigma}")

    # Compute Directional Change Extremes
    tops, bottoms = directional_change(
        data['close'].to_numpy(), 
        data['high'].to_numpy(), 
        data['low'].to_numpy(), 
        sigma
    )

    # Plotting the results
    fig, ax = plt.subplots()
    ax.plot(data['close'], label='Close Price')
    idx = data.index
    for top in tops:
        ax.plot(idx[top[1]], top[2], marker='o', color='green', markersize=4, label='Top' if top == tops[0] else "")
    for bottom in bottoms:
        ax.plot(idx[bottom[1]], bottom[2], marker='o', color='red', markersize=4, label='Bottom' if bottom == bottoms[0] else "")
    ax.legend()
    st.pyplot(fig)

elif algo_option == "Perceptually Important Points":
    st.header("Perceptually Important Points Algorithm")

    # PIPs Parameters
    n_pips = st.sidebar.slider('Select Number of PIPs', 2, 20, 5)
    dist_measure = st.sidebar.selectbox(
        "Distance Measurement Method",
        ("Euclidean Distance", "Perpendicular Distance", "Vertical Distance")
    )

    dist_measure_map = {"Euclidean Distance": 1, "Perpendicular Distance": 2, "Vertical Distance": 3}
    dist_measure_int = dist_measure_map[dist_measure]

    st.write(f"Number of PIPs: {n_pips}")
    st.write(f"Distance Measurement Method: {dist_measure}")

    # Compute PIPs
    i = 1198  # Example index, adjust as needed
    x = data['close'].iloc[i-40:i].to_numpy()
    pips_x, pips_y = find_pips(x, n_pips, dist_measure_int)

    # Plotting the results
    fig, ax = plt.subplots()
    ax.plot(x, label='Close Price')
    for i in range(n_pips):
        ax.plot(pips_x[i], pips_y[i], marker='o', color='red', label='PIP' if i == 0 else "")
    ax.legend()
    st.pyplot(fig)

# Adding a footer with a disclaimer
st.markdown("""
---
**Disclaimer**: This is a demonstration application for educational purposes. 
The algorithms used here are for technical analysis, and trading involves substantial risk. 
Ensure you understand the risks before trading based on these or any other algorithms.
""")
