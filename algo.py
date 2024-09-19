#!/usr/bin/env python3

"""
Automated Chart Patterns
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

"""
# Rolling Window Algorithm
Confirms if there is a local top detected given the current index
@inputs(data, current_index, order)
@return(bool)
"""
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

"""
Rolling Window for local tops and bottoms

@data _top
top[0]: confirmation index
top[1]: top index
top[2]: price @ index

bottom[0]: confirmation index
bottom[1]: bottom index
bottom[2]: price @ index
"""
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


"""
Directional Change Algorithm
@Input(close price, high of current index, low of the current index, sigma: hreshold percentage)
@Return

extreme[0] = confirmation index
extreme[1] = index of extreme
extreme[2] = price of extreme
"""
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

"""
Perceptually Important Points
    # dist_measure
    # 1 = Euclidean Distance
    # 2 = Perpindicular Distance
    # 3 = Vertical Distance

@input(data, number of pips, distance measuring method)
@return()
"""
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Chart Pattern Technical Analysis")

    parser.add_argument('--algo', default = "roll", help="Algorithm Argument Flag. \nOptions: roll[rolling-window], dir[directional_change], pip[Perceptually Important Points]")
    return parser.parse_args()



if __name__ == "__main__":

    data = pd.read_csv('data/BTCUSDT86400.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')

    args = parse_arguments()

    if args.algo == "roll":

        tops, bottoms = rw_extremes(data['close'].to_numpy(), 10)
        data['close'].plot()
        idx = data.index
        for top in tops:
            plt.plot(idx[top[1]], top[2], marker='o', color='green')
        for bottom in bottoms:
            plt.plot(idx[bottom[1]], bottom[2], marker='o', color='red')
        plt.show()
    elif args.algo == "dir":
        tops, bottoms = directional_change(data['close'].to_numpy(), data['high'].to_numpy(), data['low'].to_numpy(), 0.02)
        pd.Series(data['close'].to_numpy()).plot()
        idx = data.index
        for top in tops:
            plt.plot(top[1], top[2], marker='o', color='green', markersize=4)
        plt.show()
    elif args.algo == "pip":
        i = 1198
        x = data['close'].iloc[i-40:i].to_numpy()
        pips_x, pips_y = find_pips(x, 5, 2)

        pd.Series(x).plot()
        for i in range(5):
            plt.plot(pips_x[i], pips_y[i], marker='o', color='red')
        plt.show()
