import yfinance as yf
import numpy as np
# from mpl_finance import candlestick_ohlc
import pandas as pd
# import matplotlib.dates as mpl_dates

import datetime
def plot_candle(data,ax,trend_col = '',trend_line_col = ''):
    # fig, ax = plt.subplots()
    stock_prices = data.copy()
    # stock_prices.index = [i for i in range(stock_prices.shape[0])]

    # "up" dataframe will store the stock_prices
    # when the closing stock price is greater
    # than or equal to the opening stock prices
    # trend_col = 'trend_sign3'
    # trend_line_col = ''
    up = stock_prices[stock_prices.close >= stock_prices.open]
    c_up = stock_prices[stock_prices[trend_col] == 1]

    # "down" dataframe will store the stock_prices
    # when the closing stock price is
    # lesser than the opening stock prices
    down = stock_prices[stock_prices.close < stock_prices.open]
    c_down = stock_prices[stock_prices[trend_col] == -1]

    # When the stock prices have decreased, then it
    # will be represented by blue color candlestick
    col1 = 'green'

    # When the stock prices have increased, then it
    # will be represented by green color candlestick
    col2 = 'red'

    # Setting width of candlestick elements
    width = .8
    width2 = .08
    # Plotting up prices of the stock
    artists = []
    # fig, ax = plt.subplots()
    bar1 = ax.bar(up.index, up.close - up.open, width, bottom=up.open, color=col1)
    artists.append(bar1)
    bar2 = ax.bar(up.index, up.high - up.close, width2, bottom=up.close, color=col1)
    artists.append(bar2)
    bar3 = ax.bar(up.index, up.low - up.open, width2, bottom=up.open, color=col1)
    artists.append(bar3)
    s1 = ax.scatter(c_up.index, c_up.high, marker='^', color=col1, s=15)
    artists.append(s1)
    # Plotting down prices of the stock
    bar4 = ax.bar(down.index, down.close - down.open, width, bottom=down.open, color=col2)
    artists.append(bar4)
    bar5 = ax.bar(down.index, down.high - down.open, width2, bottom=down.open, color=col2)
    artists.append(bar5)
    bar6 = ax.bar(down.index, down.low - down.close, width2, bottom=down.close, color=col2)
    artists.append(bar6)
    s2 = ax.scatter(c_down.index, c_down.high, marker='v', color=col2, s=15)
    artists.append(s2)
    # plt.plot(stock_prices.index,stock_prices['peak_price'],'.r-')
    if trend_line_col != '':
        l1 = ax.plot(stock_prices.index, stock_prices[trend_line_col], 'r')
        artists.append(l1)
    # rotating the x-axis tick labels at 30degree

    # out_dic = {'artist_list':artists}

    return artists


def add_candle_trend(data,trend_sign_col = 'trend_sign'):
    prior_high = data['high'].shift(1)
    prior_low = data['low'].shift(1)
    prior_open = data['open'].shift(1)
    prior_close = data['close'].shift(1)

    data = data.assign(p_High = prior_high
                       ,p_Low = prior_low
                       ,p_Open = prior_open
                       ,p_Close = prior_close
                       ,High_inc = np.sign(data['high']-prior_high)
                       ,Low_dec = np.sign(-data['low']+prior_low))

    cur_trend = 0
    trends = []


    for idx, row in data.iterrows():
        if cur_trend== 0:
            if row['High_inc'] == 1 and row['Low_dec'] == -1:
                cur_trend = 1
            elif row['High_inc'] == -1 and row['Low_dec'] == 1:
                cur_trend = -1
        else:
            if cur_trend == 1:
                if row['High_inc']>=0 or row['Low_dec'] <= 0:
                    cur_trend = 1

                else:
                    cur_trend = -1

            else:
                if row['High_inc']<=0 or row['Low_dec'] >= 0:
                    cur_trend = -1

                else:
                    cur_trend = 1


        trends.append(cur_trend)


    data[trend_sign_col] =  trends

    return data

def candle_trend_all(data,resize_para = {'type_t':'continous','length':3}):
    # data = m_data0
    if len(resize_para)==0:
        data_out = add_candle_trend(data,trend_sign_col='trend_sign1')
    else:
        type_t = resize_para['type_t']
        length = resize_para['length']
        data_rz,data_1 = agg_ohlc(data, type_t=type_t, length=length, group_col='rs_col')
        trend_col = 'trend_sign'+str(length)+'rs'
        data_rz = add_candle_trend(data_rz,trend_sign_col=trend_col)
        data_rz = mark_last(data_rz, trend_sign_col=trend_col)
        data_rz = data_rz.rename(columns= {'high':'rs_high','low':'rs_low'})

        data_t= data_1.merge(data_rz[[trend_col,trend_col+'last','rs_high','rs_low','rs_col']],how = 'left',on = 'rs_col')
        trends = []
        m_hit = 0
        prior_trend = -3
        for idx, row in data_t.iterrows():
            if row[trend_col+'last'] != 1:
                trend = row[trend_col]
                m_hit = 0
            else:
                if prior_trend != row[trend_col]:
                    m_hit = 0
                prior_trend = row[trend_col]
                if m_hit ==0:
                    trend = row[trend_col]
                else:
                    trend = -row[trend_col]
                if (row[trend_col] == 1 and row['rs_high'] == row['high']) or (
                        row[trend_col] == -1 and row['rs_low'] == row['low']):
                    m_hit = 1

            trends.append(trend)
        t_args = {}
        t_args['trend_sign'+str(length)] = trends
        data_out = data_t.assign(**t_args)
    return data_out

def remove_1_day_trend(data,trend_sign_col = 'trend_sign'):
    #retired function
    prior_trend = data[trend_sign_col].shift(1)
    next_trend = data[trend_sign_col].shift(-1)
    data = data.assign(prior_trend =prior_trend ,
                       next_trend =next_trend )
    OnePlus_trends = []
    for idx, row in data.iterrows():
        if (row['prior_trend'] == row['next_trend']) and (row['prior_trend'] != row[trend_sign_col]):
            oneplus = row['prior_trend']
        else:
            oneplus = row[trend_sign_col]
        OnePlus_trends.append(oneplus)
    data = data.assign(OnePlus_trends = OnePlus_trends)
    return data
def mark_last(data,trend_sign_col = 'trend_sign'):
    #data is ordered in time
    next_trend = data[trend_sign_col].shift(-1)
    args = {}
    args[trend_sign_col+'last'] = (data[trend_sign_col] != next_trend).map(int)
    data = data.assign(**args)
    return data
def merge_last(m_data,s_data):
    s_data = s_data.merge(m_data[['last_trend','trend_sign']],left_index = True,right_index = True).fillna(0)
    return s_data
def cal_bench_move(data,last_trend_col = 'last_trend',trend_col = 'trend_sign',high_col= 'high',low_col= 'low'):
    move_df = data[data[last_trend_col]==1]
    move_df = move_df.assign(peak_price = move_df.apply(lambda row: row[high_col] if row[trend_col] == 1 else row[low_col],axis = 1))
    p_peak = move_df['peak_price'].shift(1)
    p_dlt = move_df['peak_price'] - p_peak
    prior_p_dlt = p_dlt.shift(1)
    move_df = move_df.assign(p_peak = p_peak,
                             p_dlt = p_dlt,
                             prior_p_dlt = prior_p_dlt,
                             rebounce_rate = p_dlt/prior_p_dlt*(-move_df[trend_col])
                             )
    return move_df
def get_bench(move_df,start_index = -2, end_index = -1):
    dlt_df = move_df[start_index:end_index]
    bench = dlt_df['p_dlt'].sum()
    return bench
def get_raw(symbol,start_time, end_time):
    data = yf.download(symbol, start_time, end_time)
    data.columns = [v.lower() for v in data.columns]

    data['date'] = data.index
    data['x'] =[i for i in range(data.shape[0])]
    return data

def daily_rebounce(data,move_df,trend_col = 'trend_sign',high_col= 'high',low_col= 'low'):
    data_mv = data.merge(move_df[['p_dlt','prior_p_dlt','p_peak','rebounce_rate']],how = 'left',left_index = True, right_index = True)
    data_mv = data_mv.assign(
                             prior_p_dlt = data_mv['prior_p_dlt'].fillna(method='bfill')
                             ,p_high = data_mv[high_col].shift(1)
                             ,p_low = data_mv[low_col].shift(1))
    dp_dlt = data_mv.apply(
        lambda row: (row[high_col] - row['p_high']) if row[trend_col] == 1 else (row[low_col] - row['p_low']), axis=1)
    data_mv = data_mv.assign(dp_dlt = dp_dlt
                             ,day_rebouce_rate = dp_dlt/data_mv['prior_p_dlt']*(-data_mv[trend_col]))
    return data_mv


def relative_strength(m_data0, s_data0):
    m_data = m_data0.copy()

    m_data = add_candle_trend(m_data)
    m_data = mark_last(m_data, trend_sign_col='trend_sign')
    mv_df = cal_bench_move(m_data, last_trend_col='last_trend', trend_col='trend_sign', high_col='high', low_col='low')
    m_data_d = daily_rebounce(m_data, mv_df, trend_col='trend_sign', high_col='high', low_col='low')
    m_data_d.columns = ['m_' + v for v in m_data_d]

    s_data = s_data0.copy()
    s_data = merge_last(m_data, s_data)
    sv_df = cal_bench_move(s_data,last_trend_col='last_trend', trend_col='trend_sign', high_col='high', low_col='low')
    s_data_d = daily_rebounce(s_data, sv_df, trend_col='trend_sign', high_col='high', low_col='low')


    s_data_d.columns = ['s_' + v for v in s_data_d]
    # m_bench = get_bench(mv_df,start_index = -4, end_index = -1)
    # s_bench = get_bench(sv_df,start_index = -4, end_index = -1)
    # mv_df = mv_df.assign(m_rel_dlt = mv_df['p_dlt']/m_bench)
    # sv_df = sv_df.assign(s_rel_dlt=sv_df['p_dlt'] / s_bench)
    cmp_s = m_data_d.merge(s_data_d,left_index = True,right_index = True)
    return cmp_s, m_data,s_data

def trend_allinone(symbol, start_time, end_time):
    data = yf.download(symbol, start_time, end_time)
    data.columns = [v.lower() for v in data.columns]
    data = add_candle_trend(data)
    data = remove_1_day_trend(data)
    plot_candle(data, trend_col='trend_sign')

def agg_ohlc(data,type_t='continous',length = 2,group_col = 'rs_col'):
    data_1 = gp_col_gen(data, type_t=type_t,length = length, group_col = group_col)
    ohlc_dic = {'date': 'max', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                'volume': 'sum'}
    data_rz = data_1.groupby(by = group_col,as_index =False).agg(ohlc_dic)

    return data_rz,data_1
def gp_col_gen(data, type_t='continous',length = 2, group_col = 'rs_col'):
    if type_t == 'continous':
        data[group_col] = [int(i /length) for i in range(data.shape[0])]
    if type_t == 'week':
        pass
    return data


def trend_line(df1, col='low'):
    df = df1.copy()
    while df.shape[0] > max(4,df.shape[0]/10):
        x = df.x
        y = df[col]

        p = np.poly1d(np.polyfit(x, y, 1))
        y1 = p(x)
        df = df.assign(trend_line=y1)
        if col == 'low':
            df = df[df['low'] <= df['trend_line']]
        else:
            df = df[df['high'] >= df['trend_line']]
        print(df.shape)
    x = df1.x
    y_hat = p(x)
    df1 = df1.assign(trend_line=y_hat)
    return p, df1

def trend_linew(df1, col='low',decay = 0.2):
    df = df1.copy()
    len_t = df.shape[0]
    w = np.array([np.exp(-(len_t - i-1)*decay) for i in range(len_t)])
    df = df.assign(w = w)
    while df.shape[0] > max(4,df.shape[0]/10):
        x = df.x
        y = df[col]
        w0 =df['w']
        p = np.poly1d(np.polyfit(x, y, 1,w=w0))
        y1 = p(x)
        df = df.assign(trend_line=y1)
        if col == 'low':
            df = df[df['low'] <= df['trend_line']]
        else:
            df = df[df['high'] >= df['trend_line']]
        print(df.shape)
    x = df1.x
    y_hat = p(x)
    df1 = df1.assign(trend_line=y_hat)
    return p, df1
