import datetime
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import yfinance as yf
import sys
sys.path.append('/home/wang/Documents/NewIdeas/')
from DT_Tools import *


import tkinter as tk


# import matplotlib.animation as animation
#
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk, FigureCanvasTkAgg


start_time = '2023-01-26'
end_time = '2023-10-17'
m_data0= get_raw('SPY', start_time, end_time)
s_data0 = get_raw('AAPL', start_time, end_time)
m_data0[['high']].plot()
plt.show()
m_data = m_data0.copy()
m_data = candle_trend_all(m_data,resize_para = {'type_t':'continous','length':3})
m_data = mark_last(m_data, trend_sign_col='trend_sign3')
mv_df = cal_bench_move(m_data, last_trend_col='trend_sign3last', trend_col='trend_sign3', high_col='high', low_col='low')
m_data_d = daily_rebounce(m_data,mv_df,trend_col = 'trend_sign3',high_col= 'high',low_col= 'low')
i = 0


import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np

class Application(tk.Frame):
    def __init__(self, master=None):
        self.symbol_var = tk.StringVar()
        self.months_var = tk.StringVar()
        self.size_var = tk.StringVar()
        self.clen_var = tk.StringVar()
        self.pdone_var = tk.StringVar()
        self.sidx_var = tk.StringVar()
        self.stepd_var  = tk.StringVar()

        tk.Frame.__init__(self,master)
        self.createWidgets()
        # root = master
        self.dlt_idx = 0
        # self.data = data.copy()

    def createWidgets(self):

        fig=plt.figure(figsize=(8,8))
        ax=fig.add_subplot(111)
        canvas=FigureCanvasTkAgg(fig,master=root)
        canvas.get_tk_widget().grid(row=0,column=3,columnspan = 3, rowspan = 16)
        canvas.draw()
        # raw load------------------------------------------------
        self.loadbutton = tk.Button(master=root, text="load", command=self.load)
        self.loadbutton.grid(row=0, column=0,rowspan = 3,sticky='N')

        self.symbol_label = tk.Label(master=root,text = "Symbol")
        self.symbol_label.grid(row=0, column=1)
        self.symbol_entry = tk.Entry(master=root, textvariable = self.symbol_var)
        self.symbol_entry.grid(row=0, column=2)

        self.month_label = tk.Label(master=root, text="Months to load")
        self.month_label.grid(row=1, column=1)
        self.month_entry = tk.Entry(master=root, textvariable = self.months_var)
        self.month_entry.grid(row=1, column=2)

        self.data_label = tk.Label(master=root, text="Loaded")
        self.data_label.grid(row=2, column=1)
        self.size_entry = tk.Entry(master=root, textvariable=self.size_var)
        self.size_entry.grid(row=2, column=2)
        # base process ------------------------------------------------
        self.processbutton = tk.Button(master=root, text="process", command=self.process0)
        self.processbutton.grid(row=3, column=0, rowspan=2,sticky='N')

        self.clen_label = tk.Label(master=root, text="candle length")
        self.clen_label.grid(row=3, column=1)
        self.clen_entry = tk.Entry(master=root, textvariable=self.clen_var)
        self.clen_entry.grid(row=3, column=2)

        self.done_label = tk.Label(master=root, text="Done")
        self.done_label.grid(row=4, column=1)
        self.done_entry = tk.Entry(master=root, textvariable=self.pdone_var)
        self.done_entry.grid(row=4, column=2)
        # plot ------------------------------------------------
        self.plotbutton = tk.Button(master=root, text="plot", command=lambda: self.plot(canvas, ax))
        self.plotbutton.grid(row=5, column=0, rowspan=2,sticky='N')

        self.sidx_label = tk.Label(master=root, text="start index")
        self.sidx_label.grid(row=5, column=1)
        self.sidx_entry = tk.Entry(master=root, textvariable=self.sidx_var)
        self.sidx_entry.grid(row=5, column=2)

        self.stepd_label = tk.Label(master=root, text="stepDelt")
        self.stepd_label.grid(row=6, column=1)
        self.stepd_entry = tk.Entry(master=root, textvariable=self.stepd_var)
        self.stepd_entry.grid(row=6, column=2)



    def load(self):
        #get the symbol
        self.symbol = self.symbol_var.get()
        months = int(self.months_var.get())
        nt = datetime.datetime.now()
        start_time = (nt - datetime.timedelta(days= months*30)).strftime('%Y-%m-%d')
        end_time = nt.strftime('%Y-%m-%d')
        self.data = get_raw(self.symbol, start_time, end_time)

        print(self.data.shape)
        self.size_var.set(self.data.shape[0])

    def process0(self):
        print('get from clen:', self.clen_var.get(), "--")
        self.clen = int(self.clen_var.get())
        self.trend_sign_col = 'trend_sign' + str(self.clen)

        self.sidx = int(self.sidx_var.get())
        self.stepd = int(self.stepd_var.get())
        self.pdone_var.set('Done')


    def plot(self,canvas,ax):
        # c = ['r','b','g']  # plot marker colors
        ax.clear()         # clear axes from previous plot
        base_dt = self.data[:(self.sidx +self.dlt_idx)]
        print('get data:',base_dt.shape)
        #---------------process data-------------
        base_dt = candle_trend_all(base_dt, resize_para={'type_t': 'continous', 'length': self.clen})
        print('candle trend')
        self.base_dt = mark_last(base_dt, trend_sign_col=self.trend_sign_col)
        print('last marked')

        self.mv_df = cal_bench_move(self.base_dt, last_trend_col=self.trend_sign_col+'last', trend_col=self.trend_sign_col, high_col='high',
                               low_col='low')
        print('trend_move:',self.mv_df.shape)
        #ignore the last trend, as there always be a point at the last trend it does not mean the trend end
        #use -2,-3,-4,-5 to determine prior trend

        if self.mv_df.iloc[-2]['peak_price']>self.mv_df.iloc[-4]['peak_price']:
            last_trend_sign = 1
        elif self.mv_df.iloc[-2]['peak_price']<self.mv_df.iloc[-4]['peak_price']:
            last_trend_sign = -1
        else:
            if self.mv_df.iloc[-3]['peak_price'] > self.mv_df.iloc[-5]['peak_price']:
                last_trend_sign = 1
            elif self.mv_df.iloc[-3]['peak_price'] < self.mv_df.iloc[-5]['peak_price']:
                last_trend_sign = -1
            else:
                last_trend_sign = 0

        if last_trend_sign != 0:
            if last_trend_sign == 1:
                t_col = 'low'
            else:
                t_col = 'high'
            print('t_col ', t_col)
            self.mv_df_tr = self.mv_df.iloc[:-1][self.mv_df[self.trend_sign_col]==-last_trend_sign]
            p0 = self.mv_df_tr.iloc[-1]['peak_price']
            dt0 =self.mv_df_tr.iloc[-1]['date']
            # last_trend_sign = -1
            for idx,row in self.mv_df_tr.iloc[:-1][::-1].iterrows():
                if row['peak_price']*last_trend_sign<p0*last_trend_sign:
                    p0 = row['peak_price']
                    dt0 = row['date']
                else:
                    break

            trend_start_date = dt0
            trend_end_date = self.mv_df.iloc[-2]['date']
            print('last trend',last_trend_sign,'start',trend_start_date,'end',trend_end_date)

            trend_df = base_dt[(base_dt['date']>=trend_start_date)&(base_dt['date']<=trend_end_date)]

            print('trend_df size',trend_df.shape)
            if trend_df.shape[0]>5:

                p, _= trend_line(trend_df, col=t_col)
                def trend_line_predict(row,p):
                    if row['date']<trend_start_date:
                        out = np.nan
                    elif row['date'] <( trend_end_date + datetime.timedelta(days = 15)):
                        out = p(row['x'])
                    else:
                        out = np.nan
                    return out
                base_dt = base_dt.assign(last_trend_prime = base_dt.apply(lambda  row:trend_line_predict(row,p),axis = 1))
            else:
                print('too little point')
                base_dt = base_dt.assign(last_trend_prime=np.nan)
        else:
            base_dt = base_dt.assign(last_trend_prime=np.nan)

        #----------------------------------------
        _ = plot_candle(base_dt, ax, trend_col=self.trend_sign_col, trend_line_col='last_trend_prime')
        self.dlt_idx+=self.stepd
        # for i in range(3):
        #     theta = np.random.uniform(0,360,10)
        #     r = np.random.uniform(0,1,10)
        #     ax.plot(theta,r,linestyle="None",marker='o', color=c[i])
        canvas.draw()

root=tk.Tk()
self=Application(master=root)
self.mainloop()

cmp,m_data,s_data = relative_strength(m_data0, s_data0)

def trend_back(mv_df,peak_col = 'peak_price',trend_col = 'trend_sign3'):

    max_df = mv_df[mv_df[trend_col]==1]
    min_df = mv_df[mv_df[trend_col]==-1]
    n_max = max_df[peak_col].shift(-1)
    n_min = min_df[peak_col].shift(-1)
    max_trend = np.sign(n_max-max_df[peak_col])
    min_trend = np.sign(n_min-min_df[peak_col])
    max_df = max_df.assign(max_trend = max_trend
                           ,max_tdate = max_df['date'])

    min_df = min_df.assign(min_trend=min_trend
                           ,min_tdate = min_df['date'])
    mv_df0 = mv_df.merge(max_df[['date','max_trend','max_tdate']],how = 'left',on = 'date')\
        .merge(min_df[['date','min_trend','min_tdate']],how= 'left',on = 'date')
    mv_df0 = mv_df0.assign(max_trend = mv_df0['max_trend'].fillna(method='ffill')
                           ,min_trend = mv_df0['min_trend'].fillna(method='ffill'))
    mv_df0 = mv_df0.assign(max_trend_p = mv_df0['max_trend'].shift(1)
                           ,min_trend_p  =mv_df0['min_trend'].shift(1))
    mv_df1 = mv_df0[((mv_df0['max_trend_p']==1)&(mv_df0['max_trend']==-1))|((mv_df0['min_trend_p']==-1)&(mv_df0['min_trend']==1))]
    return mv_df1









