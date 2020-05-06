# %%
#%%
import os
import numpy
import numpy as np
import csv
from MetaTrader5 import *
import pandas as pd
import warnings
from numpy import *
from pandas._libs.index import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#number of time series for getmetadata
count = 20

def estimate_coef(x,y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    return(b_0, b_1)

    # number of observations/points
    x=Calculate_X(y)
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    return(b_1)

def slope(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    return m


def Calculate_X(Y) :
    import numpy as np
    t=len(Y)
    a = np.empty(t)
    b = np.arange(1, t+1, 1)
    ind = np.arange(len(a))
    np.put(a, ind, b)
    return (a)
def regression_line(x,b):

   
    y_pred = b[0] + b[1]*x

    return y_pred
def Regline(Array) :
   C=np.ravel(Array)
   X=Calculate_X(C).astype(np.int64)
   B=estimate_coef(X,C)
   R=regression_line(X,B)
   return numpy.array(np.ravel(R),dtype=float)
def GetMetaData(MetaSymbol,size=count) :   
    MT5Initialize()
    #MT5WaitForTerminal()
    rates=MT5CopyRatesFromPos(MetaSymbol, MT5_TIMEFRAME_M15, 0, size)
    MT5Shutdown()
    close =numpy.array( [y.close for y in rates],dtype=int)
    close=close.reshape(-1,1)
    openn =numpy.array( [y.open for y in rates],dtype=int)
    openn=openn.reshape(-1,1)
    high =numpy.array( [y.high for y in rates],dtype=int)
    high=high.reshape(-1,1)
    low =numpy.array( [y.low for y in rates],dtype=int)
    low=low.reshape(-1,1)
    Vol =numpy.array( [y.real_volume for y in rates])
    Vol=Vol.reshape(-1,1)
    tick =numpy.array( [y.tick_volume for y in rates])
    tick=tick.reshape(-1,1)
    spread =numpy.array( [y.spread for y in rates])
    spread=spread.reshape(-1,1)
        
    pure_array=(numpy.concatenate((openn,high,low,close,Vol,tick,spread), axis=1))
    return pure_array#[::-1]
 
def TimeSeries(array):

    return array[::-1]
def PolyReg(Array,degree=2):
    C=np.ravel(Array)
    X=Calculate_X(C).astype(np.int64)
    R=polyfit(X,C,degree)
    return numpy.array(np.ravel(R),dtype=float)
def polyfit(x, y, degree):
    results = {}

    coeffs = numpy.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = numpy.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)
    Finale=yhat[::-1]                         # or [p(z) for z in x]
    ybar = numpy.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = numpy.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = numpy.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return (Finale)

Symbols=[
            'IRO1GOLG0001',
            'IRO1MADN0001',
            'IRO1MSMI0001',
            'IRO1CHML0001',
            'IRO1KNRZ0001',
            'IRO1FOLD0001',
            'IRO1PARS0001',
            'IRO1PNES0001',
            'IRO3ZOBZ0001',
            'IRO1BMLT0001',
            'IRO1PNBA0001',
            'IRO1BTEJ0001',
            'IRO1PTAP0001',
            'IRO1PTEH0001',
            'IRO3PZGZ0001',
            'IRO1PNTB0001',
            'IRO1PARK0001',
            'IRO1PKLJ0001',
            'IRO1GDIR0001',
            'IRO1PRDZ0001',
            'IRO1KVEH0001',
            'IRO1PASN0001',
            'IRO1SAND0001',
            'IRO3JPRZ0001',
            'IRO1FKHZ0001',
            'IRO1MAPN0001',
            'IRO1MOBN0001',
            'IRO1IKHR0001',
            'IRO3IMFZ0001',
            'IRO1AYEG0001',
            'IRO1SIPA0001',
            'IRO1ALBZ0001',
            'IRO1DALZ0001',
            'IRO1HWEB0001',
            'IRO1SEPP0001',
            'IRO1OIMC0001',
            'IRO1BSDR0001',
            'IRO1KSIM0001',
            'IRO1PJMZ0001',
            'IRO1BANK0001',
            'IRO3PNLZ0001',
            'IRO1MARK0001',
            'IRO3KHMZ0001',
            'IRO3PGHZ0001',
            'IRO1BFJR0001',
            'IRO1RSAP0001',
            'IRO1IKCO0001',
            'IRO1ZMYD0001',
            'IRO1PKOD0001',
            'IRO1IPTR0001',
            'IRO1SSAP0001',
            'IRO1TRNS0001',
            'IRO1KSHJ0001',
            'IRO1BAHN0001',
            'IRO3FOHZ0001',
            'IRO1PSHZ0001',
            'IRO1PKHA0001',
            'IRO7VHYP0001',
            'IRO3ARFZ0001',
            'IRO3PRZZ0001',
            'IRO3BDYZ0001',
            'IRO1SEPK0001',
            'IRO3APDZ0001',
            'IRO1APPE0001',
            'IRO1MKBT0001',
            'IRO1HMRZ0001',
            'IRO1ARDK0001',
            'IRO1SGAZ0001',
            'IRO1SISH0001',
            'IRO1TSRZ0001',
            'IRO1FRVR0001',
            'IRO1SORB0001',
            'IRO1AMLH0001',
            'IRO1CRBN0001',
            'IRO1NKOL0001',
            'IRO1GTSH0001',
            'IRO1PKER0001',
            'IRO1PASH0001',
            'IRO1SHOY0001',
            'IRO1SHND0001',
            'IRO1TAIR0001',
            'IRO1SNMA0001',
            'IRO1BALI0001',
            'IRO1COMB0001',
            'IRO1BHMN0001',
            'IRO1RENA0001',
            'IRO1AZAB0001',
            'IRO1ROOI0001',
            'IRO1VSIN0001',
            'IRO1GGAZ0001',
            'IRO1GSBE0001',
            'IRO1IAGM0001',
            'IRO1DADE0001',
            'IRO1RKSH0001',
            'IRO1TBAS0001',
            'IRO3MPRZ0001',
            'IRO3OSHZ0001',
            'IRO3HORZ0001',
            'IRO1BPST0001',
            'IRO1NAFT0001',
            'IRO1HSHM0001',
            'IRO1KSKA0001',
            'IRO3TIGZ0001',
            'IRO7KKHP0001'

            ]
SymbolDict={
    'IRX6XTPI0006':"شاخص کل6",
    'IRX6XTPI0026':"شاخص کل (هم وزن)6",
    'IRXYXTPI0026':"شاخص قیمت (هم وزن)6",
    'IRXZXOCI0006':"شاخص کل فرابورس6",
    'IRX6XWAI0006':"شاخص قیمت 50 شرکت6",
    'IRO1GOLG0001':"کگل",
    'IRO1MADN0001':"ومعادن",
    'IRO1MSMI0001':"فملی",
    'IRO1CHML0001':"کچاد",
    'IRO1KNRZ0001':"کنور",
    'IRO1FOLD0001':"فولاد",
    'IRO1PARS0001':"پارس",
    'IRO1PNES0001':"شپنا",
    'IRO3ZOBZ0001':"ذوب",
    'IRO1BMLT0001':"وبملت",
    'IRO1PNBA0001':"شبندر",
    'IRO1BTEJ0001':"وتجارت",
    'IRO1PTAP0001':"تاپیکو",
    'IRO1PTEH0001':"شتران",
    'IRO3PZGZ0001':"زاگرس",
    'IRO1PNTB0001':"شبریز",
    'IRO1PARK0001':"شاراک",
    'IRO1PKLJ0001':"فارس",
    'IRO1GDIR0001':"وغدیر",
    'IRO1PRDZ0001':"شپدیس",
    'IRO1KVEH0001':"کاوه",
    'IRO1PASN0001':"پارسان",
    'IRO1CHML0001':"کچاد",
    'IRO1SAND0001':"وصندوق",
    'IRO3JPRZ0001':"چکاپا",
    'IRO1FKHZ0001':"فخوز",
    'IRO1MAPN0001':"رمپنا",
    'IRO1MOBN0001':"مبین",
    'IRO1IKHR0001':"وخارزم",
    'IRO3IMFZ0001':"سمگا",
    'IRO1AYEG0001':"پردیس",
    'IRO1SIPA0001':"خساپا",
    'IRO1ALBZ0001':"والبر",
    'IRO1DALZ0001':"دالبر",
    'IRO1HWEB0001':"های وب",
    'IRO1SEPP0001':"شسپا",
    'IRO1OIMC0001':"وامید",
    'IRO1BSDR0001':"وبصادر",
    'IRO1KSIM0001':"فاسمین",
    'IRO1PJMZ0001':"جم",
    'IRO1BANK0001':"وبانک",
    'IRO3PNLZ0001':"شاوان",
    'IRO1MARK0001':"فاراک",
    'IRO3KHMZ0001':"میدکو",
    'IRO3PGHZ0001':"شغدیر",
    'IRO1BFJR0001':"بفجر",
    'IRO1RSAP0001':"ولساپا",
    'IRO1IKCO0001':"خودرو",
    'IRO1ZMYD0001':"خزامیا",
    'IRO1PKOD0001':"خپارس",
    'IRO1IPTR0001':"پترول",
    'IRO1SSAP0001':"وساپا",
    'IRO1TRNS0001':"بترانس",
    'IRO1KSHJ0001':"حکشتی",
    'IRO1BAHN0001':"فباهنر",
    'IRO3FOHZ0001':"هرمز",
    'IRO1PSHZ0001':"شیراز",
    'IRO1PKHA0001':"شخارک",
    'IRO7VHYP0001':"واحیا",
    'IRO3ARFZ0001':"ارفع",
    'IRO3PRZZ0001':"شراز",
    'IRO3BDYZ0001':"دی",
    'IRO1SEPK0001':"سپ",
    'IRO3APDZ0001':"اپرداز",
    'IRO1APPE0001':"آپ",
    'IRO1MKBT0001':"اخابر",
    'IRO1HMRZ0001':"همراه",
    'IRO1ARDK0001':"کسرا",
    'IRO1SGAZ0001':"کگاز",
    'IRO1SISH0001':"کساپا",
    'IRO1TSRZ0001':"کرازی",
    'IRO1FRVR0001':"فرآور",
    'IRO1SORB0001':"فسرب",
    'IRO1AMLH0001':"شاملا",
    'IRO1CRBN0001':"شکربن",
    'IRO1NKOL0001':"شکلر",
    'IRO1GTSH0001':"شگل",
    'IRO1PKER0001':"کرماشا",
    'IRO1PASH0001':"پاکشو",
    'IRO1SHOY0001':"شوینده",
    'IRO1SHND0001':"پسهند",
    'IRO1TAIR0001':"پتایر",
    'IRO1SNMA0001':"وصنعت",
    'IRO1BALI0001':"وبوعلی",
    'IRO1COMB0001':"تکمبا",
    'IRO1BHMN0001':"خبهمن",
    'IRO1RENA0001':"ورنا",
    'IRO1AZAB0001':"فاذر",
    'IRO1ROOI0001':"کروی",
    'IRO1VSIN0001':"وسینا",
    'IRO1GGAZ0001':"قزوین",
    'IRO1GSBE0001':"قثابت",
    'IRO1IAGM0001':"مرقام",
    'IRO1DADE0001':"مداران",
    'IRO1RKSH0001':"رکیش",
    'IRO1TBAS0001':"کطبس",
    'IRO3MPRZ0001':"ثپردیس",
    'IRO3OSHZ0001':"ثعمرا",
    'IRO3HORZ0001':"وهور",
    'IRO1BPST0001':"وپست",
    'IRO1NAFT0001':"ونفت",
    'IRO1HSHM0001':"حفاری",
    'IRO1KSKA0001':"چکاوه",
    'IRO3TIGZ0001':"تبرک",
    'IRO3AVLZ0001':"داوه",
    'IRO7KKHP0001':"خکرمان",
    'IRO1TAYD0001':"حتاید",
    'IRO1DTIP0001':"تیپیکو",
    'IRO7KMOP0001':"لکما",
    'IRO3KRMZ0001':"کرمان",
    'IRO3KSGZ0001':"کشرق",
    'IRO1DSBH0001':"دسبحان",
    'IRO1MHKM0001':"خمهر",
    'IRO1TGOS0001':"وتوس",
    'IRO7ARAP0001':"حاریا",
 }



#شاخص کل
WI=numpy.array((GetMetaData(SymbolDict[ 'IRX6XTPI0006'])),dtype=int) 

#close_price
WIC=WI[:,[3]].astype(np.float64)

#data_frame
dfWIC=pd.DataFrame(WIC[:,0])

#normalize
df_norm_dfWIC = (dfWIC - dfWIC.mean())/dfWIC.std()
print(df_norm_dfWIC)

AI=numpy.array((GetMetaData(SymbolDict[ 'IRO1PNES0001'])),dtype=int) 

#close_price
AIC=AI[:,[3]].astype(np.float64)

#data_frame
dfAIC=pd.DataFrame(AI[:,0])

#normalize
df_norm_dfAIC = (dfAIC - dfAIC.mean())/dfAIC.std()
print(df_norm_dfAIC)

# ta inja kodiye ke az ghabl dashtam hala mikham bakhshi ke in paien hast ba in data kar kone




# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df = pd.read_csv("synchrony_sample.csv")
print(df)
print(type(df))

overall_pearson_r = df.corr().iloc[0,1]
print(f"Pandas computed Pearson r: {overall_pearson_r}")
# out: Pandas computed Pearson r: 0.2058774513561943

r, p = stats.pearsonr(df.dropna()['S1_Joy'], df.dropna()['S2_Joy'])  # be jaye S1_Joy mikham  df_norm_dfAIC bashe , 
print(f"Scipy computed Pearson r: {r} and p-value: {p}")
# out: Scipy computed Pearson r: 0.20587745135619354 and p-value: 3.7902989479463397e-51

# Compute rolling window synchrony
f,ax=plt.subplots(figsize=(7,3))
df.rolling(window=30,center=True).median().plot(ax=ax)
ax.set(xlabel='Time',ylabel='Pearson r')
ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}");

# %%

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

d1 = df['S1_Joy']
d2 = df['S2_Joy']
seconds = 5
fps = 30
rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
offset = np.ceil(len(rs)/2)-np.argmax(rs)
f,ax=plt.subplots(figsize=(14,3))
ax.plot(rs)
ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads',ylim=[.1,.31],xlim=[0,301], xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
plt.legend()

# %%
