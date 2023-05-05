# %%
import pandas as pd
import numpy as np
import holidays
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# %% [markdown]
# ### Dataset

# %%
# loading raw data，移除airport相關欄位(目前用不到)
usecol = ['year_actu_depa', 'month_actu_depa', 'day_actu_depa',
          'hr_actu_depa', 'min_actu_depa', 'delay_dest']
flight = pd.read_csv('./US_1722_flights_info_IQR.csv', usecols=usecol)

''' Week '''
# 將年月日欄位合併成一個columns並且設為datetime格式
flight["date"] = pd.to_datetime(flight["year_actu_depa"].astype(str) + "/" +
                                flight["month_actu_depa"].astype(str) + "/" +
                                flight["day_actu_depa"].astype(str))

# 移除2023年
flight = flight[flight["year_actu_depa"] != 2023]

# 將日期轉換成星期，星期表達方式是0-6，0是星期一，6是星期日
flight["week"] = flight["date"].dt.weekday

# 定義week：0-4為平日，5,6為假日，切分區間
# 因pd.cut()，左邊值不包含故設為-1, 右邊值包含故為6
bins = [-1, 4, 6]
labels = ['on_weekday', 'weekend']
flight['week_classify'] = pd.cut(flight['week'], bins=bins, labels=labels)

''' holidays '''
# 取得2017-2022 US holiday date跟holiday names
holiday_list = []
for holiday in holidays.US(years=[2017, 2018, 2019, 2020, 2021, 2022]).items():
    holiday_list.append(holiday)

holidays_df = pd.DataFrame(holiday_list, columns=["date", "holiday"])

# 選擇重大節日，有連假的節日，有八個
eight_holidays = ["New Year's Day", "Martin Luther King Jr. Day", "Washington's Birthday",
                  "Memorial Day", "Independence Day", "Labor Day", "Thanksgiving", "Christmas Day"]

# holiday是八個重大節日才要
holiday_df = holidays_df[holidays_df['holiday'].isin(eight_holidays)]

# 轉成date欄位轉成datetime格式
holiday_df['date'] = pd.to_datetime(holiday_df['date'])

# delay_rate_by_holiday merge holiday_df
flight = flight.merge(holiday_df, left_on="date", right_on="date", how="left")

# 檢查是否成功merge
# delay_rate_by_holiday[delay_rate_by_holiday['holiday'].notna()]

# 判斷delay_rate_by_holiday的date欄位跟holiday_df是否符合
flight['is_holidays'] = flight['date'].isin(holiday_df['date'])

''' Time'''
# 定義小時切分區間
# 因pd.cut()，左邊值不包含故設為-1, 右邊值包含故為23
bins = [-1, 6, 12, 18, 23]
labels = ['early morning', 'morning', 'afternoon', 'night']
flight['time_period'] = pd.cut(
    flight['hr_actu_depa'], bins=bins, labels=labels)

''' 疫情前後 '''
year_bin = [2016, 2019, 2022]
year_labels = ['before_epidemic', 'after_epidemic']
flight['year_epidemic'] = pd.cut(
    flight['year_actu_depa'], bins=year_bin, labels=year_labels)


''' 篩選畫圖所需欄位'''

# 圖一
delay_rate_by_year_month = flight[[
    "year_actu_depa", "month_actu_depa", "delay_dest"]]
# 圖二
delay_rate_by_year = flight[["year_actu_depa", "delay_dest", "year_epidemic"]]
# 圖三
delay_rate_by_holiday = flight[[
    "date", "holiday", "delay_dest", "year_epidemic"]]
# 圖四
delay_rate_by_hours = flight[["hr_actu_depa", "delay_dest"]]

# %% [markdown]
# ### Plot - 1

# %%


def flight_year_month(dataset, delay_time):
    ''' Data '''
    flight_count = pd.DataFrame()
    # by time_period 總航班
    flight_count['year_month_count'] = dataset.groupby(
        ['year_actu_depa', 'month_actu_depa']).count()[['delay_dest']]

    # by time_period 有誤點的航班
    flight_count['year_month_delay_count'] = dataset[dataset['delay_dest'] > delay_time].groupby(
        ['year_actu_depa', 'month_actu_depa']).count()[['delay_dest']]

    # 有delay的航班 / 總航班數
    flight_count['delay_rate'] = flight_count["year_month_delay_count"] / \
        flight_count["year_month_count"]

    # 重新排序，並將depa_month轉成字串
    flight_count = flight_count.reset_index(
        drop=False).sort_values('month_actu_depa')
    flight_count['month_actu_depa'] = flight_count['month_actu_depa'].astype(
        str)

    # 只選擇疫情後年份
    #flight_count = flight_count[flight_count['year_actu_depa'].isin([2020, 2021, 2022])]

    # 這個版本是存放疫情後，有需要要再更改
    flight_count.to_csv(
        '../output_data/Time_Delay_by_year_month_Final.csv', index=False)

    ''' Plot '''
    # 圖表背景的風格
    sns.set_style('darkgrid')

    # 調色板風格，其顏色較柔和
    sns.set_palette('pastel')

    # 使用relplot可以將hue產生的圖示放在圖的外面，不會擋到線
    # kind="line"是畫線圖的意思，height&aspect是設定大小，relplot不適用plt.subplots()
    sns.relplot(data=flight_count, x="month_actu_depa", y="delay_rate", hue="year_actu_depa", kind="line",
                palette="bright", height=4, aspect=1.5).set(title="Delay Rate by Year & Month", xlabel="Month", ylabel="Delay Rate")

    # 將圖表另存出來為'XXX.png'，還可以存為jpg、svg等格式的圖片
    plt.savefig('../images/Time_Delay_by_year_month_Final.png')

# %% [markdown]
# #### Plot 1-1

# %%


def flight_year_epidemic_month(dataset, delay_time):
    ''' Data '''
    flight_year_epidemic_count = pd.DataFrame()
    # 設定year_epidemic
    dataset['year_epidemic'] = pd.cut(
        dataset['year_actu_depa'], bins=year_bin, labels=year_labels)

    # by time_period 總航班
    flight_year_epidemic_count['year_month_count'] = dataset.groupby(
        ['year_epidemic', 'month_actu_depa']).count()[['delay_dest']]
    # by time_period 有誤點的航班
    flight_year_epidemic_count['year_month_delay_count'] = dataset[dataset['delay_dest'] > delay_time].groupby(
        ['year_epidemic', 'month_actu_depa']).count()[['delay_dest']]

    # 有delay的航班 / 總航班數
    flight_year_epidemic_count['delay_rate'] = flight_year_epidemic_count["year_month_delay_count"] / \
        flight_year_epidemic_count["year_month_count"]

    # 排序，設定型態
    flight_year_epidemic_count = flight_year_epidemic_count.reset_index(
        drop=False).sort_values('month_actu_depa')
    flight_year_epidemic_count['month_actu_depa'] = flight_year_epidemic_count['month_actu_depa'].astype(
        str)

    # to csv
    flight_year_epidemic_count.to_csv(
        '../output_data/Time_Delay_by_year_month_epidemic_Final.csv', index=False)

    ''' Plot '''
    # 圖表背景的風格
    sns.set_style('darkgrid')

    # 調色板風格，其顏色較柔和
    sns.set_palette('pastel')

    sns.relplot(data=flight_year_epidemic_count, x="month_actu_depa", y="delay_rate", hue="year_epidemic", color='gray',
                kind="line",  height=4, aspect=1.5).set(title="Delay Rate by Month & year_epidemic", xlabel="Month", ylabel="Delay Rate")

    # 將圖表另存出來為'XXX.png'，還可以存為jpg、svg等格式的圖片
    plt.savefig('../images/Time_Delay_by_year_month_epidemic_Final.png')

# %% [markdown]
# ### Plot - 2

# %%


def Yearly_Delay(dataset, delay_time):
    ''' Data '''
    year_delay = pd.DataFrame()
    # by year 取得總航班數
    year_delay['year_count'] = dataset.groupby(
        "year_actu_depa").count()[['delay_dest']]
    # by year 取得有delay的航班
    year_delay['year_delay_count'] = dataset[dataset['delay_dest'] >
                                             delay_time].groupby("year_actu_depa").count()[['delay_dest']]

    # 有delay的航班 / 總航班數
    year_delay['delay_rate'] = year_delay['year_delay_count'] / \
        year_delay['year_count']

    # 重新排序
    year_delay = year_delay.reset_index(
        drop=False).sort_values('year_actu_depa')

    # to csv
    year_delay.to_csv(
        '../output_data/Time_Delay_By_Yearly_Final.csv', index=False)

    ''' Plot '''
    # 調色板風格，其顏色較柔和
    sns.set_palette('pastel')

    # 圖表背景的風格，灰(白)底有格線
    sns.set_style('darkgrid')

    # 設定畫布大小
    plt.figure(figsize=(12, 6))

    # lineplot
    sns.lineplot(data=year_delay, x="year_actu_depa", y="delay_rate", marker='o').set(
        title='Delay rate by year', xlabel="Year", ylabel="Delay rate")

    # plt.show()
    # 將圖表另存出來為'XXX.png'，還可以存為jpg、svg等格式的圖片
    plt.savefig('../images/Time_Delay_By_Yearly_Final.png')


# %% [markdown]
# ### Plot - 3

# %%
def Holiday_Delay(dataset, delay_time):
    ''' Data '''
    holiday_delay = pd.DataFrame()
    # by year 取得總航班數
    holiday_delay['holiday_count'] = dataset.groupby(
        ["holiday", "year_epidemic"]).count()[['delay_dest']]
    # by year 取得有delay的航班
    holiday_delay['holiday_delay_count'] = dataset[dataset['delay_dest'] >
                                                   delay_time].groupby(["holiday", "year_epidemic"]).count()[['delay_dest']]

    # 有delay的航班 / 總航班數
    holiday_delay['delay_rate'] = holiday_delay['holiday_delay_count'] / \
        holiday_delay['holiday_count']

    # 重新排序
    holiday_delay = holiday_delay.reset_index(drop=False)
    # 再把日期合併進來以便x軸排序
    holiday_delay = holiday_delay.merge(
        holiday_df, left_on="holiday", right_on="holiday", how="left")
    holiday_delay = holiday_delay.sort_values('date')

    # to csv
    holiday_delay.to_csv(
        '../output_data/Time_Delay_By_Holiday_Final.csv', index=False)

    ''' Plot '''
    # 調色板風格，其顏色較柔和
    sns.set_palette('pastel')

    # 圖表背景的風格，灰(白)底有格線
    sns.set_style('darkgrid')

    # 設定畫布大小
    plt.figure(figsize=(15, 6))

    # lineplot
    sns.lineplot(data=holiday_delay, x="holiday", y="delay_rate", hue="year_epidemic", marker='o').set(
        title='Delay rate by Holiday', xlabel="Holiday", ylabel="Delay rate")

    # 將圖表另存出來為'XXX.png'，還可以存為jpg、svg等格式的圖片
    plt.savefig('../images/Time_Delay_By_Holiday_Final.png')

# %% [markdown]
# ### Plot - 4

# %%


def Hour_Delay(dataset, delay_time):
    ''' Data '''
    hour_delay = pd.DataFrame()
    # by year 取得總航班數
    hour_delay['hour_count'] = dataset.groupby(
        "hr_actu_depa").count()[['delay_dest']]
    # by year 取得有delay的航班
    hour_delay['hour_delay_count'] = dataset[dataset['delay_dest'] >
                                             delay_time].groupby("hr_actu_depa").count()[['delay_dest']]

    # 有delay的航班 / 總航班數
    hour_delay['delay_rate'] = hour_delay['hour_delay_count'] / \
        hour_delay['hour_count']

    # 重新排序
    hour_delay = hour_delay.reset_index(drop=False).sort_values('hr_actu_depa')
    hour_delay['hr_actu_depa'] = hour_delay['hr_actu_depa'].astype(str)

    # to csv
    hour_delay.to_csv(
        '../output_data/Time_Delay_By_Hour_Final.csv', index=False)

    ''' Plot '''
    # 調色板風格，其顏色較柔和
    sns.set_palette('pastel')

    # 圖表背景的風格，灰(白)底有格線
    sns.set_style('darkgrid')

    # 設定畫布大小
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # barplot
    sns.barplot(data=hour_delay, x="hr_actu_depa",
                y="hour_count", palette="dark:#84C1FF", ax=ax1)
    # 設定左側y軸標籤
    ax1.set_ylabel('hour_count')

    # 畫不同的Y軸
    ax2 = ax1.twinx()
    # lineplot
    sns.lineplot(data=hour_delay, x="hr_actu_depa",
                 y="delay_rate", marker='o', ax=ax2)

    # 設定右側y軸標籤
    ax2.set_ylabel('delay rate')
    # 不顯示網格線
    ax2.yaxis.grid(False)

    # 設定title跟x軸文字
    ax1.set_title('Delay rate by Hour')
    ax1.set_xlabel('Hour')

    # 自定義Y軸標籤格式
    ax1.yaxis.set_major_formatter(
        FuncFormatter(lambda x, loc: "{:.0f}".format(x)))

    # 將圖表另存出來為'XXX.png'，還可以存為jpg、svg等格式的圖片
    plt.savefig('../images/Time_Delay_By_Hour_Final.png')

# %% [markdown]
# ### Run


# %%
if __name__ == "__main__":
    flight_year_month(delay_rate_by_year_month, 5)
    flight_year_epidemic_month(delay_rate_by_year_month, 5)
    Yearly_Delay(delay_rate_by_year, 5)
    Holiday_Delay(delay_rate_by_holiday, 5)
    Hour_Delay(delay_rate_by_hours, 5)
