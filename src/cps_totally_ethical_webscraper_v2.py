from bs4 import BeautifulSoup as BS
from selenium import webdriver
from functools import reduce
import pandas as pd
import time
import xport
import pandas as pd
import numpy as np

def render_page(url):
        driver = webdriver.Chrome('/Users/cp/Downloads/chromedriver')
        driver.get(url)
        time.sleep(3)
        r = driver.page_source
        driver.quit()
        return r

def scraper2(page, dates):
    output = pd.DataFrame()

    for d in dates:

        url = str(str(page) + str(d))

        r = render_page(url)

        soup = BS(r, "html.parser")
        container = soup.find('lib-city-history-observation')
        check = container.find('tbody')

        data = []

        for c in check.find_all('tr', class_='ng-star-inserted'):
            for i in c.find_all('td', class_='ng-star-inserted'):
                trial = i.text
                trial = trial.strip('  ')
                data.append(trial)

        if len(data)%2 == 0:
            hour = pd.DataFrame(data[0::10], columns = ['hour'])
            temp = pd.DataFrame(data[1::10], columns = ['temp'])
            dew = pd.DataFrame(data[2::10], columns = ['dew'])
            humidity = pd.DataFrame(data[3::10], columns = ['humidity'])
            wind_speed = pd.DataFrame(data[5::10], columns = ['wind_speed'])
            pressure =  pd.DataFrame(data[7::10], columns = ['pressure'])
            precip =  pd.DataFrame(data[8::10], columns = ['precip'])
            cloud =  pd.DataFrame(data[9::10], columns = ['cloud'])

        dfs = [hour, temp,dew,humidity,  wind_speed,  pressure, precip, cloud]
        df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)

        df_final['Date'] = str(d) + "-" + df_final.iloc[:, :1].astype(str)

        output = output.append(df_final)

    print('Scraper done!')
    output = output[['hour', 'temp', 'dew', 'humidity', 'wind_speed', 'pressure',
                     'precip', 'cloud']]

    return output

def make_year_dates(yr= 2020):
    if yr%4 ==0:
        feb_days = 30
    else:
        feb_days = 29
    yr = str(yr)
    
    january = []
    for i in range(1, 32):
        i = str(i)
        january.append(f"{yr}-1-{i}")
   
    feb = []
    for i in range(1, feb_days):
        i = str(i)
        feb.append(f"{yr}-2-{i}")
    
    march = []
    for i in range(1, 32):
        i = str(i)
        march.append(f"{yr}-3-{i}")
        
   
    april = []
    for i in range(1, 31):
        i = str(i)
        april.append(f"{yr}-4-{i}")
        
    may = []
    for i in range(1, 32):
        i = str(i)
        may.append(f"{yr}-5-{i}")
    june = []
    for i in range(1, 31):
        i = str(i)
        june.append(f"{yr}-6-{i}")
    july = []
    for i in range(1, 32):
        i = str(i)
        july.append(f"{yr}-7-{i}")
    
    august = []
    for i in range(1, 32):
        i = str(i)
        august.append(f"{yr}-8-{i}")
    september = []
    for i in range(1, 31):
        i = str(i)
        september.append(f"{yr}-9-{i}")
    
    october = []
    for i in range(1, 32):
        i = str(i)
        october.append(f"{yr}-10-{i}")
    november = []
    for i in range(1, 31):
        i = str(i)
        november.append(f"{yr}-11-{i}")
    december = []
    for i in range(1, 32):
        i = str(i)
        december.append(f"{yr}-12-{i}")
    yr_total = january+feb+ march+ april+ may+ june + july + august+ september+ october+ november+ december
    
    return yr_total

if __name__=='__main__':
    pass


