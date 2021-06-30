import time
import json
#from pprint import pprint
from selenium import webdriver
import pandas as pd


driver = webdriver.Chrome(executable_path='C:\Webdriver/chromedriver.exe')
driver.get("https://www.youtube.com/watch?v=B5oIbP4ZYf8")  
time.sleep(10)
title = driver.find_element_by_xpath('//*[@id="container"]/h1/yt-formatted-string').text
print(title)


for i in range(0,10, 5):
    start_value = str(i) + "00"
    end_value = int(start_value) + 500
    end_value = str(end_value)

    driver.execute_script('window.scrollTo(' + str(start_value) + ',' + str(end_value) + ');')

    time.sleep(2)
comment_elems = driver.find_elements_by_xpath('//*[@id="content-text"]')

all_comments = [elem.text for elem in comment_elems]

with open('commentsPython.json', 'w') as f:
    json.dump(all_comments, f)

Create_CSV = pd.read_csv("commentsPython.json",header=None,delimiter = ',')

abc=Create_CSV.T  
abc.to_csv('commentsPython.csv', index = None) 
