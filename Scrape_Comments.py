import time
import json
#from pprint import pprint
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import numpy as np
import nltk as nlp
#from selenium.common.exceptions import NoSuchElementException

# --------------------------------------------------------------------
# change chrome path to local installation
chrome_path = "C:\Webdriver/chromedriver.exe"
# change youtube URL to scrape different video's comments
page_url = "https://www.youtube.com/watch?v=xZFG6c_zUdI"  
# --------------------------------------------------------------------

# --------------- PAGE ACCESS ---------------
# accessing the page holding comments (here: youtube)
driver = webdriver.Chrome(executable_path=chrome_path)
driver.get(page_url)
time.sleep(5)  # give the page some time to load

# --------------- FETCH TITLE ---------------
# get the video's title
title = driver.find_element_by_xpath('//*[@id="container"]/h1/yt-formatted-string').text
print(title)

# --------------- LOAD ALL COMMENTS ---------------
# defining the numbers here so we can reference and easily change them
SCROLL_PAUSE_TIME = 7
CYCLES = 10
# we know there's always exactly one HTML element, so let's access it
html = driver.find_element_by_tag_name('html')
# first time needs to not jump to the very end in order to start
html.send_keys(Keys.PAGE_DOWN)  # doing it twice for good measure
html.send_keys(Keys.PAGE_DOWN)  # one time sometimes wasn't enough
# adding extra time for initial comments to load
# if they fail (because too little time allowed), the whole script breaks
time.sleep(SCROLL_PAUSE_TIME * 3)
# and now for loading the hidden comments by scrolling down and up
for i in range(CYCLES):
    html.send_keys(Keys.END)
    time.sleep(SCROLL_PAUSE_TIME)
    # might not be necessary; try out without it.
    # html.send_keys(Keys.PAGE_UP)
    # time.sleep(SCROLL_PAUSE_TIME)






# --------------- GETTING THE COMMENT TEXTS ---------------
comment_elems = driver.find_elements_by_xpath('//*[@id="content-text"]')
# pprint(comment_elems)  # for double-checking
all_comments = [elem.text for elem in comment_elems]

# --------------- WRITING TO OUTPUT FILE ---------------
with open('comments50.json', 'w') as f:
    json.dump(all_comments, f)


