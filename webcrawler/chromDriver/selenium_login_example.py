# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 12:17:21 2019

@author: Moji
"""

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from time import sleep
import re
import pandas as pd

path_chromedriver =r"/Users/soniabaee/Documents/Projects/EyeCar/Code/chromDriver/chromedriver"
driver = webdriver.Chrome(executable_path= path_chromedriver)
base_site="https://insight.shrp2nds.us/login/auth"
driver.get(base_site)
sleep(1)

sleep(2)
# fill in the username field xpath might be different in your case
nikname=driver.find_element_by_xpath("""//*[@id="username"]""").send_keys("""ik3r@virginia.edu""")
# fill in the password field filled xpath might be different in your case
driver.find_element_by_xpath("""//*[@id="password"]""").send_keys("""foxku111""")
# click on the login button on the site
driver.find_element_by_xpath("//button[@type='submit']").click()
sleep(2)
driver.find_element_by_xpath('//a[@href="/data/index"]').click()
driver.find_element_by_xpath('//a[@href="/data/index"]').click()
'''do something when u entered the page ....
example here waits for a specific element to be visible and then click on it'''

card = driver.find_element_by_id('query-card')
card.find_element_by_partial_link_text('card-link').click()

driver.find_element_by_css_selector('button[ng-click="onShowLoadModal()"]').click()

table = driver.find_element_by_xpath("//table/tbody/tr[7]/td[1]");
table.find_element_by_css_selector('button[ng-click="onLoadQuery(query)"]').click()

driver.find_element_by_css_selector('button[ng-click="onShowOutputView()"]').click()
driver.find_element_by_css_selector('a[ng-click="select()"]').click()

places_xpath= """//*[@id="header_v2"]/div/nav/ul[1]/li[3]/a"""
wait = WebDriverWait(driver, 5)
wait.until(EC.visibility_of_element_located((By.XPATH, places_xpath)))


driver.find_element_by_xpath(places_xpath).click()
sleep(3)
driver.get("an element that is loaded....")

#### when driver is on a page ...
html = driver.page_source
soup = BeautifulSoup(html,'lxml')