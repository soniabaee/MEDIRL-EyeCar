#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:32:06 2019

@author: soniabaee
"""

import scrapy
from scrapy.selector import HtmlXPathSelector
from scrapy.http import Request

class QuotesSpider(scrapy.Spider): 
   
    name = "quotes"
    
    def __init__(self, username='', password='',  **kwargs):
        super(QuotesSpider, self).__init__(**kwargs)

        self.dict = {}
        self.Name=[]
        self.Text=[]
        self.Time=[]

        if not username or not password:
            raise ValueError("You need to provide valid email and password!")
        else:
            self.username = username
            self.password = password

        self.start_urls = ['https://insight.shrp2nds.us/login']
        
    
    
    def parse(self, response):
        return scrapy.FormRequest.from_response(
                response,
                formxpath='//form[contains(@action, "login")]',
                formdata={'username': self.username,'pass': self.password},
                callback=self.parse_home
        )
        
    def parse_home(self, response):
        href = 'https://insight.shrp2nds.us/query/index#/output?queryID=108040'
        return scrapy.Request(url=href,callback=self.parse_page,)

    def parse_page(self, response):
        res = response.body
        peoples = scrapy.Selector(text=res).xpath('//h3/a/text()').extract()
        people_link = scrapy.Selector(text=res).xpath('//h3/a/@href').extract()
        print('Choose the conversation')
        print(peoples,people_link)
        for i in range(len(peoples)):
            print(i,peoples[i])
        convo = int(input())

        yield scrapy.Request(url='https://insight.shrp2nds.us/login'+ people_link[convo], callback=self.parse_convo)