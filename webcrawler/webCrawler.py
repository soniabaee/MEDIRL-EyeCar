#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 17:49:11 2019

@author: soniabaee
"""

import scrapy
from scrapy.selector import HtmlXPathSelector
from scrapy.http import Request

class LoginSpider(scrapy.Spider):
    name = 'https://insight.shrp2nds.us/data/index'
    start_urls = ['https://insight.shrp2nds.us/login/auth#/output?queryID=108012']

    def parse(self, response):
        return scrapy.FormRequest.from_response(
            response,
            formdata={'username': 'ik3r@virginia.edu', 'password': 'foxku111'},
            callback=self.after_login
        )

    def after_login(self, response):
        # check login succeed before going on
        if "authentication failed" in response.body:
            self.logger.error("Login failed")
            return

        # continue scraping with authenticated session...
        # We've successfully authenticated, let's have some fun!
        else:
            return Request(url="https://insight.shrp2nds.us/data/index",
                   callback=self.parse_tastypage)
       


import json

from scrapy.crawler import Crawler
from scrapy.loader import ItemLoader
from scrapy.loader.processors import TakeFirst, MapCompose, Join
from scrapy import log, signals, Spider, Item, Field
from scrapy.settings import Settings
from twisted.internet import reactor


# define an item class
class DmozItem(Item):
    title = Field()
    link = Field()
    desc = Field()


# define an item loader with input and output processors
class DmozItemLoader(ItemLoader):
    default_input_processor = MapCompose(str.strip)
    default_output_processor = TakeFirst()

    desc_out = Join()


# define a pipeline
class JsonWriterPipeline(object):
    def __init__(self):
        self.file = open('items.jl', 'wb')

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item


# define a spider
class DmozSpider(Spider):
    name = "dmoz"
    allowed_domains = ["dmoz.org"]
    start_urls = [
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Books/",
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Resources/"
    ]

    def parse(self, response):
        for sel in response.xpath('//ul/li'):
            loader = DmozItemLoader(DmozItem(), selector=sel, response=response)
            loader.add_xpath('title', 'a/text()')
            loader.add_xpath('link', 'a/@href')
            loader.add_xpath('desc', 'text()')
            yield loader.load_item()


# callback fired when the spider is closed
def callback(spider, reason):
    stats = spider.crawler.stats.get_stats()  # collect/log stats?

    # stop the reactor
    reactor.stop()


# instantiate settings and provide a custom configuration
settings = Settings()
settings.set('ITEM_PIPELINES', {
    '__main__.JsonWriterPipeline': 100
})

# instantiate a crawler passing in settings
crawler = Crawler(settings)

# instantiate a spider
spider = DmozSpider()

# configure signals
crawler.signals.connect(callback, signal=signals.spider_closed)

# configure and start the crawler
crawler.configure()
crawler.crawl(spider)
crawler.start()

# start logging
log.start()

# start the reactor (blocks execution)
reactor.run()
   