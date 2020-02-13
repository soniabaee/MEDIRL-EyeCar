# -*- coding: utf-8 -*-
import scrapy
from scrapy.shell import inspect_response


class RedditbotSpider(scrapy.Spider):
    name = 'redditbot'
    allowed_domains = ['www.reddit.com/r/gameofthrones/']
    start_urls = ['http://www.reddit.com/r/gameofthrones//']

    def parse(self, response):
        print(response.body)
        inspect_response(response, self)
