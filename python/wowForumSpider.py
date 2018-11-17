"""
Created on 10/19/17 at 9:06 AM
@author: David
"""

"""
This scrapy spider is designed to scrape forum information from the EU and US WoW forums.
Parsing is performed using a combination of BeautifulSoup, xpath, and regular expressions.
"""

import re
import scrapy
from bs4 import BeautifulSoup
from scrapy.http import Request
import urllib.parse as urlparse
from scrapy.shell import inspect_response

class wowForumSpider(scrapy.Spider):

	name = 'wowForum'
	allowed_domains = ['us.battle.net']
	start_urls = ['https://us.battle.net/forums/en/wow/']
    
	def parse(self, response):
		"""Login to the phpBB board and call the after_login method."""
		return [scrapy.FormRequest.from_response(
                response,
                formdata={'username': 'Dascienz@gmail.com', 'password': 'Systemthrice88'},
                callback=self.after_login)]

	def after_login(self, response):
		"""Check login success before requesting forum links."""
		if b"authentication failed" in response.body:
			self.logger.error("Login failed!")
			return
		else:
			links = response.xpath("//a//@href").re("/forums/en/wow/\d+.")
			for link in links:
				url = urlparse.urljoin(response.url, link)
				yield scrapy.Request(url, callback=self.parse_topics, dont_filter=False)
    
	def parse_topics(self, response):
		"""Request topic links."""
		links = response.xpath("//a//@href").re("/forums/en/wow/topic/\d+.")
		for link in links:
			url = urlparse.urljoin(response.url, link)
			yield scrapy.Request(url, callback=self.parse_posts, dont_filter=False)
		
		Next = response.xpath("//a[contains(@class,'Next') or contains(@class, 'next')]//@href").extract_first()
		if Next is not None:
			yield scrapy.Request(response.urljoin(Next), callback=self.parse_topics, dont_filter=False)

	def parse_posts(self, response):
		"""Collect post-content divs for parsing."""
	
		divs = response.xpath('//div[@class="TopicPost-content"]').extract()
		
		for div in divs:
			soup = BeautifulSoup(div, 'lxml')
			for html in soup.find_all('blockquote'):
					html.replaceWith("")	
			if re.findall('(Author--blizzard)',str(soup.find_all('div'))) != []:
				name = soup.find('span').text.strip('\n')
				lvlRaceClass = 'NaN'
				achievements = 'NaN'
				postCount = soup.find_all('span')[2].text.split(' ')[0].strip('\n\n')
				datetime = "".join(re.findall("(\d+/\d+/\d+[ ]\d+:\d+[ ][A-Z]+)|(\d+/\d+/\d+[ ]\d+:\d+)[ \"A-Z]+", str(soup.find_all('a')))[0])
				postText = soup.find_all('div')[-1].text
				yield {'name':name,'lvlRaceClass':lvlRaceClass,'achievements':achievements,'postCount':postCount,'datetime':datetime,'postText':postText}
			else:
				name = soup.find('span').text.strip('\n')
				lvlRaceClass = re.findall('"Author-class[ a-z]+">\n(.*?)\n</span>',str(soup.find_all('span')))[0]
				achievements = re.findall('</i>(\d+)',str(soup.find_all('span')))[0]
				postCount = re.findall('(\d+)[ ][Pp]os[ts]',str(soup.find_all('span')))[0]
				datetime = "".join(re.findall("(\d+/\d+/\d+[ ]\d+:\d+[ ][A-Z]+)|(\d+/\d+/\d+[ ]\d+:\d+)[ \"A-Z]+", str(soup.find_all('a')))[0])	
				postText = soup.find_all('div')[-1].text
				yield {'name':name,'lvlRaceClass':lvlRaceClass,'achievements':achievements,'postCount':postCount,'datetime':datetime,'postText':postText}
        
		Next = response.xpath("//a[contains(@class,'Next') or contains(@class, 'next')]//@href").extract_first()
		if Next is not None:
			yield scrapy.Request(response.urljoin(Next), callback=self.parse_posts, dont_filter=False)