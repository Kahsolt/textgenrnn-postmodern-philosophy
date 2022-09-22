#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/18 

# 从一些哲学向的微信公众号爬取文章
# 修改 TRACKINGS 变量以追踪栏目

from time import sleep
from dataclasses import asdict
from random import gauss, random
from argparse import ArgumentParser
from traceback import print_exc

from bs4 import BeautifulSoup
from requests import Session

from util import *

TRACKINGS = {   # { 'BIZ': ['album_id', ...] }
  'MzU3NjY5MzE1OQ==': [       # 后现代主义哲学 by 阿月
    '1901800772409802763',    # 文化批判
    '1901795827996475400',    # 女权主义
    '1901791942024151042',    # 社会分析
    '2500940578789031937',    # 哲学课程
  ],
  'Mzg4Nzc2Mjg2Ng==': [       # 思庐哲学
    '2350335243620122625',    # 每周哲学话题
    '2351592845343653891',    # 哲学文创
    '2351597440740556801',    # 哲学情诗
    '2360324632655888385',    # 哲学歌曲
    '2360326879880085507',    # 闭观计划
    '2351598557935370241',    # 分析之境
    '2351599486218731521',    # 图解哲普
  ],

  # 路标读书会

}

HEARDERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:104.0) Gecko/20100101 Firefox/104.0'
}

# https://github.com/OxOOo/ProxyPoolWithUI
# 国内暂时没有较为稳定的代理，建议是别用
USE_PROXY = False
PROXIES = {
  'http': 'http://127.0.0.1:2333',
  'https': 'http://127.0.0.1:2333',
}

PAGE_SIZE = 10             


def GET(url, retry=2):
  while retry > 0:
    retry -= 1
    try:
      print(f'[GET] {url}')
      resp = http.get(url)
      sleep(min(7, max(1, gauss(mu=5, sigma=1))))
      return resp
    except Exception as e:
      print_exc()
  
  breakpoint()

def GET_html(url):
  resp = GET(url)
  html = BeautifulSoup(resp.content, features='html5lib')
  return html

def GET_json(url):
  resp = GET(url)
  return resp.json()


def browse_article_urls(biz: str, album_id: str) -> str:
  api = 'https://mp.weixin.qq.com/mp/appmsgalbum'

  def album_index_url():       # 返回html，用于浏览器GUI显示
    params = {
      'action': 'getalbum',
      '__biz': biz,
      'album_id': album_id,
    }
    return make_url(api, params)

  def album_index_paging_url(last_msgid='', last_itemidx=''):   # 返回json
    params = {
      'action': 'paging',
      '__biz': biz,
      'album_id': album_id,
      'count': 10,                     # 页面大小，网页版SDK默认值为10
      'begin_msgid': last_msgid,       # 偏移位置 (msgid和itemidx可能是个用于唯一索引文档的联合主键)
      'begin_itemidx': last_itemidx,
    }

    # 下列参数可能非必须
    #params.update({
    #  'cur_msgid': 'undefined',  # 空字符串也行
    #  'cur_itemidx': 1,          # 不知为何，好像其他数字都是一个效果
    #  'uin': '',
    #  'key': '',
    #  'pass_ticket': '',
    #  'wxtoken': '777',
    #  'devicetype': '',
    #  'clientversion': '',
    #  'appmsg_token': '',
    #  'x5': '0',
    #  'f': 'json',
    #})

    return make_url(api, params)

  last_msgid, last_itemidx = '', ''
  while True:
    url = album_index_paging_url(last_msgid, last_itemidx)
    data = GET_json(url)
    article_list = data['getalbum_resp']['article_list']
    if len(article_list) == 0: break

    for x in article_list:
      yield cleanify_url(x['url'], filter_keys=['__biz', 'mid', 'idx', 'sn'])

    if data['getalbum_resp']['continue_flag'] == 1:
      last_msgid   = x['msgid']
      last_itemidx = x['itemidx']
    else:
     break


def parse_article(url) -> Article:
  html = GET_html(url)

  # title
  h1 = html.find('h1', attrs={'id': 'activity-name'})
  title = h1.text.strip()

  # author
  try:
    div = html.find('div', attrs={'id': 'meta_content'})
    span = div.find('span', attrs={'class': 'rich_media_meta rich_media_meta_text'})    # NOTE: this might NOT be stable
    author = span.text.strip()
  except:
    author = ''
  
  # organization
  span = html.find('span', attrs={'id': 'profileBt'})
  a = span.find('a', attrs={'id': 'js_name'})
  organization = a.text.strip()
  
  # content
  div = html.find('div', attrs={'id': 'js_content'})
  content = div.text.replace('\xa0', '').replace('\u200b', '').strip()

  return Article(url, title, author, organization, content)


def crawl(args):
  global http
  http = Session()
  http.headers.update(HEARDERS)
  if USE_PROXY: http.proxies.update(PROXIES)

  crawler_database = load_crawler_database()
  downloaded_urls = {x['url'] for x in crawler_database}
  cnt_new = 0

  try:
    for biz, album_ids in TRACKINGS.items():
      for album_id in album_ids:
        try:
          for url in browse_article_urls(biz, album_id):
            if url in downloaded_urls:
              if args.mode == 'update': break       # when 'update', meets the new-old border line, ignore all older articles
              elif args.mode == 'sync': continue    # when 'sync', just ignore the current one
              else: pass                            # when 'force_sync', force re-download all articles

            try:
              x = parse_article(url)
              crawler_database.append(asdict(x))
              downloaded_urls.add(url)
              cnt_new += 1
            except Exception as e:
              print_exc()
              print(f'[Error] failed crawling {url}')
              breakpoint()

        except Exception as e:
          print_exc()
          print(f'[Error] during crawling {biz}-{album_id}')
      
      save_crawler_database(crawler_database)
  except KeyboardInterrupt:
    print('[Exit] on Ctrl+C')
  finally:
    save_crawler_database(crawler_database)
  
  print(f'[Crawl] done, added {cnt_new} new records')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--mode', default='update', choices=['update', 'sync', 'force_sync'], 
                      help='`sync` for the first time, then `update` only the differentials; `force_sync` will re-download & overwrite all')
  args = parser.parse_args()

  crawl(args)