# -*- coding: utf-8 -*-
"""
small translater based on Baidu translation api.

AUTHOR: Yue Peng
EMAIL: ypeng7@outlook.com
DATE: 2018.10.09
"""
import http.client
import hashlib
import json
import urllib
import random


class BaiduTranslate(object):
    def __init__(self, appid, secretKey):
        self._appid = appid
        self._secretKey = secretKey
        self.LANG = ["zh", "en", "yue", "wyw", "jp", "kor", "fra", "spa", "th", "ara", "ru", "pt", "de", "it", "el", "nl", "pl", "bul", "est", "dan", "fin", "cs", "rom", "slo", "swe", "hu", "cht", "vie"]
        self.FOREIGN_LANG = [l for l in self.LANG if l != "zh"]

    def translate(self, fromLang, toLang, srcString):
        """translate function based on baidu fanyi api
        
        [description]
        
        Arguments:
            appid -- [description]
            secretKey -- [description]
            fromLang -- [description]
            toLang -- [description]
            srcString -- input sentece
        """

        httpClient = None
        myurl = '/api/trans/vip/translate'
        q = srcString
        salt = random.randint(32768, 65536)

        sign = self._appid + q + str(salt) + self._secretKey
        sign = hashlib.md5(sign.encode()).hexdigest()
        myurl = myurl + '?appid=' + self._appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign

        trRet = ""
        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', myurl)
            response = httpClient.getresponse()
            ret =  response.read().decode("utf-8")
            jobj = json.loads(ret)
            trRet = str(jobj['trans_result'][0]['dst'])
        except Exception as e:
            print(e)
        finally:
            if httpClient:
                httpClient.close()
        
        return trRet


if __name__ == "__main__":
    # appid = '201810090002171181'
    # secretKey = 'hUNxtY0xjFCwKCNEcPfr1'
    fromLang = 'en'
    toLang = 'zh'
    fanyi = BaiduTranslate()
    print(fanyi.translate(fromLang, toLang, "Hello, world! And I feel Good!"))
