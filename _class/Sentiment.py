
from datetime import timedelta, datetime as dt
import os
from dotenv import load_dotenv
import re
import warnings

from newsapi import NewsApiClient

from requests_cache import CachedSession

from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

from typing import Optional
from CustomTypes import Days

class Sentiment:
    def __init__(self):
        load_dotenv('_class/api.env')

        if os.getenv('API_KEY') == None:
            raise ValueError("API_KEY not found in environment variables")

        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

    @staticmethod
    def get_cache():
        session = CachedSession('URL_cache', backend='sqlite', expire_after=None)
        return session


    def search(self, query: str, *, n: int, lookback: Days) -> list:
        key = os.getenv('API_KEY')
        newsapi = NewsApiClient(api_key=key)

        date = dt.today() - timedelta(days=lookback)

        articles = newsapi.get_everything(q=query,
                                          from_param=date,
                                          language='en',
                                          sort_by='publishedAt',
                                          page_size=n)

        desc = [article['title'] + " " + article['description'] for article in articles['articles']]
        return desc

    # def get_page_contents(self, url: str, elements: list = r'^p$|^h[1-6]$', session: Optional[CachedSession] = None) -> str:
    #     if not session:
    #         session = self.get_cache()
    #
    #     response = session.get(url)
    #     if response.status_code != 200:
    #         warnings.warn(f"Failed to get page contents from {url}", UserWarning)
    #         return ""
    #
    #
    #     soup = BeautifulSoup(response.text, 'html.parser')
    #     page_contents = "".join([element.get_text() for element in soup.find_all(elements)])
    #
    #     if not page_contents:
    #         warnings.warn(f"No text found in {url}", UserWarning)
    #         return ""
    #
    #     return page_contents

    def get_score_all(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)

        logits = outputs.logits.squeeze()
        p_val = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()

        return p_val

    def compose_sentiment(self, text: str) -> str:
        p_val = self.get_score_all(text)
        if p_val[0] + p_val[2] == 0:
            return 0.5 # Neutral if no positive or negative sentiment

        score = p_val[2] / (p_val[0] + p_val[2]) # \frac{positive}{positive + negative}
        return score

    def get_sentiment(self, query: str, n: int, lookback: Days) -> dict:
        search_results = self.search(query, n=n, lookback=lookback)
        sentiments = []

        for desc in search_results:
            score = self.compose_sentiment(desc)
            sentiments.append(score)

        ewma_sentiment = pd.Series(sentiments).ewm(halflife=2).mean().iloc[-1]
        return ewma_sentiment


