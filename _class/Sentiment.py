import re

import googlesearch as gs

import requests
from requests_cache import CachedSession

from bs4 import BeautifulSoup

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch

from typing import Optional

class Sentiment:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

    @staticmethod
    def get_cache():
        session = CachedSession('URL_cache', backend='sqlite', expire_after=None)
        return session

    @staticmethod
    def search(self, query: str, n: int, tld: str = 'com') -> list:
        search_results = gs.search(query, num=n)
        return list(search_results)

    @staticmethod
    def get_page_contents(self, url: str, elements: list = r'^p$|^h[1-6]$', session: Optional[CachedSession] = None) -> str:
        if not session:
            session = self.get_cache()

        response = session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        page_contents = "".join([element.get_text() for element in soup.find_all(elements)])

        return page_contents

    def get_score_all(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)

        logits = outputs.logits.squeeze()
        pval = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()

        return pval

    def compose_sentiment(self, text: str) -> str:
        pval = self.get_score_all(text)
        score = ((pval[2] - pval[0]) + 1) / 2  # Normalize to (0, 1)