
from datetime import timedelta, datetime as dt
import os
import warnings
from dotenv import load_dotenv

from .Cache import SentimentCache

from newsapi import NewsApiClient

import pandas as pd

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from .CustomTypes import Days


class Sentiment:
    """
    FinBERT Sentiment Analysis for Financial News.
    """
    def __init__(self, api_key_path: os.PathLike, api_key_var: str) -> None:
        warnings.filterwarnings("ignore")
        if not os.path.exists(api_key_path):
            raise FileNotFoundError(api_key_path)

        self.set_api_key(api_key_path)

        if os.getenv(api_key_var) is None:
            raise ValueError(f"{api_key_var} not found in environment variables")

        self.key_var = api_key_var

        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        self.cache = self._init_cache()

    @staticmethod
    def set_api_key(path: os.PathLike | str) -> None:
        if not str(path).split('.')[-1] == 'env':
            raise ValueError("Invalid file type. Must be a .env file")

        load_dotenv(path)

    @staticmethod
    def _init_cache(name: str = 'sentiment.db', exp_seconds: int = 3600) -> SentimentCache:
        return SentimentCache(name, exp_seconds)

    def search(self, query: str, *, n: int, lookback: Days) -> list:
        key = os.getenv(self.key_var)
        newsapi = NewsApiClient(api_key=key)

        date = dt.today() - timedelta(days=lookback)

        articles = newsapi.get_everything(q=query,
                                          from_param=date,
                                          language='en',
                                          sort_by='publishedAt',
                                          page_size=n)

        filtered_articles = []

        neutral_string = 'confident.' # Evaluates to a sentiment polarity of 0.0 (neutral). 200 IQ solution.

        for article in articles['articles']:
            if (article['description'] is None or article['description'] == '') and \
                    (article['title'] is None or article['title'] == ''):
                continue  # Skip articles with both description and title as None or empty

            # Ensure no None values remain
            if article['description'] is None:
                article['description'] = neutral_string
            if article['title'] is None:
                article['title'] = neutral_string

            filtered_articles.append(article)

        desc = [article['description'] + ' ' + article['title'] for article in filtered_articles]

        if not desc:
            raise ValueError("No articles found for the given query")

        return desc

    def get_score_all(self, text: str) -> dict:

        assert text and isinstance(text, str), "Input must be a non-empty string"

        inputs = self.tokenizer(text,
                                return_tensors='pt',
                                padding='max_length',
                                truncation=True,
                                max_length=512)

        outputs = self.model(**inputs)

        logits = outputs.logits.squeeze()
        p_val = torch.nn.functional.softmax(logits, dim=0).detach().numpy()
        return p_val

    def compose_sentiment(self, text: str) -> str:
        p_val = self.get_score_all(text)

        score = p_val[1] - p_val[2]

        return score

    def get_sentiment(self, query: str, n: int, lookback: Days) -> float:
        cache_query = f"{query} {lookback=} {n=}"

        cache_response = self.cache.get(cache_query)

        # Cache hit
        if cache_response is not None:
            return cache_response

        # Cache miss
        search_results = self.search(query, n=n, lookback=lookback)
        sentiments = []

        for desc in search_results:
            score = self.compose_sentiment(desc)
            sentiments.append(score)

        if not sentiments:
            self.cache.cache(cache_query, 0.0)
            return .5  # Neutral if no sentiment found

        ewma_sentiment = pd.Series(sentiments).ewm(halflife=2).mean().iloc[-1]
        self.cache.cache(cache_query, ewma_sentiment)
        return ewma_sentiment


