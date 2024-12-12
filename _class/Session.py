import requests
import requests_cache



class Session:
    """CachedSession wrapper with custom dunder methods and properties."""

    def __init__(self, name: str = 'YfCache',
                 expire_after: int = 3600*24,
                 allowable_methods: tuple = ('GET', 'POST')) -> None:

        self.session = requests_cache.CachedSession(name,
                                                    expire_after=expire_after,
                                                    allowable_methods=allowable_methods)


    def get_session(self) -> requests_cache.CachedSession:
        return self.session

    def get(self, url: str, **kwargs) -> requests.Response:
        return self.session.get(url, **kwargs)

    def clear(self) -> None:
        self.session.cache.clear()


    def __str__(self):
        return f"{self.session_info}"

    def __len__(self):
        return len(self.session.cache.responses)

    def __int__(self):
        return self.session.__sizeof__() / 1000  # Decimal kilobytes (Not KiB)

    @property
    def session_info(self):

        info = {
            'cache_name': self.session.cache_name,
            'expire_after': self.session.expire_after,
            'allowable_methods': self.session.allowable_methods,
            'response_count': len(self.session.cache.responses),
            'cache_size': self.session.__sizeof__() / 1000
        }
