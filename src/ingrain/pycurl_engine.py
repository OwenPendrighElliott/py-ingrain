import pycurl
import json
import certifi
from io import BytesIO

from typing import Tuple, List


class PyCURLEngine:
    def __init__(
        self,
        timeout: int = 600,
        connect_timeout: int = 600,
        header: List[str] = ["Content-Type: application/json"],
        user_agent: str = "ingrain-client/1.0.0",
    ):
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.header = header
        self.user_agent = user_agent

        self.curl = pycurl.Curl()
        self.curl.setopt(pycurl.HTTPHEADER, self.header)
        self.curl.setopt(pycurl.TIMEOUT, self.timeout)
        self.curl.setopt(pycurl.CONNECTTIMEOUT, self.connect_timeout)
        self.curl.setopt(pycurl.USERAGENT, self.user_agent)
        self.curl.setopt(pycurl.CAINFO, certifi.where())
        
        self.curl.setopt(pycurl.TCP_NODELAY, 1)
        self.curl.setopt(pycurl.ENCODING, 'gzip, deflate')
        self.curl.setopt(pycurl.FORBID_REUSE, 0)
        self.curl.setopt(pycurl.FRESH_CONNECT, 0)
        self.curl.setopt(pycurl.TCP_KEEPALIVE, 1)
        self.curl.setopt(pycurl.DNS_CACHE_TIMEOUT, 3600)

    def _execute(self) -> Tuple[dict, int]:
        response = BytesIO()
        self.curl.setopt(pycurl.WRITEFUNCTION, response.write)
        self.curl.perform()
        return json.loads(response.getvalue()), self.curl.getinfo(pycurl.HTTP_CODE)

    def post(self, url: str, data: dict) -> Tuple[dict, int]:
        self.curl.setopt(pycurl.URL, url)
        self.curl.setopt(pycurl.CUSTOMREQUEST, "POST")
        self.curl.setopt(pycurl.POSTFIELDS, json.dumps(data))
        return self._execute()

    def get(self, url: str) -> Tuple[dict, int]:
        self.curl.setopt(pycurl.URL, url)
        self.curl.setopt(pycurl.CUSTOMREQUEST, "GET")
        return self._execute()

    def put(self, url: str, data: dict) -> Tuple[dict, int]:
        self.curl.setopt(pycurl.URL, url)
        self.curl.setopt(pycurl.CUSTOMREQUEST, "PUT")
        self.curl.setopt(pycurl.POSTFIELDS, json.dumps(data))
        return self._execute()

    def delete(self, url: str) -> Tuple[dict, int]:
        self.curl.setopt(pycurl.URL, url)
        self.curl.setopt(pycurl.CUSTOMREQUEST, "DELETE")
        return self._execute()

    def patch(self, url: str, data: dict) -> Tuple[dict, int]:
        self.curl.setopt(pycurl.URL, url)
        self.curl.setopt(pycurl.CUSTOMREQUEST, "PATCH")
        self.curl.setopt(pycurl.POSTFIELDS, json.dumps(data))
        return self._execute()

    def close(self):
        self.curl.close()

    def __del__(self):
        self.close()
