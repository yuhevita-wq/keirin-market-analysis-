from __future__ import annotations

import re
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.yen-joy.net"
URL = f"{BASE}/kaisai"


def main() -> int:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; keirin-market-analysis/0.1; research collector; +https://github.com/yuhevita-wq/keirin-market-analysis-)"
    })
    r = s.get(URL, timeout=30)
    print("status=", r.status_code)
    print("final_url=", r.url)
    print("content_type=", r.headers.get("content-type"))
    print("bytes=", len(r.content))
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(r.url, a["href"])
        if href not in links:
            links.append(href)

    print("link_count=", len(links))
    print("--- candidate links ---")
    pat = re.compile(r"/(kaisai|race|calendar)|20\d{4}")
    shown = 0
    for href in links:
        if pat.search(href):
            print(href)
            shown += 1
            if shown >= 150:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
