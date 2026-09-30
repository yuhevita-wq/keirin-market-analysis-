from __future__ import annotations

import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.yen-joy.net"
URL = f"{BASE}/kaisai"
SAMPLES = [
    f"{BASE}/kaisai/race/forecast/detail/202609/48/20260909/20260911/9",
    f"{BASE}/kaisai/race/result/detail/202609/75/20260912/20260912/1",
]


def dump_tables(session: requests.Session, url: str) -> None:
    print("\n=== sample ===", url)
    r = session.get(url, timeout=30)
    print("status=", r.status_code, "bytes=", len(r.content), "final=", r.url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    tables = soup.find_all("table")
    print("table_count=", len(tables))
    for i, table in enumerate(tables[:20]):
        cls = " ".join(table.get("class", []))
        print(f"-- table[{i}] class={cls!r}")
        rows = table.find_all("tr")
        for row in rows[:4]:
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            print(cells[:24])


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

    for sample in SAMPLES:
        dump_tables(s, sample)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
