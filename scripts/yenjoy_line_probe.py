from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

URL = "https://www.yen-joy.net/kaisai/race/forecast/line/202609/48/20260909/20260911"


def clean(s: str) -> str:
    return " ".join(s.replace("\xa0", " ").replace("\u3000", " ").split())


def main() -> int:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; keirin-market-analysis/0.4; research probe)",
        "Accept-Language": "ja,en;q=0.7",
    })
    r = s.get(URL, timeout=60)
    print("status", r.status_code, "bytes", len(r.content), "final", r.url)
    r.raise_for_status()
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "lxml")
    print("title", clean(soup.title.get_text(" ", strip=True)) if soup.title else "")

    tables = [t for t in soup.find_all("table") if t.find_parent("table") is None]
    print("top_level_tables", len(tables))
    for i, table in enumerate(tables[:30]):
        print("\nTABLE", i, "class", table.get("class", []))
        for tr in table.find_all("tr")[:12]:
            if tr.find_parent("table") is not table:
                continue
            cells = []
            for c in tr.find_all(["th", "td"]):
                if c.find_parent("tr") is tr and c.find_parent("table") is table:
                    cells.append(clean(c.get_text(" ", strip=True))[:250])
            if cells:
                print(cells)

    print("\nFORECAST DETAIL LINKS")
    seen = set()
    for a in soup.find_all("a", href=True):
        u = urljoin(r.url, a["href"])
        if "/forecast/detail/" in u and u not in seen:
            seen.add(u)
            print(clean(a.get_text(" ", strip=True)), u)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
