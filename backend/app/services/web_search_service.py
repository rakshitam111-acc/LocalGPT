"""Live Web Search Service for Real-Time Knowledge Retrieval (Perplexity-Style)."""

import json
import ssl
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
from urllib.parse import urlparse


class WebSearchService:
    """Performs live web search without requiring external API keys."""

    @staticmethod
    def search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        results = []
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # 1. DuckDuckGo Lite Search
        try:
            post_data = urllib.parse.urlencode({"q": query}).encode("utf-8")
            req = urllib.request.Request(
                "https://lite.duckduckgo.com/lite/",
                data=post_data,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(req, context=ctx, timeout=6.0) as res:
                soup = BeautifulSoup(res.read(), "html.parser")
                links = soup.find_all("a", class_="result-link")
                snippets = soup.find_all("td", class_="result-snippet")

                for l, s in zip(links[:max_results], snippets[:max_results]):
                    title = l.get_text().strip()
                    raw_url = l.get("href", "").strip()
                    snippet = s.get_text().strip()

                    # Clean DuckDuckGo redirect if any
                    if "uddg=" in raw_url:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                        clean_url = parsed.get("uddg", [raw_url])[0]
                    else:
                        clean_url = raw_url

                    domain = urlparse(clean_url).netloc.replace("www.", "") if clean_url else "Web"
                    if title and snippet:
                        results.append({
                            "title": title,
                            "url": clean_url,
                            "snippet": snippet,
                            "domain": domain,
                            "favicon": f"https://www.google.com/s2/favicons?domain={domain}&sz=32",
                        })
                if results:
                    return results
        except Exception as e:
            print(f"[DuckDuckGo Lite search notice]: {e}")

        # 2. Wikipedia Search API Fallback
        try:
            wiki_search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote_plus(query)}&limit={max_results}&format=json"
            w_req = urllib.request.Request(wiki_search_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(w_req, context=ctx, timeout=4.0) as w_res:
                data = json.loads(w_res.read().decode("utf-8"))
                titles = data[1] if len(data) > 1 else []
                urls = data[3] if len(data) > 3 else []
                for t, u in zip(titles[:max_results], urls[:max_results]):
                    results.append({
                        "title": t,
                        "url": u,
                        "snippet": f"Encyclopedia article about {t} on Wikipedia.",
                        "domain": "wikipedia.org",
                        "favicon": "https://www.google.com/s2/favicons?domain=wikipedia.org&sz=32",
                    })
        except Exception as e:
            print(f"[Wikipedia fallback notice]: {e}")

        return results

    @classmethod
    def format_search_context(cls, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Fetch results and format prompt augmentation with citations."""
        results = cls.search(query, max_results=max_results)
        if not results:
            return {"context": "", "sources": []}

        context_lines = [f"### LIVE WEB SEARCH RESULTS FOR: '{query}'\n"]
        for idx, r in enumerate(results, 1):
            context_lines.append(
                f"[{idx}] {r['title']} ({r['domain']})\nURL: {r['url']}\nSummary: {r['snippet']}\n"
            )

        return {
            "context": "\n".join(context_lines),
            "sources": results,
        }
