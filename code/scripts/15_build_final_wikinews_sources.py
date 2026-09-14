# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Build the frozen 60-source Wikinews input from public MediaWiki metadata.

Selection is deterministic: six broad Wikinews categories, ten eligible pages
per category, fixed seed, pilot titles excluded.  Target actors are conservative
heuristic candidates and are explicitly marked for audit before final calls.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.parse
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline_common import ROOT, jsonl_read, jsonl_write_new, word_count, write_manifest


API = "https://en.wikinews.org/w/api.php"
CATEGORIES = (
    ("Environment", "environment"),
    ("Politics and conflicts", "politics"),
    ("Economy and business", "economy"),
    ("Science and technology", "science"),
    ("Health", "health"),
    ("Crime and law", "law"),
)
EXCLUDED_TITLE = re.compile(r"\b(editorial|opinion|satire|liveblog|sports score|gossip)\b", re.I)
GENERIC = {"the", "a", "an", "on", "in", "after", "before", "according", "wikinews"}
VERB_HINTS = re.compile(
    r"\b(approved|announced|arrested|called|confirms?|criticized|declared|died|filed|found|has|have|is|met|ordered|plans?|said|signed|supports?|was|were|will|won)\b",
    re.I,
)

# Audited against each article's title and lead; frozen for the final run.
ACTOR_OVERRIDES = {
    "S011": "Victorian Government", "S012": "Church of Scientology",
    "S013": "Japanese government and asbestos manufacturers", "S014": "Philippine authorities and volunteers",
    "S015": "Lou Jost", "S016": "New Jersey school officials", "S017": "Tuam 2 Ltd",
    "S018": "Tropical Storm Danielle", "S019": "Asian ministers", "S020": "Australian supermarkets",
    "S021": "Joe Arpaio", "S022": "women protesters in Goma", "S023": "Australian Government",
    "S024": "Thai anti-coup protesters", "S025": "Danish security officials", "S026": "Democrats.com",
    "S027": "Chinese government", "S028": "Andy Martin", "S029": "Canadian military leadership",
    "S030": "People's Alliance for Democracy", "S031": "Corus", "S032": "Savarino Companies",
    "S033": "Taiwan External Trade Development Council", "S034": "AutoTronics Taipei organizers",
    "S035": "Australian transport investigators", "S036": "Bach Technology", "S037": "L. Alan Winters",
    "S038": "Hewlett-Packard", "S039": "U.S. Department of Transportation", "S040": "JJB Sports",
    "S041": "Roger D. Kornberg", "S042": "Wikimedia Foundation", "S043": "Taiwan Design Center",
    "S044": "Samsung Electronics", "S045": "Taiwan gaming event organizers", "S046": "NASA",
    "S047": "British Association of Dermatologists", "S048": "George Smoot and John Mather",
    "S049": "Anonymous", "S050": "Port Authority of New York and New Jersey", "S051": "World Bank",
    "S052": "Taiwan Sports Affairs Council", "S053": "Center for Global Development",
    "S054": "Stephen Schoenthaler", "S055": "international twin-study researchers",
    "S056": "University of Southern California", "S057": "World Bank", "S058": "Scottish Government",
    "S059": "World Bank", "S060": "KHS Mountain Bike Corporation", "S061": "Andrew Martinez",
    "S062": "Iraqi security forces", "S063": "U.S. federal agents", "S064": "Maribel Cuevas",
    "S065": "Garda S�och�na", "S066": "Premise Media", "S067": "Matthew Wright",
    "S068": "U.S. House of Representatives", "S069": "Kevin Underwood", "S070": "Kunming attackers",
}


def api_get(params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode({"maxlag": "5", **params, "format": "json"})
    request = urllib.request.Request(API + "?" + query, headers={
        "User-Agent": "social-framing-inversion-research/1.0 (reproducible corpus builder)",
        "Accept": "application/json",
    })
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 20 * (attempt + 1)
            time.sleep(delay)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def category_titles(category: str) -> list[str]:
    payload = api_get({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
                       "cmlimit": "50", "cmtype": "page"})
    return [item["title"] for item in payload.get("query", {}).get("categorymembers", [])]


def fetch_pages(titles: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch current revision content in bounded batches.

    The ``extracts`` property silently lowers whole-article requests to one
    page on this endpoint.  Revision content has no such limitation, so we
    retrieve the current wikitext in batches and deterministically strip the
    small amount of Wikinews markup needed for the source text.
    """
    pages: dict[str, dict[str, Any]] = {}
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        payload = api_get({
            "action": "query", "titles": "|".join(batch), "prop": "revisions|info",
            "rvprop": "ids|timestamp|content", "rvslots": "main",
            "formatversion": "2", "inprop": "url", "redirects": "1",
        })
        raw_pages = payload.get("query", {}).get("pages", [])
        if isinstance(raw_pages, dict):
            raw_pages = list(raw_pages.values())
        for page in raw_pages:
            revisions = page.get("revisions") or []
            revision = revisions[0] if revisions else {}
            slots = revision.get("slots") or {}
            main = slots.get("main") or {}
            wikitext = str(main.get("content") or revision.get("content") or "")
            if page.get("missing") or not wikitext:
                continue
            text = wikitext_to_plain(wikitext)
            pages[page["title"]] = {"pageid": page.get("pageid"), "title": page["title"],
                                     "text": text, "fullurl": page.get("fullurl"),
                                     "revision_id": str(revision.get("revid", "")),
                                     "timestamp": revision.get("timestamp")}
        time.sleep(0.4)
    return pages


def wikitext_to_plain(wikitext: str) -> str:
    """Convert the common Wikinews markup to deterministic plain text."""
    text = re.sub(r"<!--.*?-->", " ", wikitext, flags=re.S)
    text = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<ref\b[^>]*/>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove nested-free templates repeatedly; most remaining templates are
    # infoboxes or citation helpers and should not enter the article text.
    for _ in range(8):
        new = re.sub(r"\{\{[^{}]*\}\}", " ", text, flags=re.S)
        if new == text:
            break
        text = new
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)
    text = re.sub(r"\[\[([^:\]|]+)\|([^]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^:\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[[^:\]]*:[^]]+\]\]", " ", text)
    text = re.sub(r"\[https?://\S+\s+([^]]+)\]", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"^\s*[=*]{2,6}\s*(.*?)\s*[=*]{2,6}\s*$", r"\1", text, flags=re.M)
    text = re.sub(r"^\s*[*#:;]+\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-]{4,}\s*$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_revisions(titles: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch revision ids/timestamps for selected pages in one batched request."""
    payload = api_get({"action": "query", "titles": "|".join(titles),
                       "prop": "revisions", "rvprop": "ids|timestamp", "redirects": "1"})
    revisions: dict[str, dict[str, Any]] = {}
    for page in payload.get("query", {}).get("pages", {}).values():
        revision = (page.get("revisions") or [{}])[0]
        revisions[page.get("title")] = {"revision_id": str(revision.get("revid", "")),
                                         "timestamp": revision.get("timestamp")}
    return revisions


def infer_target_actor(title: str, text: str) -> str:
    first = re.split(r"(?<=[.!?])\s+|\n+", text.strip())[0]
    first = re.sub(r"^(?:[A-Z][a-z]+,\s+)?\d{1,2}\s+\w+\s+\d{4}\s*", "", first).strip()
    spans = re.findall(r"\b(?:[A-Z][A-Za-z0-9'’&.-]*)(?:\s+[A-Z][A-Za-z0-9'’&.-]*){0,5}", first)
    spans = [span.strip(" ,;:") for span in spans if span.lower() not in GENERIC and len(span) > 2]
    if spans:
        spans.sort(key=lambda span: (len(span.split()), len(span)), reverse=True)
        return spans[0]
    head = re.split(r"\s+(?:has|have|is|was|were|said|announced|approved|after|before)\s+", title, maxsplit=1, flags=re.I)[0]
    return head.strip(" :,-") or title


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic final Wikinews source set.")
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.count != 60:
        raise SystemExit("The frozen final experiment requires exactly 60 sources.")
    output = ROOT / "data" / "raw" / "full_sources.jsonl"
    if output.exists() and not args.resume:
        raise SystemExit(f"Refusing to overwrite {output}; use --resume")
    pilot_titles = {row.get("title") for row in jsonl_read(ROOT / "data" / "development" / "sources.jsonl")}
    category_pages_map: dict[str, dict[str, dict[str, Any]]] = {}
    for category, _ in CATEGORIES:
        titles = category_titles(category)
        if not titles:
            raise SystemExit(f"Category {category!r} returned no page titles")
        category_pages_map[category] = fetch_pages(titles)
    eligible_by_category: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for category, _ in CATEGORIES:
        for title, page in category_pages_map[category].items():
            if title in pilot_titles or EXCLUDED_TITLE.search(title):
                continue
            if word_count(page["text"]) < 180 or len(page["text"]) < 900:
                continue
            eligible_by_category[category][title] = page
    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    used: set[str] = set(pilot_titles)
    category_counts: dict[str, int] = {}
    for category, domain in CATEGORIES:
        candidates = [title for title in eligible_by_category[category] if title not in used]
        candidates.sort()
        rng.shuffle(candidates)
        chosen = candidates[:10]
        if len(chosen) < 10:
            raise SystemExit(f"Category {category!r} has only {len(chosen)} eligible non-pilot pages")
        category_counts[category] = len(chosen)
        for title in chosen:
            used.add(title)
            page = eligible_by_category[category][title]
            selected.append({
                "source_id": f"S{len(selected) + 11:03d}", "title": title,
                "provenance_url": "", "publication_date": "", "license": "CC BY 2.5",
                "raw_article_text": page["text"],
                "revision_id": page.get("revision_id", ""),
                "_revision_timestamp": page.get("timestamp"),
                "_fullurl": page.get("fullurl"),
                "source_type": "article", "domain": domain,
                "target_actor": ACTOR_OVERRIDES.get(f"S{len(selected) + 11:03d}", infer_target_actor(title, page["text"])),
                "selection_category": category, "selection_seed": args.seed,
                "target_actor_audit_status": "manual_audited",
            })
    for row in selected:
        row["publication_date"] = (row.pop("_revision_timestamp", "") or "")[:10]
        row.pop("_fullurl", None)
        row["provenance_url"] = f"https://en.wikinews.org/w/index.php?title={urllib.parse.quote(row['title'].replace(' ', '_'))}&oldid={row['revision_id']}"
    if len(selected) != args.count or any(not row["revision_id"] for row in selected):
        raise SystemExit(f"Selected {len(selected)} sources, expected {args.count}")
    jsonl_write_new(output, selected, resume=args.resume)
    manifest = write_manifest("15_build_final_wikinews_sources", "full", {
        "output": str(output.relative_to(ROOT)), "source_count": len(selected),
        "seed": args.seed, "categories": category_counts, "pilot_excluded": sorted(pilot_titles),
        "target_actor_audit_status": "manual_audited",
        "selection_rule": "six Wikinews categories, ten eligible non-pilot pages per category, deterministic shuffle",
    })
    print(f"Built {len(selected)} sources -> {output}; manifest={manifest}")


if __name__ == "__main__":
    main()
