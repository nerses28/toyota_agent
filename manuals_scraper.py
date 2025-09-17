import os
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, date
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from requests.adapters import HTTPAdapter, Retry
from collections import defaultdict

API = "https://diva-api.tweddle.app"
PORTAL = "https://customerportal.tweddle-aws.eu"
REQUEST_TIMEOUT = 60

PRODUCT_LIMIT = 15
OUT_DIR = "./manuals"
USE_SIBLING_YEARS = True
MERGE_PRODUCTS = True

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/json",
    "Accept-Language": "en-GB,en;q=0.9",
})
session.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 502, 503, 504],
            allowed_methods=["GET"],
        )
    ),
)

def get_products() -> List[Dict[str, Any]]:
    r = session.get(f"{API}/pubhub/info/products", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    print("Products:", len(data))
    return data

# ---------------------- MERGE (collapse) logic ----------------------

def _parse_iso_date(d: Optional[str]) -> date:
    if not d:
        return date.min
    try:
        base = d.split("T", 1)[0]
        return datetime.fromisoformat(base).date()
    except Exception:
        return date.min

def _rank_for_merge(p: Dict[str, Any]) -> Tuple[date, int]:
    d = _parse_iso_date(p.get("lineOffDate"))
    y_raw = p.get("year")
    y = int(y_raw) if (y_raw is not None and str(y_raw).isdigit()) else -1
    return (d, y)

def merge_products_latest(all_products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[Any, Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for prod in all_products:
        key = (prod.get("brand"), prod.get("model"), prod.get("modelType"), prod.get("ngtdModelId"))
        groups[key].append(prod)

    merged: List[Dict[str, Any]] = []
    for key, items in groups.items():
        best = max(items, key=_rank_for_merge)
        merged.append(best)

    print(f"Merged products: {len(merged)} groups from {len(all_products)} originals")
    return merged

# ---------------------- Years selection ----------------------

def years_for_same_product(prod: Dict[str, Any], all_products: List[Dict[str, Any]], cap: int = 12) -> List[str]:
    key = (prod.get("brand"), prod.get("model"), prod.get("modelType"), prod.get("ngtdModelId"))
    years: List[int] = []
    for p in all_products:
        if (p.get("brand"), p.get("model"), p.get("modelType"), p.get("ngtdModelId")) == key:
            y = p.get("year")
            if y is not None and str(y).isdigit():
                years.append(int(y))
    years = sorted(set(years), reverse=True)
    return [str(y) for y in years[:cap]]

def pick_years_for_product(prod: Dict[str, Any], all_products: List[Dict[str, Any]]) -> List[str]:
    def year_from_prod(p: Dict[str, Any]) -> List[str]:
        y = p.get("year")
        return [str(int(y))] if (y is not None and str(y).isdigit()) else []

    if USE_SIBLING_YEARS:
        years = years_for_same_product(prod, all_products)
        return years if years else year_from_prod(prod)
    else:
        return year_from_prod(prod)

# ---------------------- Publications page parsing ----------------------

def build_publications_url(prod: Dict[str, Any], years: List[str], language: str = "en") -> str:
    params = [
        ("brand",        prod.get("brand") or ""),
        ("model",        prod.get("model") or ""),
        ("modelType",    (prod.get("modelType") or prod.get("model") or "")),
        ("ngtdModelId",  str(prod.get("ngtdModelId") or "")),
        ("language",     language),
    ]
    for i, y in enumerate(years):
        params.append((f"year[{i}]", y))
    q = "&".join(f"{k}={quote(v)}" for k, v in params if v)
    return f"{PORTAL}/publications?{q}"

def parse_next_data(html: str) -> Optional[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find("script", id="__NEXT_DATA__", type="application/json")
    if not tag:
        return None
    try:
        return json.loads(tag.text)
    except Exception:
        return None

def collect_publications(next_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []

    def visit(node: Any):
        if isinstance(node, dict):
            if "partNumber" in node:
                found.append({
                    "partNumber":      node.get("partNumber"),
                    "publicationType": node.get("publicationType") or node.get("type") or node.get("category"),
                    "language":        node.get("language") or node.get("lang") or node.get("locale"),
                    "title":           node.get("title") or node.get("name") or node.get("label"),
                    "lineOffDate":     node.get("lineOffDate") or node.get("effectiveDate") or "",
                    "modelType":       node.get("modelType") or node.get("model") or "",
                    "ngtdModelId":     node.get("ngtdModelId") or node.get("modelId") or "",
                    "year":            node.get("year"),
                })
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(next_data)

    uniq: Dict[str, Dict[str, Any]] = {}
    for x in found:
        pn = x.get("partNumber")
        if pn:
            uniq[pn] = x
    return list(uniq.values())

def pick_latest_en_om(pubs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    def norm_date(d: str) -> str:
        return (d or "").split("T")[0]

    oms = [p for p in pubs
           if (p.get("publicationType") == "OM")
           and isinstance(p.get("language"), str)
           and p["language"].lower().startswith("en")]
    if not oms:
        return None
    oms.sort(key=lambda p: norm_date(p.get("lineOffDate") or ""), reverse=True)
    return oms[0]

def get_pdf_link(pub: Dict[str, Any]) -> Optional[str]:
    pn = pub.get("partNumber")
    model_type = pub.get("modelType")
    line_off = (pub.get("lineOffDate") or "").split("T")[0]
    if not (pn and model_type and line_off):
        print("Missing keys for pdfLink:", {"partNumber": pn, "modelType": model_type, "lineOffDate": line_off})
        return None

    params = {"partNumber": pn, "modelType": model_type, "lineOffDate": line_off}
    from requests import Request
    prepared = Request("GET", f"{API}/publications/content/pdfLink", params=params).prepare()
    print("pdfLink request:", prepared.url)
    r = session.send(prepared, timeout=REQUEST_TIMEOUT)

    if r.status_code in (400, 404, 500):
        try:
            print("pdfLink error:", r.status_code, r.json())
        except Exception:
            print("pdfLink error:", r.status_code, r.text[:200])
        return None

    r.raise_for_status()
    data = r.json()
    print("pdfLink response:", json.dumps(data, indent=2))
    return data.get("url")

# ---------------------- File naming & download ----------------------

def safe_filename(s: str) -> str:
    keep = "._- "
    cleaned = "".join(c if (c.isalnum() or c in keep) else "_" for c in (s or ""))
    cleaned = cleaned.strip("_ ")[:200]
    return cleaned or "manual"

def preferred_year(pub: Dict[str, Any], prod: Dict[str, Any]) -> Optional[str]:
    for candidate in (pub.get("year"), prod.get("year")):
        if candidate is not None and str(candidate).isdigit():
            return str(int(candidate))
    line_off = (pub.get("lineOffDate") or "")
    if len(line_off) >= 4 and line_off[:4].isdigit():
        return line_off[:4]
    return None

def make_named_pdf(brand: str, model: str, model_type: str, year: Optional[str], part_number: str) -> str:
    segments = [brand, model, model_type, year, part_number]
    segments = [safe_filename(x) for x in segments if x and safe_filename(x)]
    base = ".".join(segments) if segments else "manual"
    base = base.replace(' ', '_')
    return f"{base}.pdf"

def download_pdf(url: str, prod: Dict[str, Any], pub: Dict[str, Any]) -> Optional[str]:
    os.makedirs(OUT_DIR, exist_ok=True)

    brand = prod.get("brand") or ""
    model = prod.get("model") or ""
    model_type = (pub.get("modelType") or prod.get("modelType") or prod.get("model") or "")
    year = preferred_year(pub, prod)
    pn = pub.get("partNumber") or ""

    name = make_named_pdf(brand, model, model_type, year, pn)
    path = os.path.join(OUT_DIR, name)

    if os.path.exists(path):
        print(f"Skip download (exists): {path}")
        return path

    print(f"Downloading PDF -> {path}")
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "pdf" not in ctype.lower():
            print(f"Warning: unexpected Content-Type: {ctype}")
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 64):
                if chunk:
                    f.write(chunk)
    print("Saved:", path)
    return path

# ---------------------- main ----------------------

def main():
    all_products = get_products()

    if MERGE_PRODUCTS:
        base_list = merge_products_latest(all_products)
    else:
        base_list = all_products

    products = base_list[:PRODUCT_LIMIT] if PRODUCT_LIMIT else base_list

    for idx, p in enumerate(products, 1):
        years = pick_years_for_product(p, all_products)
        pub_url = build_publications_url(p, years=years, language="en")

        print(f"\n[{idx}/{len(products)}] {p.get('brand')} | {p.get('model')} | {p.get('modelType')} | years={years}")
        print("Publications page:", pub_url)

        html = session.get(pub_url, timeout=REQUEST_TIMEOUT).text
        next_data = parse_next_data(html)
        if not next_data:
            print("No __NEXT_DATA__ found")
            continue

        pubs = collect_publications(next_data)
        print(f"Publications in page: {len(pubs)}")

        om = pick_latest_en_om(pubs)
        if not om:
            print("No EN Owner's Manual on this page.")
            continue

        print("\nSelected OM publication:")
        print(json.dumps(om, indent=2))

        pdf_url = get_pdf_link(om)
        print("Direct PDF URL:", pdf_url)
        if not pdf_url:
            continue

        download_pdf(pdf_url, p, om)

if __name__ == "__main__":
    main()
