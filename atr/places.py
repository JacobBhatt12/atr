"""Keyless local-place search for the hosted connector, using OpenStreetMap.

Nominatim's public service asks clients to stay below one request per second. Searches are
serialized and cached in-process, which also makes repeated chatbot questions inexpensive.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
from collections import OrderedDict

NOMINATIM = "https://nominatim.openstreetmap.org/search"
MAX_CACHE = 256
CACHE_TTL = 30 * 60

# Nominatim understands OpenStreetMap categories better than conversational plurals.
# Keep the user's words when no category matches so named-place searches still work.
CATEGORIES = (
    (r"\b(coffee shops?|coffeehouses?|cafes?|cafés?)\b", "cafe"),
    (r"\b(restaurants?|places? to eat|food)\b", "restaurant"),
    (r"\b(bars?|pubs?|taverns?)\b", "bar"),
    (r"\b(grocery stores?|grocers?|supermarkets?)\b", "supermarket"),
    (r"\b(pharmacies|pharmacy|drugstores?|chemists?)\b", "pharmacy"),
    (r"\b(bookstores?|book shops?)\b", "bookstore"),
    (r"\b(hotels?|lodging|accommodation)\b", "hotel"),
    (r"\b(gas stations?|petrol stations?|fuel)\b", "fuel"),
    (r"\b(hospitals?)\b", "hospital"),
    (r"\b(urgent care|clinics?)\b", "clinic"),
    (r"\b(dentists?|dental clinics?)\b", "dentist"),
    (r"\b(gyms?|fitness centers?|fitness centres?)\b", "fitness centre"),
    (r"\b(parks?|playgrounds?)\b", "park"),
    (r"\b(museums?)\b", "museum"),
    (r"\b(movie theaters?|movie theatres?|cinemas?)\b", "cinema"),
    (r"\b(banks?)\b", "bank"),
    (r"\b(atms?)\b", "atm"),
    (r"\b(laundromats?|laundries?)\b", "laundry"),
    (r"\b(hair salons?|barbers?|barber shops?)\b", "hairdresser"),
    (r"\b(car repair|mechanics?|auto repair)\b", "car repair"),
)

_lock = asyncio.Lock()
_last_request = 0.0
_cache: OrderedDict[tuple[str, str, int], tuple[float, list[dict]]] = OrderedDict()
_geocode_cache: OrderedDict[str, tuple[float, dict | None]] = OrderedDict()


async def _nominatim_get(client, params: dict) -> list[dict]:
    """One policy-compliant request to the shared public Nominatim service."""
    global _last_request
    async with _lock:
        now = time.monotonic()
        await asyncio.sleep(max(0, 1.05 - (now - _last_request)))
        response = await client.get(NOMINATIM, params=params)
        _last_request = time.monotonic()
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, list) else []


async def geocode_location(client, location: str) -> dict | None:
    """Resolve a human location for other keyless tools such as weather."""
    location = re.sub(r"\s+", " ", location).strip(" ,.")
    if not location:
        raise ValueError("location is required")
    if len(location) > 160:
        raise ValueError("location is too long")
    key = location.casefold()
    now = time.monotonic()
    cached = _geocode_cache.get(key)
    if cached and now - cached[0] < CACHE_TTL:
        _geocode_cache.move_to_end(key)
        return cached[1]
    data = await _nominatim_get(client, {"q": location, "format": "jsonv2", "addressdetails": 1, "limit": 1})
    item = data[0] if data else None
    result = None
    if item:
        try:
            result = {"latitude": float(item["lat"]), "longitude": float(item["lon"]),
                      "name": item.get("display_name") or location,
                      "country_code": (item.get("address") or {}).get("country_code", "").lower()}
        except (KeyError, TypeError, ValueError):
            result = None
    _geocode_cache[key] = (time.monotonic(), result)
    _geocode_cache.move_to_end(key)
    while len(_geocode_cache) > MAX_CACHE:
        _geocode_cache.popitem(last=False)
    return result


def normalized_query(query: str) -> str:
    query = re.sub(r"\s+", " ", query).strip(" ,.")
    for pattern, category in CATEGORIES:
        if re.search(pattern, query, re.I):
            return category
    return query


def _address(item: dict) -> str:
    address = item.get("address") or {}
    street = " ".join(x for x in (address.get("house_number"), address.get("road")) if x)
    parts = [street, address.get("neighbourhood") or address.get("suburb"),
             address.get("city") or address.get("town") or address.get("village"),
             address.get("state"), address.get("postcode")]
    compact = ", ".join(dict.fromkeys(x for x in parts if x))
    return compact or str(item.get("display_name") or "")


def _distance_km(lat: float, lon: float, center: tuple[float, float] | None) -> float | None:
    if center is None:
        return None
    lat2, lon2 = center
    p1, p2 = math.radians(lat), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


async def search_places(client, query: str, location: str, limit: int = 8,
                        center: tuple[float, float] | None = None) -> list[dict]:
    """Return compact place records. `center` is optional and only used for distances."""
    query, location = query.strip(), location.strip()
    if not query or not location:
        raise ValueError("query and location are required")
    if len(query) > 120 or len(location) > 160:
        raise ValueError("query or location is too long")
    limit = max(1, min(int(limit), 10))
    key = (query.casefold(), location.casefold(), limit)
    now = time.monotonic()
    cached = _cache.get(key)
    if cached and now - cached[0] < CACHE_TTL:
        _cache.move_to_end(key)
        return cached[1]

    cached = _cache.get(key)
    now = time.monotonic()
    if cached and now - cached[0] < CACHE_TTL:
        _cache.move_to_end(key)
        return cached[1]
    raw = await _nominatim_get(client, {
            "q": f"{normalized_query(query)}, {location}",
            "format": "jsonv2",
            "addressdetails": 1,
            "extratags": 1,
            "namedetails": 1,
            "limit": limit,
        })

    results = []
    for item in raw if isinstance(raw, list) else []:
        try:
            lat, lon = float(item["lat"]), float(item["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        extra = item.get("extratags") or {}
        names = item.get("namedetails") or {}
        name = item.get("name") or names.get("name") or str(item.get("display_name") or "").split(",", 1)[0]
        osm_type, osm_id = item.get("osm_type"), item.get("osm_id")
        result = {
            "name": name or "Unnamed place",
            "category": str(item.get("type") or item.get("category") or "place").replace("_", " "),
            "address": _address(item),
            "latitude": lat,
            "longitude": lon,
            "map": f"https://www.openstreetmap.org/{osm_type}/{osm_id}" if osm_type and osm_id else
                   f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=17/{lat}/{lon}",
        }
        distance = _distance_km(lat, lon, center)
        if distance is not None:
            result["distance_km"] = round(distance, 2)
        for source, target in (("opening_hours", "hours"), ("phone", "phone"),
                               ("contact:phone", "phone"), ("website", "website"),
                               ("contact:website", "website")):
            if extra.get(source) and target not in result:
                result[target] = extra[source]
        results.append(result)

    _cache[key] = (time.monotonic(), results)
    _cache.move_to_end(key)
    while len(_cache) > MAX_CACHE:
        _cache.popitem(last=False)
    return results


def format_places(query: str, location: str, results: list[dict]) -> str:
    if not results:
        return (f"No OpenStreetMap results for {query!r} near {location}. Try a broader category "
                "(for example: cafe, restaurant, bookstore, pharmacy, hotel) or a more specific location.")
    lines = [f"{len(results)} OpenStreetMap results for {query!r} near {location}:"]
    for index, result in enumerate(results, 1):
        details = [result["category"], result["address"]]
        if "distance_km" in result:
            details.append(f"{result['distance_km']:g} km away")
        lines.append(f"{index}. {result['name']} — " + " · ".join(x for x in details if x))
        optional = [f"hours: {result['hours']}" for _ in [0] if result.get("hours")]
        optional += [f"phone: {result['phone']}" for _ in [0] if result.get("phone")]
        optional += [f"website: {result['website']}" for _ in [0] if result.get("website")]
        if optional:
            lines.append("   " + " · ".join(optional))
        lines.append(f"   map: {result['map']} · coordinates: {result['latitude']}, {result['longitude']}")
    lines.append("Data © OpenStreetMap contributors. Results may be incomplete and do not include ratings.")
    return "\n".join(lines)
