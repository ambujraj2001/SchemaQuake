from schemaquake.tools import (
    book, cancel, probe_schema, read_policies, search_flights, search_hotels,
)
from schemaquake.world import WorldState


def test_world_loads():
    world = WorldState.fresh()
    assert len(world.flights) >= 10
    assert len(world.hotels) >= 10
    assert "TravelWorld" in world.policies_md


def test_search_flights_filters():
    world = WorldState.fresh()
    res = search_flights(world, origin="BLR", destination="DEL", max_price=6000)
    assert res["results"], "expected at least one BLR-DEL flight under 6000"
    for f in res["results"]:
        assert f["from"] == "BLR" and f["to"] == "DEL"
        assert f["price"] <= 6000


def test_book_then_cancel_refundable():
    world = WorldState.fresh()
    res = search_flights(world, origin="BLR", destination="DEL", max_price=10000)
    target = next(f for f in res["results"] if f["refundable"])
    b = book(world, target["id"])
    assert b["status"] == "confirmed"
    assert b["refundable"] is True
    c = cancel(world, b["booking_id"])
    assert c["status"] == "refunded"


def test_book_nonrefundable_cannot_cancel():
    world = WorldState.fresh()
    res = search_flights(world, origin="BLR", destination="DEL", max_price=10000)
    target = next(f for f in res["results"] if not f["refundable"])
    b = book(world, target["id"])
    c = cancel(world, b["booking_id"])
    assert c["status"] == "denied"


def test_probe_schema_returns_defaults():
    world = WorldState.fresh()
    s = probe_schema(world)
    assert s["flight_price_field"] == "price"
    assert s["refundable_field"] == "refundable"
    assert s["price_unit"] == "rupees"


def test_search_hotels_by_city():
    world = WorldState.fresh()
    res = search_hotels(world, city="BOM")
    assert res["results"]
    for h in res["results"]:
        assert h["city"] == "BOM"


def test_read_policies_has_version():
    world = WorldState.fresh()
    p = read_policies(world)
    assert "Document version" in p["policies_md"]
