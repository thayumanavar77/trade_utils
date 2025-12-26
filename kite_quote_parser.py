"""
Kite (Zerodha) WebSocket Quote message parser.
 - Decode Zerodha WebSocket binary messages using Kite v3 spec (Copilot Assisted)
Usage:
- Call parse_websocket_message(raw_bytes_or_hex) to get a list of parsed packets.
- Each packet is a dict with fixed quote fields and `depth` containing 5 bids + 5 asks.

Notes / assumptions:
- Network byte order (big-endian) for all fields.
- Prices are in paise for non-currency instruments -> divide by 100 to get INR (float).
- If instrument is a currency, set currency=True to use the currency scaling (divide by 10_000_000).
- Timestamps in the packet are treated as UNIX seconds by default. If your feed uses milliseconds,
  set timestamp_in_ms=True to get correct datetimes.
- Depth entry format (per Kite doc): quantity (int32), price (int32, paise), orders (int16), pad (int16).
  There are 10 entries (5 bids then 5 offers), each 12 bytes, occupying packet bytes [64:184).
- The parser is resilient to truncated packets and will parse whatever fields are available.
"""

from struct import unpack_from, calcsize
from datetime import datetime, timezone
from typing import Union, List, Dict, Any
import pprint


# helpers to read big-endian unsigned values
def _u16(data: bytes, offset: int) -> int:
    return unpack_from(">H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return unpack_from(">I", data, offset)[0]


def _i32(data: bytes, offset: int) -> int:
    return unpack_from(">i", data, offset)[0]


def _u16_safe(data: bytes, offset: int) -> Union[int, None]:
    try:
        return _u16(data, offset)
    except Exception:
        return None


def _u32_safe(data: bytes, offset: int) -> Union[int, None]:
    try:
        return _u32(data, offset)
    except Exception:
        return None


def _i32_safe(data: bytes, offset: int) -> Union[int, None]:
    try:
        return _i32(data, offset)
    except Exception:
        return None


def _to_price(value_int: int, currency: bool = False) -> float:
    """Convert the raw int price to human float price.
    - currency=False: divide by 100 (paise -> INR)
    - currency=True: divide by 10_000_000 (4 decimal places)
    """
    if value_int is None:
        return None
    if currency:
        return value_int / 10_000_000.0
    return value_int / 100.0


def _to_dt_from_unix(val: int, ms: bool = False) -> Union[datetime, None]:
    if val is None:
        return None
    if ms:
        return datetime.fromtimestamp(val / 1000.0, tz=timezone.utc)
    return datetime.fromtimestamp(val, tz=timezone.utc)


def parse_quote_packet(packet: bytes, timestamp_in_ms: bool = False, currency: bool = False) -> Dict[str, Any]:
    """Parse a single quote packet bytes (per Kite Quote Packet Structure).
    Expects packet to start at packet offset 0.
    """
    out: Dict[str, Any] = {}
    pkt_len = len(packet)

    # helper to safe-read
    def u32_off(o):
        return _u32_safe(packet, o) if o + 4 <= pkt_len else None

    def i32_off(o):
        return _i32_safe(packet, o) if o + 4 <= pkt_len else None

    def u16_off(o):
        return _u16_safe(packet, o) if o + 2 <= pkt_len else None

    # Fixed fields up to offset 64 (some may be missing if truncated)
    out["packet_length"] = pkt_len
    out["instrument_token"] = u32_off(0)

    # According to doc:
    # 0-4 instrument_token
    # 4-8 Last traded price
    # 8-12 Last traded quantity
    # 12-16 Average traded price
    # 16-20 Volume traded for the day
    # 20-24 Total buy quantity
    # 24-28 Total sell quantity
    # 28-32 Open price
    # 32-36 High price
    # 36-40 Low price
    # 40-44 Close price
    # 44-48 Last traded timestamp
    # 48-52 Open Interest
    # 52-56 OI day high
    # 56-60 OI day low
    # 60-64 Exchange timestamp
    fields = [
        ("last_traded_price_raw", 4),
        ("last_traded_qty", 8),
        ("avg_traded_price_raw", 12),
        ("volume", 16),
        ("total_buy_qty", 20),
        ("total_sell_qty", 24),
        ("open_price_raw", 28),
        ("high_price_raw", 32),
        ("low_price_raw", 36),
        ("close_price_raw", 40),
        ("last_traded_timestamp_raw", 44),
        ("open_interest", 48),
        ("oi_day_high", 52),
        ("oi_day_low", 56),
        ("exchange_timestamp_raw", 60),
    ]

    for name, off in fields:
        out[name] = u32_off(off)

    # Convert price raw fields to human readable prices (float) using _to_price
    for raw_name in ["last_traded_price_raw", "avg_traded_price_raw", "open_price_raw",
                     "high_price_raw", "low_price_raw", "close_price_raw"]:
        raw = out.get(raw_name)
        out[raw_name.replace("_raw", "")] = _to_price(raw, currency)

    # Convert timestamps
    out["last_traded_timestamp"] = _to_dt_from_unix(out.get("last_traded_timestamp_raw"), ms=timestamp_in_ms)
    out["exchange_timestamp"] = _to_dt_from_unix(out.get("exchange_timestamp_raw"), ms=timestamp_in_ms)

    # Depth area starts at offset 64; each entry = 12 bytes (quantity:int32, price:int32, orders:int16, pad:int16)
    depth_offset = 64
    depth_entries = []
    entry_size = 12
    max_entries = 10  # 5 bids + 5 offers

    for i in range(max_entries):
        base = depth_offset + i * entry_size
        if base + entry_size > pkt_len:
            # not enough bytes for another full entry, stop
            break
        qty = _u32_safe(packet, base)
        price_raw = _u32_safe(packet, base + 4)
        orders = _u16_safe(packet, base + 8)
        # pad = _u16_safe(packet, base + 10)  # ignored
        depth_entries.append({
            "quantity": qty,
            "price_raw": price_raw,
            "price": _to_price(price_raw, currency),
            "orders": orders
        })

    # Split into bids and offers (first 5 = bids, next 5 = offers)
    bids = depth_entries[:5]
    offers = depth_entries[5:10]

    out["depth"] = {
        "bids": bids,
        "offers": offers
    }

    # compute simple totals
    out["depth_totals"] = {
        "bid_qty_sum": sum(e["quantity"] or 0 for e in bids),
        "offer_qty_sum": sum(e["quantity"] or 0 for e in offers),
        "bid_orders_sum": sum(e["orders"] or 0 for e in bids),
        "offer_orders_sum": sum(e["orders"] or 0 for e in offers),
        "parsed_depth_entries": len(depth_entries)
    }

    return out


def parse_websocket_message(raw: Union[bytes, str], timestamp_in_ms: bool = False, currency: bool = False) -> List[Dict[str, Any]]:
    """
    Parse a single Kite WebSocket message (may contain multiple quote packets).
    Input: raw bytes or hex string.
    Returns a list of parsed quote packets (dicts).
    """
    if isinstance(raw, str):
        # assume hex string
        raw_bytes = bytes.fromhex(raw)
    else:
        raw_bytes = raw

    # At least 4 bytes for the message header
    if len(raw_bytes) < 4:
        raise ValueError("raw message too short to contain websocket header")

    offset = 0
    packet_count = _u16(raw_bytes, offset); offset += 2
    first_packet_len = _u16(raw_bytes, offset); offset += 2
    # note: spec repeats a length before each packet. We'll loop packet_count times and read lengths.
    packets_out = []

    # We already read the length for the first packet. Seek back 2 bytes to allow the loop to read each length uniformly.
    offset_for_lengths = 2
    offset = 4  # start of first packet (global)

    # We'll iterate reading [length][packet bytes] pairs packet_count times.
    offset = 2  # go back to read the first length including preceding packet_count bytes
    # Re-read packet_count boundary properly:
    packet_count = _u16(raw_bytes, 0)
    offset = 2
    for p_index in range(packet_count):
        if offset + 2 > len(raw_bytes):
            # length header not available -> truncated
            raise ValueError(f"Truncated message: packet length header for packet {p_index} missing")

        pkt_len = _u16(raw_bytes, offset)
        offset += 2
        if offset + pkt_len > len(raw_bytes):
            # truncated packet: we will parse the available bytes (best effort)
            packet_bytes = raw_bytes[offset: len(raw_bytes)]
            # Optionally: you can raise an error instead
        else:
            packet_bytes = raw_bytes[offset: offset + pkt_len]
        offset += pkt_len

        parsed = parse_quote_packet(packet_bytes, timestamp_in_ms=timestamp_in_ms, currency=currency)
        parsed["declared_packet_length"] = pkt_len
        parsed["packet_index_in_message"] = p_index
        packets_out.append(parsed)

    return packets_out


# Example usage with the hex you provided (the latest sample)
if __name__ == "__main__":
    sample_hex = (
        "000100b8071c5f070001b05d000000010001af54000017a1000001d5000002290001af450001b189"
        "0001ac160001b17a6941584000001e8200001e9500001d6669415841000000080001b04e000300"
        "00000000010001b04900010000000000010001b03f00010000000000030001b03a000300000000"
        "00010001b02b00010000000000020001b07100020000000000010001b076000100000000000100"
        "01b07b00010000000000020001b08000010000000000010001b08500010000"
    )
    parsed_packets = parse_websocket_message(sample_hex, timestamp_in_ms=False, currency=False)
    pprint.pprint(parsed_packets, width=160)
