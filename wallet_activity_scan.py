#!/usr/bin/env python3
import sys
import grpc
import hashlib
import json
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, List

from base58 import b58encode_check, b58decode_check
from eth_utils import keccak

# TRON proto stubs must be importable in your environment
from api.api_pb2 import NumberMessage, BytesMessage
from api.api_pb2_grpc import WalletStub
from core.contract.balance_contract_pb2 import TransferContract
from core.contract.asset_issue_contract_pb2 import TransferAssetContract

# ========== EDIT THESE CONSTANTS ==========
NODE = "127.0.0.1:50051"          # gRPC node
WALLET = "TEf3PqyXoquJnkefR8S1dd5838zwBbKvkg"  # Base58 (T...) OR hex (with/without 0x / 0x41 prefix)
START_BLOCK = 0
END_BLOCK = 76000000
MAX_WORKERS = 10
INCLUDE_EMPTY_BLOCKS = False
# =========================================

GRPC_MAX_MSG = 100 * 1024 * 1024  # 100MB for large blocks

# Precomputed topics for TRC20/TRC721/TRC1155
TRANSFER_TOPIC = keccak(text='Transfer(address,address,uint256)')
TRANSFER_SINGLE_TOPIC = keccak(text='TransferSingle(address,address,address,uint256,uint256)')
TRANSFER_BATCH_TOPIC = keccak(text='TransferBatch(address,address,address,uint256[],uint256[])')


# ---------------- Address helpers ----------------
def tron_b58(addr_bytes_21_or_20: bytes) -> str:
    if len(addr_bytes_21_or_20) == 20:
        return b58encode_check(b'\x41' + addr_bytes_21_or_20).decode()
    return b58encode_check(addr_bytes_21_or_20).decode()

def tron_b58_to_20(addr_b58: str) -> bytes:
    raw = b58decode_check(addr_b58)
    if len(raw) == 21 and raw[0] == 0x41:
        return raw[1:]
    elif len(raw) == 20:
        return raw
    raise ValueError("Unexpected Base58-decoded length")

def topic32_to_addr20(topic_bytes_32: bytes) -> bytes:
    return topic_bytes_32[-20:]

def normalize_wallet_input(s: str) -> Tuple[bytes, str]:
    s = s.strip()
    if s.startswith('T'):
        raw20 = tron_b58_to_20(s)
        return raw20, tron_b58(raw20)
    h = s.lower()
    if h.startswith('0x'):
        h = h[2:]
    try:
        b = bytes.fromhex(h)
    except ValueError:
        raise ValueError("Invalid wallet; use T... or hex.")
    if len(b) == 21 and b[0] == 0x41:
        raw20 = b[1:]
    elif len(b) == 20:
        raw20 = b
    else:
        raise ValueError("Hex must be 20 bytes (40 hex) or 21 bytes with 0x41 prefix.")
    return raw20, tron_b58(raw20)

def addr_bytes_equals(a: Optional[bytes], target20: bytes) -> bool:
    if not a:
        return False
    if len(a) == 21 and a[0] == 0x41:
        return a[1:] == target20
    if len(a) == 20:
        return a == target20
    if len(a) == 32:
        return a[-20:] == target20
    return False


# ---------------- General helpers ----------------
def txid_of(tx) -> str:
    raw = tx.raw_data.SerializeToString()
    return hashlib.sha256(raw).hexdigest()

def to_iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat()

def direction_for(from_match: bool, to_match: bool) -> str:
    if from_match and to_match:
        return "self"
    if from_match:
        return "out"
    if to_match:
        return "in"
    return "other"


# ---------------- Per-TX info (logs + internal) ----------------
def process_transaction_info(stub: WalletStub, txid: str, block_number: int, block_timestamp_ms: int, target20: bytes) -> List[dict]:
    filtered = []
    try:
        info = stub.GetTransactionInfoById(BytesMessage(value=bytes.fromhex(txid)))

        # Internal transfers
        for i, internal_tx in enumerate(getattr(info, "internal_transactions", [])):
            caller_match = addr_bytes_equals(internal_tx.caller_address, target20)
            to_match = addr_bytes_equals(internal_tx.transferTo_address, target20)
            if caller_match or to_match:
                filtered.append({
                    "protocol": "INTERNAL",
                    "type": "internal_transfer",
                    "direction": "caller" if caller_match and not to_match else
                                 "in" if to_match and not caller_match else
                                 "self",
                    "block_number": block_number,
                    "block_timestamp_ms": block_timestamp_ms,
                    "block_timestamp_iso": to_iso(block_timestamp_ms),
                    "txid": txid,
                    "index": i,
                    "caller": tron_b58(internal_tx.caller_address) if internal_tx.caller_address else None,
                    "to": tron_b58(internal_tx.transferTo_address) if internal_tx.transferTo_address else None,
                    "hash": internal_tx.hash.hex() if internal_tx.hash else None,
                })

        # Logs (TRC20/TRC721 + TRC1155)
        for lg in getattr(info, "log", []):
            if not getattr(lg, "topics", None):
                continue

            # TRC20/TRC721 Transfer
            if lg.topics[0] == TRANSFER_TOPIC:
                from20 = topic32_to_addr20(lg.topics[1])
                to20 = topic32_to_addr20(lg.topics[2])
                from_match = (from20 == target20)
                to_match = (to20 == target20)
                if from_match or to_match:
                    token_addr_b58 = tron_b58(lg.address if len(lg.address) in (20, 21) else lg.address[-20:])
                    value = int.from_bytes(lg.data.rjust(32, b'\x00'), 'big')
                    filtered.append({
                        "protocol": "TRC20/TRC721",
                        "type": "Transfer",
                        "direction": direction_for(from_match, to_match),
                        "block_number": block_number,
                        "block_timestamp_ms": block_timestamp_ms,
                        "block_timestamp_iso": to_iso(block_timestamp_ms),
                        "txid": txid,
                        "contract": token_addr_b58,
                        "from": tron_b58(from20),
                        "to": tron_b58(to20),
                        "value": str(value),
                    })

            # TRC1155 TransferSingle
            elif lg.topics[0] == TRANSFER_SINGLE_TOPIC:
                operator20 = topic32_to_addr20(lg.topics[1])
                from20 = topic32_to_addr20(lg.topics[2])
                to20 = topic32_to_addr20(lg.topics[3])
                involved = (operator20 == target20) or (from20 == target20) or (to20 == target20)
                if involved:
                    token_addr_b58 = tron_b58(lg.address if len(lg.address) in (20, 21) else lg.address[-20:])
                    token_id = int.from_bytes(lg.data[:32], 'big')
                    value = int.from_bytes(lg.data[32:64], 'big')
                    filtered.append({
                        "protocol": "TRC1155",
                        "type": "TransferSingle",
                        "direction": "operator" if operator20 == target20 else direction_for(from20 == target20, to20 == target20),
                        "block_number": block_number,
                        "block_timestamp_ms": block_timestamp_ms,
                        "block_timestamp_iso": to_iso(block_timestamp_ms),
                        "txid": txid,
                        "contract": token_addr_b58,
                        "operator": tron_b58(operator20),
                        "from": tron_b58(from20),
                        "to": tron_b58(to20),
                        "id": str(token_id),
                        "value": str(value),
                    })

            # TRC1155 TransferBatch (ids/values packed in data; keep raw hex)
            elif lg.topics[0] == TRANSFER_BATCH_TOPIC:
                operator20 = topic32_to_addr20(lg.topics[1])
                from20 = topic32_to_addr20(lg.topics[2])
                to20 = topic32_to_addr20(lg.topics[3])
                involved = (operator20 == target20) or (from20 == target20) or (to20 == target20)
                if involved:
                    token_addr_b58 = tron_b58(lg.address if len(lg.address) in (20, 21) else lg.address[-20:])
                    filtered.append({
                        "protocol": "TRC1155",
                        "type": "TransferBatch",
                        "direction": "operator" if operator20 == target20 else direction_for(from20 == target20, to20 == target20),
                        "block_number": block_number,
                        "block_timestamp_ms": block_timestamp_ms,
                        "block_timestamp_iso": to_iso(block_timestamp_ms),
                        "txid": txid,
                        "contract": token_addr_b58,
                        "operator": tron_b58(operator20),
                        "from": tron_b58(from20),
                        "to": tron_b58(to20),
                        "raw_data_hex": lg.data.hex(),
                    })

    except grpc.RpcError as e:
        print(f"RPC Error for TXID {txid} in block {block_number}: {e.details()}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error processing TXID {txid} in block {block_number}: {e}", file=sys.stderr)

    return filtered


# ---------------- Block scanner (wallet-filtered) ----------------
def scan_block_range_for_wallet(node: str, wallet20: bytes, wallet_b58: str,
                                start_block: int, end_block: int,
                                max_workers: int, include_empty_blocks: bool):
    channel = grpc.insecure_channel(node, options=[('grpc.max_receive_message_length', GRPC_MAX_MSG)])
    stub = WalletStub(channel)

    all_block_results = []
    summary = {
        "wallet": wallet_b58,
        "scanned_blocks": 0,
        "unique_txids": 0,
        "total_events": 0,
        "trx_sun_in": 0,
        "trx_sun_out": 0,
        "trc10_in": {},
        "trc10_out": {},
        "trc20_in": {},
        "trc20_out": {},
        "fees_sun": 0
    }
    seen_txids = set()

    for height in range(start_block, end_block + 1):
        try:
            block = stub.GetBlockByNum(NumberMessage(num=height))
            block_number = block.block_header.raw_data.number
            block_timestamp_ms = block.block_header.raw_data.timestamp

            block_result = {
                "block_number": block_number,
                "block_timestamp_ms": block_timestamp_ms,
                "block_timestamp_iso": to_iso(block_timestamp_ms),
                "wallet_tx_count": 0,
                "wallet_events_count": 0,
                "events": []
            }

            tx_ids: List[str] = []
            for tx in getattr(block, "transactions", []):
                txid = txid_of(tx)
                tx_ids.append(txid)

                for c in tx.raw_data.contract:
                    # TRX
                    if c.parameter.type_url.endswith('TransferContract'):
                        tc = TransferContract()
                        c.parameter.Unpack(tc)
                        from_match = addr_bytes_equals(tc.owner_address, wallet20)
                        to_match = addr_bytes_equals(tc.to_address, wallet20)
                        if from_match or to_match:
                            dirn = direction_for(from_match, to_match)
                            amt = int(tc.amount)
                            if dirn in ("in", "self"):
                                summary["trx_sun_in"] += amt
                            if dirn in ("out", "self"):
                                summary["trx_sun_out"] += amt
                            block_result["events"].append({
                                "protocol": "TRX",
                                "type": "Transfer",
                                "direction": dirn,
                                "block_number": block_number,
                                "block_timestamp_ms": block_timestamp_ms,
                                "block_timestamp_iso": to_iso(block_timestamp_ms),
                                "txid": txid,
                                "from": tron_b58(tc.owner_address),
                                "to": tron_b58(tc.to_address),
                                "amount_sun": str(amt),
                            })
                            if txid not in seen_txids:
                                seen_txids.add(txid)

                    # TRC10
                    elif c.parameter.type_url.endswith('TransferAssetContract'):
                        tac = TransferAssetContract()
                        c.parameter.Unpack(tac)
                        from_match = addr_bytes_equals(tac.owner_address, wallet20)
                        to_match = addr_bytes_equals(tac.to_address, wallet20)
                        if from_match or to_match:
                            try:
                                asset_name = tac.asset_name.decode('utf-8')
                            except Exception:
                                asset_name = tac.asset_name.hex()
                            dirn = direction_for(from_match, to_match)
                            amt = int(tac.amount)
                            if dirn in ("in", "self"):
                                summary["trc10_in"][asset_name] = summary["trc10_in"].get(asset_name, 0) + amt
                            if dirn in ("out", "self"):
                                summary["trc10_out"][asset_name] = summary["trc10_out"].get(asset_name, 0) + amt
                            block_result["events"].append({
                                "protocol": "TRC10",
                                "type": "TransferAsset",
                                "direction": dirn,
                                "block_number": block_number,
                                "block_timestamp_ms": block_timestamp_ms,
                                "block_timestamp_iso": to_iso(block_timestamp_ms),
                                "txid": txid,
                                "asset_name": asset_name,
                                "from": tron_b58(tac.owner_address),
                                "to": tron_b58(tac.to_address),
                                "amount": str(amt),
                            })
                            if txid not in seen_txids:
                                seen_txids.add(txid)

            # logs + internal
            if tx_ids:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(process_transaction_info, stub, txid, block_number, block_timestamp_ms, wallet20)
                        for txid in tx_ids
                    ]
                    for txid, f in zip(tx_ids, futures):
                        evts = f.result()
                        if not evts:
                            continue
                        for evt in evts:
                            block_result["events"].append(evt)
                            if evt["protocol"] == "TRC20/TRC721" and evt["type"] == "Transfer":
                                try:
                                    val = int(evt.get("value", "0"))
                                except Exception:
                                    val = 0
                                contract = evt.get("contract")
                                if contract:
                                    if evt["direction"] in ("in", "self"):
                                        summary["trc20_in"][contract] = summary["trc20_in"].get(contract, 0) + val
                                    if evt["direction"] in ("out", "self"):
                                        summary["trc20_out"][contract] = summary["trc20_out"].get(contract, 0) + val
                        if any(True for _ in evts) and txid not in seen_txids:
                            seen_txids.add(txid)

            if block_result["events"]:
                block_result["wallet_events_count"] = len(block_result["events"])
                block_result["wallet_tx_count"] = len({e["txid"] for e in block_result["events"]})
                all_block_results.append(block_result)
            elif include_empty_blocks:
                all_block_results.append(block_result)

            summary["scanned_blocks"] += 1
            summary["total_events"] += block_result["wallet_events_count"]

        except grpc.RpcError as e:
            print(f"RPC Error fetching block {height}: {e.details()}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error processing block {height}: {e}", file=sys.stderr)

    summary["unique_txids"] = len(seen_txids)

    out = {
        "wallet": wallet_b58,
        "start_block": start_block,
        "end_block": end_block,
        "summary": summary,
        "blocks": all_block_results
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[SUMMARY] Scanned blocks={summary['scanned_blocks']}, unique_tx={summary['unique_txids']}, events_total={summary['total_events']}")


# ---------------- Entry point ----------------
if __name__ == '__main__':
    wallet20, wallet_b58 = normalize_wallet_input(WALLET)
    scan_block_range_for_wallet(
        node=NODE,
        wallet20=wallet20,
        wallet_b58=wallet_b58,
        start_block=START_BLOCK,
        end_block=END_BLOCK,
        max_workers=MAX_WORKERS,
        include_empty_blocks=INCLUDE_EMPTY_BLOCKS
    )
