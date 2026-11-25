"""Part 2 chain fetchers (imported by fetch_staking.py): polkadot, polygon, bsc,
celo, flow, ethereum, berachain."""
import base64
import datetime as dt
import json
import time

from fetch_staking import (TODAY, CM_SRC, coinmetrics_monthly, enc_u, eth_call, get,
                           post, quarter_starts, rpc, row, u256, S)


# ------------------------------------------------------------- xxhash64 (pure python)
_P1, _P2, _P3, _P4, _P5 = (11400714785074694791, 14029467366897019727, 1609587929392839161,
                           9650029242287828579, 2870177450012600261)
_M = (1 << 64) - 1


def _rotl(x, r):
    return ((x << r) | (x >> (64 - r))) & _M


def _round(acc, lane):
    acc = (acc + lane * _P2) & _M
    return (_rotl(acc, 31) * _P1) & _M


def xxh64(data: bytes, seed=0):
    n = len(data); i = 0
    if n >= 32:
        v = [(seed + _P1 + _P2) & _M, (seed + _P2) & _M, seed & _M, (seed - _P1) & _M]
        while i + 32 <= n:
            for k in range(4):
                v[k] = _round(v[k], int.from_bytes(data[i + 8 * k:i + 8 * k + 8], "little"))
            i += 32
        h = (_rotl(v[0], 1) + _rotl(v[1], 7) + _rotl(v[2], 12) + _rotl(v[3], 18)) & _M
        for k in range(4):
            h ^= _round(0, v[k]); h = (h * _P1 + _P4) & _M
    else:
        h = (seed + _P5) & _M
    h = (h + n) & _M
    while i + 8 <= n:
        h ^= _round(0, int.from_bytes(data[i:i + 8], "little"))
        h = (_rotl(h, 27) * _P1 + _P4) & _M; i += 8
    if i + 4 <= n:
        h ^= (int.from_bytes(data[i:i + 4], "little") * _P1) & _M
        h = (_rotl(h, 23) * _P2 + _P3) & _M; i += 4
    while i < n:
        h ^= (data[i] * _P5) & _M
        h = (_rotl(h, 11) * _P1) & _M; i += 1
    h ^= h >> 33; h = (h * _P2) & _M; h ^= h >> 29; h = (h * _P3) & _M; h ^= h >> 32
    return h


def twox64concat(b: bytes):
    return xxh64(b, 0).to_bytes(8, "little").hex() + b.hex()


# ------------------------------------------------------------- Polkadot
# storage prefixes = twox128(pallet) ++ twox128(item)
K_ACTIVE_ERA = "0x5f3e4907f716ac89b6347d15ececedca487df464e44a534ba6b0cbb32407b587"
K_ERAS_TOTAL_STAKE = "0x5f3e4907f716ac89b6347d15ececedcaa141c4fe67c2d11f4a10c6aca7a79a04"
K_ERAS_VAL_REWARD = "0x5f3e4907f716ac89b6347d15ececedca7e6ed2ee507c7b4441d59e4ded44b8a2"
K_TOTAL_ISSUANCE = "0xc2261276cc9d1f8598ea4b6a74b15c2f57c875e4cff74148e4628f264b974c80"
K_NOW = "0xf0c365c3cf59d671eb72da0e7a4113c49f1f0515f462cdcf84e0f1d6045dfcbb"
RELAY = "https://rpc.polkadot.io"
AH = "https://polkadot-asset-hub-rpc.polkadot.io"


def _le(hexstr):
    return int.from_bytes(bytes.fromhex(hexstr[2:]), "little") if hexstr else None


def _sub_block_at(url, ts_ms):
    head = rpc(url, "chain_getHeader")
    hn = int(head["number"], 16)
    now = lambda n: _le(rpc(url, "state_getStorage", [K_NOW, rpc(url, "chain_getBlockHash", [n])]))
    t_head = now(hn)
    n = int(hn - (t_head - ts_ms) / 6000)
    for _ in range(8):
        n = max(1, min(n, hn))
        t = now(n)
        if abs(t - ts_ms) < 30 * 60 * 1000:
            break
        n = int(n - (t - ts_ms) / 6000)
    return rpc(url, "chain_getBlockHash", [n]), t


def _dot_point(url, bh):
    ae = rpc(url, "state_getStorage", [K_ACTIVE_ERA, bh])
    if not ae:
        return None
    era = int.from_bytes(bytes.fromhex(ae[2:10]), "little")
    stake = _le(rpc(url, "state_getStorage", [K_ERAS_TOTAL_STAKE + twox64concat(era.to_bytes(4, "little")), bh]))
    rew = _le(rpc(url, "state_getStorage", [K_ERAS_VAL_REWARD + twox64concat((era - 1).to_bytes(4, "little")), bh]))
    iss = _le(rpc(url, "state_getStorage", [K_TOTAL_ISSUANCE, bh]))
    return era, stake / 1e10, (rew or 0) / 1e10, iss / 1e10


def polkadot():
    rows = []
    pts = [(d, d.timestamp() * 1000) for d in quarter_starts("2021-01")]
    pts.append((None, time.time() * 1000))
    for d, ts in pts:
        res = None
        for url in (RELAY, AH):
            try:
                bh, t = _sub_block_at(url, ts)
                res = _dot_point(url, bh)
                if res and res[1]:
                    break
                res = None
            except Exception as ex:
                print("polkadot", d, url, str(ex)[:120])
        if not res:
            continue
        era, stake, rew, iss = res
        date = dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).date()
        rows.append(row("polkadot", date, stake / iss, "total issuance (Balances.TotalIssuance)",
                        rew * 365 / stake if rew else None, rew * 365 / iss if rew else None, None,
                        f"{url} state_getStorage Staking.ActiveEra/ErasTotalStake/ErasValidatorReward, "
                        f"Balances.TotalIssuance @ {bh}",
                        f"era {era}; r = ErasValidatorReward(era-1)*365/ErasTotalStake(era) (1 era = 1 day); "
                        "excludes treasury share of era payout; tx fees go 80% treasury/20% author (fee_share ~0)"
                        + ("; staking on Asset Hub after Nov-2025 migration" if url == AH else "")))
    return rows


# ------------------------------------------------------------- EVM block-by-time (secant)
def evm_block_at_fast(url, ts, head=None):
    if head is None:
        head = int(rpc(url, "eth_blockNumber"), 16)
    tsof = lambda n: int(rpc(url, "eth_getBlockByNumber", [hex(n), False])["timestamp"], 16)
    th = tsof(head)
    lo, tlo = 1, tsof(1)
    hi, thi = head, th
    if ts <= tlo:
        return 1
    if ts > th:
        return None
    for _ in range(40):
        if hi - lo <= 1:
            return hi
        g = lo + int((ts - tlo) / max(1, thi - tlo) * (hi - lo))
        g = min(max(g, lo + 1), hi - 1)
        tg = tsof(g)
        if abs(tg - ts) < 600:
            return g
        if tg < ts:
            lo, tlo = g, tg
        else:
            hi, thi = g, tg
        time.sleep(0.2)
    return hi


# ------------------------------------------------------------- Polygon (StakeManager on Ethereum)
ETH_RPCS = ["https://eth.drpc.org", "https://ethereum-rpc.publicnode.com"]
STAKE_MANAGER = "0x5e3Ef299fDDf15eAa0432E6e66473ace8c13D908"
POL = "0x455e53CBB86018Ac2B8092FdCd39d8444aFFC3F6"
MATIC = "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0"


def polygon():
    url = ETH_RPCS[0]
    rows = []
    head = int(rpc(url, "eth_blockNumber"), 16)
    pts = quarter_starts("2021-01") + [None]
    prev = None
    for d in pts:
        try:
            blk = head if d is None else evm_block_at_fast(url, d.timestamp(), head)
            stake = u256(eth_call(url, STAKE_MANAGER, "0xa4769071", blk)) / 1e18
            try:
                pol = u256(eth_call(url, POL, "0x18160ddd", blk)) / 1e18
            except Exception:
                pol = 0
            sup = pol if pol > 0 else 1e10
            date = TODAY if d is None else d.date()
            rows.append(dict(date=date, blk=blk, stake=stake, sup=sup, pol=pol))
            time.sleep(0.5)
        except Exception as ex:
            print("polygon", d, str(ex)[:150])
    out = []
    for i, x in enumerate(rows):
        E = r = None
        note = "theta = StakeManager.currentValidatorSetTotalStake / token total supply (MATIC 10B fixed before POL)"
        # POL emission: annualised growth of POL totalSupply over the previous point(s), half to stakers
        d1_ = dt.date.fromisoformat(str(x["date"]))
        cands = [k for k in range(i) if rows[k]["pol"] > 0 and
                 (d1_ - dt.date.fromisoformat(str(rows[k]["date"]))).days >= 75]
        # prefer the point closest to one year earlier
        j = min(cands, key=lambda k: abs((d1_ - dt.date.fromisoformat(str(rows[k]["date"]))).days - 365)) if cands else None
        if x["pol"] > 0 and j is not None:
            d0 = dt.date.fromisoformat(str(rows[j]["date"])); d1 = dt.date.fromisoformat(str(x["date"]))
            yrs = (d1 - d0).days / 365.25
            if yrs > 0.2 and rows[j]["pol"] > 5e9:  # skip the 2023-24 MATIC->POL migration ramp
                g = (x["pol"] / rows[j]["pol"]) ** (1 / yrs) - 1
                E = 0.5 * g
                r = E * x["sup"] / x["stake"]
                note += (f"; POL supply growth {g:.4f}/yr since {d0}; E = 50% of emission "
                         "(EmissionManager split stakers/treasury, assumption from PIP-26 design)")
        out.append(row("polygon", x["date"], x["stake"] / x["sup"], "total supply (POL.totalSupply; 10B MATIC pre-migration)",
                       r, E, None, f"{url} eth_call StakeManager {STAKE_MANAGER} currentValidatorSetTotalStake, POL totalSupply @block {x['blk']}",
                       note))
    return out


# ------------------------------------------------------------- BNB Smart Chain (StakeHub)
BSC = "https://bsc-dataseed.bnbchain.org"
STAKEHUB = "0x0000000000000000000000000000000000002002"


def _dyn_addr_array(h, word_off):
    words = h[2:]
    off = int(words[64 * word_off:64 * (word_off + 1)], 16) // 32
    n = int(words[64 * off:64 * (off + 1)], 16)
    return ["0x" + words[64 * (off + 1 + k) + 24:64 * (off + 2 + k)] for k in range(n)]


def bsc():
    res = eth_call(BSC, STAKEHUB, "0xbff02e20" + enc_u(0) + enc_u(200))
    credits = _dyn_addr_array(res, 1)
    tot = 0; price = {}
    for c in credits:
        tot += u256(eth_call(BSC, c, "0x15d1f898")) / 1e18
        price[c] = u256(eth_call(BSC, c, "0x91faf0b4" + enc_u(10 ** 18))) / 1e18
    cg = get("https://api.coingecko.com/api/v3/coins/binancecoin",
             params=dict(localization="false", tickers="false", market_data="true",
                         community_data="false", developer_data="false"))["market_data"]
    sup = float(cg["total_supply"]); circ = float(cg["circulating_supply"])
    # r from credit-share price growth over ~30 days (needs archive state; may be unavailable)
    r = None; rnote = "r not available (no free archive state for share-price growth)"
    try:
        arch = "https://bsc.drpc.org"
        head = int(rpc(arch, "eth_blockNumber"), 16)
        b0 = evm_block_at_fast(arch, time.time() - 30 * 86400, head)
        num = den = 0
        for c in credits:
            p0 = u256(eth_call(arch, c, "0x91faf0b4" + enc_u(10 ** 18), b0)) / 1e18
            w = u256(eth_call(arch, c, "0x15d1f898", b0)) / 1e18
            if p0 > 0 and w > 0:
                num += w * (price[c] / p0 - 1); den += w
            time.sleep(0.3)
        r = (1 + num / den) ** (365.25 / 30) - 1
        rnote = f"r = stake-weighted growth of credit-contract BNB/share over 30d (net of validator commission) via {arch}"
    except Exception as ex:
        print("bsc r", str(ex)[:150])
    return [row("bsc", TODAY, tot / circ, "circulating supply (CoinGecko /coins/binancecoin)",
                r, 0.0, 1.0,
                f"{BSC} eth_call StakeHub {STAKEHUB} getValidators + credit totalPooledBNB ({len(credits)} validators); "
                "supply https://api.coingecko.com/api/v3/coins/binancecoin",
                f"theta vs total supply={tot/sup:.4f}; no inflation: rewards are gas fees (E=0, fee_share=1); {rnote}")]


# ------------------------------------------------------------- Celo
CELO = "https://forno.celo.org"
REGISTRY = "0x000000000000000000000000000000000000ce10"
GOLD = "0x471EcE3750Da237f93B8E339c536989b8978a438"


def _registry(name, blk="latest"):
    b = name.encode().hex()
    data = "0x853db323" + enc_u(32) + enc_u(len(name)) + b.ljust(64, "0")
    return "0x" + eth_call(CELO, REGISTRY, data, blk)[-40:]


def celo():
    rows = []
    head = int(rpc(CELO, "eth_blockNumber"), 16)
    # forno serves state only after the L2 migration (block 31056500, 2025-03-26); earlier L1 state is not available
    for d in quarter_starts("2025-04") + [None]:
        try:
            blk = "latest" if d is None else evm_block_at_fast(CELO, d.timestamp(), head)
            election = _registry("Election", blk)
            er = _registry("EpochRewards", blk)
            votes = u256(eth_call(CELO, election, "0x9a0e7d66", blk)) / 1e18
            sup = u256(eth_call(CELO, GOLD, "0x18160ddd", blk)) / 1e18
            ty = u256(eth_call(CELO, er, "0x171af90f", blk)) / 1e24  # target voting yield per epoch(day)
            r = ty * 365
            rows.append(row("celo", TODAY if d is None else d.date(), votes / sup, "total supply (CELO token totalSupply)",
                            r, r * votes / sup, None,
                            f"{CELO} eth_call Registry->Election.getTotalVotes, EpochRewards.getTargetVotingYieldParameters, "
                            f"GoldToken.totalSupply @ {blk}",
                            "r = target voting yield per epoch x 365 (protocol target; actual adjusts with rewards multiplier); "
                            "E = r*theta (approx); token totalSupply is fixed 1B on L2, so theta uses 1B"))
            time.sleep(0.3)
        except Exception as ex:
            print("celo", d, str(ex)[:150])
    return rows


# ------------------------------------------------------------- Flow
FLOW = "https://rest-mainnet.onflow.org/v1"
FLOW_SCRIPT = b"""import FlowIDTableStaking from 0x8624b52f9ddcd04a
import FlowToken from 0x1654653399040a61
access(all) fun main(): [UFix64] { return [FlowIDTableStaking.getTotalStaked(), FlowToken.totalSupply, FlowIDTableStaking.getEpochTokenPayout(), FlowIDTableStaking.getRewardCutPercentage()] }"""


def flow():
    r = S.post(f"{FLOW}/scripts", json={"script": base64.b64encode(FLOW_SCRIPT).decode(), "arguments": []}, timeout=60)
    v = json.loads(base64.b64decode(r.json()))["value"]
    staked, sup, payout, cut = (float(x["value"]) for x in v)
    cm = coinmetrics_monthly("flow")
    k = sorted(cm)[-1]
    fs = cm[k]["fee"] / (cm[k]["fee"] + cm[k]["iss"]) if (cm[k]["fee"] + cm[k]["iss"]) else None
    rr = payout * 52.18 / staked
    rows = [row("flow", TODAY, staked / sup, "total supply (FlowToken.totalSupply)", rr, rr * staked / sup * (1 - (fs or 0)), fs,
                f"{FLOW}/scripts FlowIDTableStaking.getTotalStaked/getEpochTokenPayout, FlowToken.totalSupply; fee_share {CM_SRC} ({k})",
                f"r = weekly epoch payout x 52.18 / total staked (gross; delegators pay {cut:.0%} cut to node operators); "
                "payout combines minted tokens and fees")]
    # monthly E history from Coin Metrics issuance (no staked history)
    for kk in sorted(cm):
        c = cm[kk]
        if c["sply"] and c["iss"]:
            rows.append(row("flow", kk + "-28", None, "", None, c["iss"] * 12 / c["sply"],
                            c["fee"] / (c["fee"] + c["iss"]), CM_SRC, "issuance-only row (monthly CM issuance x12 / supply)"))
    return rows


# ------------------------------------------------------------- Ethereum
DEPOSIT = "0x00000000219ab540356cBB839Cbe05303d7705Fa"


def ethereum():
    rows = []
    cm = coinmetrics_monthly("eth")
    # current: sum of beacon validator balances at head (public beacon node, head state only)
    try:
        bu = "https://ethereum-beacon-api.publicnode.com/eth/v1/beacon/states/head/validator_balances"
        j = S.get(bu, timeout=300).json()["data"]
        staked = sum(int(x["balance"]) for x in j) / 1e9
        k = sorted(cm)[-1]
        sup = cm[k]["sply"]
        iss = cm[k]["iss"] * 12
        rows.append(row("ethereum", TODAY, staked / sup, "total supply (Coin Metrics SplyCur)",
                        iss / staked, iss / sup, None,
                        f"{bu} (sum of {len(j)} balances); issuance {CM_SRC} ({k})",
                        "staked = sum of beacon balances (incl. pending-exit/withdrawable); r = CL issuance only, "
                        "excludes priority fees and MEV; fee_share not measured"))
    except Exception as ex:
        print("eth head", str(ex)[:150])
    # history before Shapella (2023-04-12): deposit-contract balance = cumulative deposits (lower bound of stake)
    url = ETH_RPCS[0]
    head = int(rpc(url, "eth_blockNumber"), 16)
    for d in quarter_starts("2021-01"):
        if d.date() > dt.date(2023, 4, 1):
            break
        try:
            blk = evm_block_at_fast(url, d.timestamp(), head)
            dep = int(rpc(url, "eth_getBalance", [DEPOSIT, hex(blk)]), 16) / 1e18
            k = d.strftime("%Y-%m")
            prevk = (d - dt.timedelta(days=2)).strftime("%Y-%m")
            sup = (cm.get(prevk) or cm.get(k) or {}).get("sply")
            rows.append(row("ethereum", d.date(), dep / sup if sup else None,
                            "total supply (Coin Metrics SplyCur); numerator = cumulative deposits",
                            None, None, None, f"{url} eth_getBalance deposit contract {DEPOSIT} @block {blk}; {CM_SRC}",
                            "pre-Shapella LOWER BOUND: excludes accrued CL rewards (~2-4% of stake by 2023); r/E not available "
                            "(CL issuance not in Coin Metrics before the Merge)"))
            time.sleep(0.5)
        except Exception as ex:
            print("eth hist", d, str(ex)[:150])
    # post-Merge monthly E (CL issuance / supply) from Coin Metrics, theta not available
    for kk in sorted(cm):
        if kk >= "2022-10":
            c = cm[kk]
            rows.append(row("ethereum", kk + "-28", None, "", None, c["iss"] * 12 / c["sply"], None, CM_SRC,
                            "issuance-only row: post-Merge CM IssTotNtv = consensus-layer issuance; x12 / supply"))
    return rows


# ------------------------------------------------------------- Berachain
def berachain():
    # No free consensus-layer staking API found; placeholder returns nothing (see NOTES.md).
    return []


CHAINS2 = {"polkadot": polkadot, "polygon": polygon, "bsc": bsc, "celo": celo,
           "flow": flow, "ethereum": ethereum, "berachain": berachain}
