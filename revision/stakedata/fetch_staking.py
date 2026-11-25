#!/usr/bin/env python
"""Fetch staking ratio (theta), nominal reward rate (r), issuance-to-stakers (E)
and fee share for 24 PoS chains from free public APIs.

Usage:
    python fetch_staking.py                # all chains
    python fetch_staking.py tezos cardano  # some chains
    python fetch_staking.py --build        # only rebuild staking_panel.csv from raw/

Each chain function returns a list of row dicts and is cached to raw/<chain>.json.
No API keys, no accounts. Generic User-Agent. Only public read endpoints.

Definitions used throughout (see NOTES.md for per-chain caveats):
    theta = staked / denominator (denominator named in theta_denominator)
    r     = nominal annual reward per staked token, gross of validator commission,
            before inflation adjustment
    E     = annual new issuance paid to stakers / denominator supply
    fee_share = share of staker rewards that comes from fees (not issuance)
"""
import datetime as dt
import json
import os
import sys
import time

import certifi
import requests

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
os.makedirs(RAW, exist_ok=True)
H = {"User-Agent": "research-client/1.0", "Accept": "application/json"}
TODAY = dt.datetime.now(dt.timezone.utc).date().isoformat()
COLS = ["chain", "date", "theta", "theta_denominator", "r", "E", "fee_share",
        "source", "retrieved", "notes"]
S = requests.Session()
S.headers.update(H)


# ----------------------------------------------------------------- helpers
def get(url, params=None, headers=None, tries=4, timeout=60):
    for i in range(tries):
        try:
            r = S.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 429 and i < tries - 1:
                time.sleep(5 * (i + 1)); continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def post(url, payload, tries=4, timeout=60):
    for i in range(tries):
        try:
            r = S.post(url, json=payload, timeout=timeout)
            if r.status_code == 429 and i < tries - 1:
                time.sleep(5 * (i + 1)); continue
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def rpc(url, method, params=None):
    j = post(url, {"jsonrpc": "2.0", "id": 1, "method": method,
                   "params": params if params is not None else []})
    if "error" in j and j["error"]:
        raise RuntimeError(f"{method}: {j['error']}")
    return j["result"]


def row(chain, date, theta, den, r, E, fee_share, source, notes=""):
    f = lambda x: None if x is None else float(x)
    return dict(chain=chain, date=str(date)[:10], theta=f(theta), theta_denominator=den,
                r=f(r), E=f(E), fee_share=f(fee_share), source=source,
                retrieved=TODAY, notes=notes)


def month_ends(start="2021-01", end=None):
    """Return list of datetimes at 00:00 UTC on the 1st of each month (=end of prev month)."""
    y, m = map(int, start.split("-"))
    out = []
    now = dt.datetime.now(dt.timezone.utc)
    while True:
        d = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
        if d > now:
            break
        out.append(d)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def quarter_starts(start="2021-01"):
    return [d for d in month_ends(start) if d.month in (1, 4, 7, 10)]


# --------------------------------------------------------- EVM helpers
def eth_call(url, to, data, block="latest"):
    if isinstance(block, int):
        block = hex(block)
    return rpc(url, "eth_call", [{"to": to, "data": data}, block])


def u256(hexstr, i=0):
    h = hexstr[2:] if hexstr.startswith("0x") else hexstr
    return int(h[64 * i:64 * (i + 1)], 16)


def enc_u(x):
    return format(int(x), "064x")


def evm_block_at(url, ts, lo=1, hi=None):
    """Binary-search the first block with timestamp >= ts (unix seconds)."""
    if hi is None:
        hi = int(rpc(url, "eth_blockNumber"), 16)
    tsof = lambda n: int(rpc(url, "eth_getBlockByNumber", [hex(n), False])["timestamp"], 16)
    if tsof(hi) < ts:
        return None
    if tsof(lo) >= ts:
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if tsof(mid) >= ts:
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------- Coin Metrics
def coinmetrics_monthly(asset, metrics=("IssTotNtv", "FeeTotNtv", "SplyCur")):
    """Monthly sums of issuance/fees and month-end supply from Coin Metrics community API."""
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = dict(assets=asset, metrics=",".join(metrics), frequency="1d",
                  start_time="2020-12-01", page_size=10000)
    data = []
    while True:
        j = get(url, params=params)
        data += j["data"]
        nxt = j.get("next_page_token")
        if not nxt:
            break
        params["next_page_token"] = nxt
    out = {}
    for d in data:
        k = d["time"][:7]
        o = out.setdefault(k, {"iss": 0.0, "fee": 0.0, "sply": None, "n": 0})
        if d.get("IssTotNtv") is not None:
            o["iss"] += float(d["IssTotNtv"])
        if d.get("FeeTotNtv") is not None:
            o["fee"] += float(d["FeeTotNtv"])
        if d.get("SplyCur") is not None:
            o["sply"] = float(d["SplyCur"])
        o["n"] += 1
    # drop partial months (coverage start/end) so ratios are not distorted
    return {k: v for k, v in out.items() if v["n"] >= 28}


CM_SRC = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics (IssTotNtv,FeeTotNtv,SplyCur)"


# ================================================================ CHAINS
# ---------------------------------------------------------------- Cardano
def cardano():
    base = "https://api.koios.rest/api/v1"
    ep = get(f"{base}/epoch_info", params={"epoch_no": "gte.230", "order": "epoch_no.asc",
                                            "select": "epoch_no,start_time,active_stake,total_rewards,fees"})
    tot = get(f"{base}/totals", params={"epoch_no": "gte.230", "order": "epoch_no.asc"})
    par = get(f"{base}/epoch_params", params={"epoch_no": "gte.230", "order": "epoch_no.asc",
                                               "select": "epoch_no,monetary_expand_rate,treasury_growth_rate"})
    tot = {t["epoch_no"]: t for t in tot}
    par = {p["epoch_no"]: p for p in par}
    rows = []
    for e in ep:
        n = e["epoch_no"]
        t, p = tot.get(n), par.get(n)
        if not t or not p or not e.get("active_stake") or not e.get("total_rewards"):
            continue
        date = dt.datetime.fromtimestamp(e["start_time"], dt.timezone.utc).date()
        if date.year < 2021:
            continue
        stake = int(e["active_stake"]); supply = int(t["supply"]); circ = int(t["circulation"])
        rew = int(e["total_rewards"]); fees = int(e["fees"]); res = int(t["reserves"])
        rho = float(p["monetary_expand_rate"])
        fee_share = fees / (rho * res + fees)
        r = rew * 73 / stake
        E = rew * (1 - fee_share) * 73 / supply
        rows.append(row("cardano", date, stake / supply, "total supply (45B - reserves, Koios totals.supply)",
                        r, E, fee_share,
                        f"{base}/epoch_info,totals,epoch_params (epoch {n})",
                        f"epoch {n}; theta vs circulation={stake/circ:.4f}; r=total_rewards*73/active_stake"))
    return rows


# ---------------------------------------------------------------- Tezos
def tezos():
    st = get("https://api.tzkt.io/v1/statistics/cyclic", params={"limit": 10000, "sort.asc": "cycle"})
    cm = coinmetrics_monthly("xtz")
    # last cycle snapshot of each month
    bym = {}
    for s in st:
        bym[s["timestamp"][:7]] = s
    keys = sorted(bym)
    rows = []
    for a, b in zip(keys[:-1], keys[1:]):
        s0, s1 = bym[a], bym[b]
        if s1["timestamp"][:4] < "2021":
            continue
        t0 = dt.datetime.fromisoformat(s0["timestamp"].replace("Z", "+00:00"))
        t1 = dt.datetime.fromisoformat(s1["timestamp"].replace("Z", "+00:00"))
        yrs = (t1 - t0).total_seconds() / (365.25 * 86400)
        created = (s1["totalCreated"] - s0["totalCreated"]) / yrs
        staked = s1["totalOwnStaked"] + s1["totalExternalStaked"]
        deleg = s1["totalOwnDelegated"] + s1["totalExternalDelegated"]
        part = staked + deleg
        bp = s1["totalBakingPower"]
        theta = part / s1["circulatingSupply"]
        E = created / s1["totalSupply"]
        r = created / bp if bp else None
        c = cm.get(b[:7]) or {}
        fs = c["fee"] / (c["fee"] + c["iss"]) if c and (c["fee"] + c["iss"]) > 0 else None
        rows.append(row("tezos", s1["timestamp"], theta, "circulating supply (tzkt); numerator = staked + delegated",
                        r, E, fs,
                        f"https://api.tzkt.io/v1/statistics/cyclic (cycle {s1['cycle']}); fee_share: {CM_SRC}",
                        f"staked-only (frozen) ratio={staked/s1['totalSupply']:.4f} of total supply; "
                        f"r = minted/totalBakingPower (post-Paris delegated tez weigh 1/3); "
                        f"minted incl. liquidity-baking subsidy"))
    return rows


# ---------------------------------------------------------------- Cosmos SDK
def _cosmos_basic(chain, lcd, denom):
    pool = get(f"{lcd}/cosmos/staking/v1beta1/pool")["pool"]
    bonded = int(pool["bonded_tokens"])
    try:
        sup = int(get(f"{lcd}/cosmos/bank/v1beta1/supply/by_denom", params={"denom": denom})["amount"]["amount"])
    except Exception:  # older SDK (e.g. Sei) only serves /supply/{denom}
        sup = int(get(f"{lcd}/cosmos/bank/v1beta1/supply/{denom}")["amount"]["amount"])
    ctax = float(get(f"{lcd}/cosmos/distribution/v1beta1/params")["params"]["community_tax"])
    return bonded, sup, ctax


def cosmos_hub():
    lcd = "https://cosmos-rest.publicnode.com"
    bonded, sup, ctax = _cosmos_basic("cosmos-hub", lcd, "uatom")
    ap = float(get(f"{lcd}/cosmos/mint/v1beta1/annual_provisions")["annual_provisions"])
    infl = float(get(f"{lcd}/cosmos/mint/v1beta1/inflation")["inflation"])
    E = ap * (1 - ctax) / sup
    return [row("cosmos-hub", TODAY, bonded / sup, "total supply (bank supply uatom)",
                ap * (1 - ctax) / bonded, E, None,
                f"{lcd}/cosmos/staking/v1beta1/pool, /cosmos/mint/v1beta1/annual_provisions, /cosmos/distribution/v1beta1/params",
                f"mint inflation={infl}, community_tax={ctax}; fees ignored (small)")]


def osmosis():
    lcd = "https://osmosis-rest.publicnode.com"
    bonded, sup, ctax = _cosmos_basic("osmosis", lcd, "uosmo")
    epp = float(get(f"{lcd}/osmosis/mint/v1beta1/epoch_provisions")["epoch_provisions"])
    par = get(f"{lcd}/osmosis/mint/v1beta1/params")["params"]
    stake_prop = float(par["distribution_proportions"]["staking"])
    epd = {"day": 365, "week": 52}.get(par.get("epoch_identifier", "day"), 365)
    ann = epp * epd * stake_prop * (1 - ctax)
    return [row("osmosis", TODAY, bonded / sup, "total supply (bank supply uosmo)",
                (ann / bonded) if ann > 0 else None, ann / sup, 1.0 if ann == 0 else None,
                f"{lcd}/osmosis/mint/v1beta1/epoch_provisions,params; /cosmos/staking/v1beta1/pool",
                f"staking proportion={stake_prop}, community_tax={ctax}; taker-fee revenue to stakers (non-OSMO) not included; "
                f"with staking proportion 0 all staker income is fee-funded (taker fees), amount not measured so r left blank")]


def injective():
    lcd = "https://injective-rest.publicnode.com"
    bonded, sup, ctax = _cosmos_basic("injective", lcd, "inj")
    ap = float(get(f"{lcd}/cosmos/mint/v1beta1/annual_provisions")["annual_provisions"])
    infl = float(get(f"{lcd}/cosmos/mint/v1beta1/inflation")["inflation"])
    ann = ap * (1 - ctax)
    return [row("injective", TODAY, bonded / sup, "total supply (bank supply inj)",
                ann / bonded, ann / sup, None,
                f"{lcd}/cosmos/mint/v1beta1/annual_provisions,inflation; /cosmos/staking/v1beta1/pool",
                f"mint inflation={infl}, community_tax={ctax}; exchange fees mostly burned (burn auction); gas fees to stakers ignored")]


def sei():
    lcd = "https://sei-rest.publicnode.com"
    bonded, sup, ctax = _cosmos_basic("sei", lcd, "usei")
    m = get(f"{lcd}/seichain/mint/v1beta1/minter")
    days = (dt.date.fromisoformat(m["end_date"]) - dt.date.fromisoformat(m["start_date"])).days + 1
    ann = float(m["total_mint_amount"]) * 365 / days * (1 - ctax)
    return [row("sei", TODAY, bonded / sup, "total supply (bank supply usei)",
                ann / bonded, ann / sup, None,
                f"{lcd}/seichain/mint/v1beta1/minter (token_release_schedule), /cosmos/staking/v1beta1/pool",
                f"release period {m['start_date']}..{m['end_date']} total {int(m['total_mint_amount'])/1e6:.0f} SEI; community_tax={ctax}; "
                "assumes all scheduled release goes to stakers; fees ignored")]


# ---------------------------------------------------------------- Hedera
def hedera():
    b = "https://mainnet-public.mirrornode.hedera.com/api/v1"
    st = get(f"{b}/network/stake")
    sp = get(f"{b}/network/supply")
    stake = st["stake_total"] / 1e8
    rel = int(sp["released_supply"]) / 1e8
    r = st["max_staking_reward_rate_per_hbar"] * 365 / 1e8
    ann = st["staking_reward_rate"] * 365 / 1e8
    return [row("hedera", TODAY, stake / rel, "released (circulating) supply, mirror node network/supply",
                r, None, None, f"{b}/network/stake, {b}/network/supply",
                f"r = max_staking_reward_rate_per_hbar*365 (rate for reward-eligible stake); "
                f"daily reward budget {st['staking_reward_rate']/1e8:.0f} HBAR (~{ann/1e6:.0f}M/yr = {ann/rel:.4f} of released supply) "
                "paid from account 0.0.800 (fees + treasury), no new minting; stake_total includes stake that declines rewards")]


# ---------------------------------------------------------------- MultiversX
def multiversx():
    u = "https://api.multiversx.com/economics"
    e = get(u)
    return [row("multiversx", TODAY, e["staked"] / e["circulatingSupply"], "circulating supply (API economics)",
                e["apr"], None, None, u,
                f"apr={e['apr']} (avg incl. top-up), baseApr={e['baseApr']}, topUpApr={e['topUpApr']}; E not exposed by API")]


# ---------------------------------------------------------------- Algorand
def algorand():
    u = "https://mainnet-api.algonode.cloud/v2/ledger/supply"
    s = get(u)
    return [row("algorand", TODAY, s["online-money"] / s["total-money"], "total money (algod ledger/supply)",
                None, None, None, u,
                "theta = online (consensus-participating) stake, not locked; staking rewards since 2025 are proposer "
                "payouts (fees + fee-sink bonus) to incentive-eligible accounts only; no free API for r")]


# ---------------------------------------------------------------- Avalanche
def avalanche():
    P = "https://api.avax.network/ext/bc/P"
    sup = int(rpc(P, "platform.getCurrentSupply", {})["supply"]) / 1e9
    vals = rpc(P, "platform.getCurrentValidators", {})["validators"]
    w_tot = 0; w_val = 0; rew_ann = 0
    for v in vals:
        w = int(v["weight"]) / 1e9
        dw = int(v.get("delegatorWeight", 0) or 0) / 1e9
        w_tot += w + dw
        st = int(v.get("stakeAmount", v["weight"])) / 1e9
        pr = int(v.get("potentialReward", 0) or 0) / 1e9
        dur = (int(v["endTime"]) - int(v["startTime"])) / (365.25 * 86400)
        if dur > 0 and st > 0:
            w_val += st
            rew_ann += pr / dur
    r = rew_ann / w_val
    g = get("https://glacier-api.avax.network/v1/networks/mainnet")
    return [row("avalanche", TODAY, w_tot / sup, "P-Chain current supply (platform.getCurrentSupply, incl. C/X-chain)",
                r, r * w_tot / sup, 0.0,
                f"{P} platform.getCurrentSupply, platform.getCurrentValidators; cross-check glacier-api /v1/networks/mainnet",
                f"r = sum(potentialReward/duration)/sum(validator own stake) over {len(vals)} validators "
                f"(gross, delegator rewards assumed same rate); E approximated r*theta; fees are burned (fee_share=0); "
                f"glacier total staked={(int(g['validatorDetails']['totalAmountStaked'])+int(g['delegatorDetails']['totalAmountStaked']))/1e9:,.0f}")]


# ---------------------------------------------------------------- Solana
def solana():
    u = "https://api.mainnet-beta.solana.com"
    inf = rpc(u, "getInflationRate")
    sup = rpc(u, "getSupply", [{"excludeNonCirculatingAccountsList": True}])["value"]
    va = rpc(u, "getVoteAccounts")
    stake = sum(v["activatedStake"] for v in va["current"] + va["delinquent"]) / 1e9
    tot = sup["total"] / 1e9; circ = sup["circulating"] / 1e9
    return [row("solana", TODAY, stake / tot, "total supply (getSupply.total)",
                inf["total"] * tot / stake, inf["validator"], None,
                f"{u} getInflationRate, getSupply, getVoteAccounts (epoch {inf['epoch']})",
                f"theta vs circulating={stake/circ:.4f} (staked incl. locked); r = inflation*supply/stake, "
                "excludes priority fees and MEV (Jito) tips; fee_share not available from RPC")]


# ---------------------------------------------------------------- Tron
def tron():
    tg = "https://api.trongrid.io"
    w = post(f"{tg}/wallet/listwitnesses", {})["witnesses"]
    votes = sum(int(x.get("voteCount", 0)) for x in w)
    cp = {p["key"]: p.get("value") for p in post(f"{tg}/wallet/getchainparameters", {})["chainParameter"]}
    pay = (cp.get("getWitnessPayPerBlock", 0) + cp.get("getWitness127PayPerBlock", 0)) / 1e6
    f = get("https://apilist.tronscanapi.com/api/funds")
    sup = float(f["totalTurnOver"])
    ann = pay * 365.25 * 86400 / 3
    return [row("tron", TODAY, votes / sup, "total supply (tronscan /api/funds totalTurnOver)",
                ann / votes, ann / sup, 0.0,
                f"{tg}/wallet/listwitnesses, /wallet/getchainparameters; https://apilist.tronscanapi.com/api/funds",
                f"theta = TRX voting for SRs/candidates (votes = staked TRX under Stake 2.0) / supply; block pay {pay} TRX/3s block; "
                "r gross before SR sharing ratio; fees burned; net supply deflationary")]


# ---------------------------------------------------------------- NEAR
def near():
    u = "https://rpc.mainnet.near.org"
    v = rpc(u, "validators", [None])
    stake = sum(int(x["stake"]) for x in v["current_validators"]) / 1e24
    blk = rpc(u, "block", {"finality": "final"})
    sup = int(blk["header"]["total_supply"]) / 1e24
    cfg = rpc(u, "EXPERIMENTAL_protocol_config", {"finality": "final"})
    mi = cfg["max_inflation_rate"]; pr = cfg["protocol_reward_rate"]
    E = (mi[0] / mi[1]) * (1 - pr[0] / pr[1])
    date = dt.datetime.fromtimestamp(int(blk["header"]["timestamp"]) / 1e9, dt.timezone.utc).date()
    rows = [row("near", date, stake / sup, "total supply (block header total_supply)",
                E * sup / stake, E, 0.0,
                f"{u} validators, block, EXPERIMENTAL_protocol_config",
                f"max_inflation={mi}, protocol_reward_rate={pr}; E assumes full validator uptime; "
                "fees 70% burned / 30% to contracts, none to stakers")]
    rows += near_history()
    return rows


def near_history():
    u = "https://archival-rpc.mainnet.fastnear.com"
    rows = []
    try:
        head = rpc(u, "block", {"finality": "final"})["header"]
    except Exception:
        return rows
    hh, ht = head["height"], int(head["timestamp"]) / 1e9
    for d in quarter_starts("2021-01"):
        ts = d.timestamp()
        h = int(hh - (ht - ts) * 1.0)  # first guess ~1 block/s, refine
        try:
            for _ in range(6):
                b = None
                for k in range(20):
                    try:
                        b = rpc(u, "block", {"block_id": h + k}); break
                    except Exception:
                        continue
                bt = int(b["header"]["timestamp"]) / 1e9
                if abs(bt - ts) < 3600:
                    break
                h = int(h - (bt - ts) * 0.9)
            hdr = b["header"]
            # validators for the epoch containing this block
            v = rpc(u, "validators", {"epoch_id": hdr["epoch_id"]})
            stake = sum(int(x["stake"]) for x in v["current_validators"]) / 1e24
            sup = int(hdr["total_supply"]) / 1e24
            cfg = rpc(u, "EXPERIMENTAL_protocol_config", {"block_id": hdr["height"]})
            mi = cfg["max_inflation_rate"]; pr = cfg["protocol_reward_rate"]
            E = (mi[0] / mi[1]) * (1 - pr[0] / pr[1])
            rows.append(row("near", dt.datetime.fromtimestamp(bt, dt.timezone.utc).date(), stake / sup,
                            "total supply (block header total_supply)", E * sup / stake, E, 0.0,
                            f"{u} block/validators(epoch_id)/EXPERIMENTAL_protocol_config at height {hdr['height']}",
                            "archival; E assumes full uptime"))
        except Exception as ex:
            print("near hist", d.date(), str(ex)[:120])
    return rows


# ---------------------------------------------------------------- Sui
def sui():
    u = "https://graphql.mainnet.sui.io/graphql"
    q = lambda s: post(u, {"query": s})
    last = q("{ epoch { epochId } }")["data"]["epoch"]["epochId"]
    rows = []
    TOTAL = 10_000_000_000
    ids = list(range(last - 1, 0, -30))[::-1]
    for e in ids:
        j = q('{ epoch(epochId:%d){ epochId startTimestamp endTimestamp totalStakeRewards totalStakeSubsidies '
              'totalGasFees validatorSet{ contents{ json } } } }' % e)
        ep = (j.get("data") or {}).get("epoch")
        if not ep or not ep.get("validatorSet") or not ep.get("endTimestamp"):
            continue
        ts = ep["validatorSet"]["contents"]["json"]["total_stake"]
        stake = int(ts) / 1e9
        t0 = dt.datetime.fromisoformat(ep["startTimestamp"].replace("Z", "+00:00"))
        t1 = dt.datetime.fromisoformat(ep["endTimestamp"].replace("Z", "+00:00"))
        per = 365.25 * 86400 / (t1 - t0).total_seconds()
        rew = int(ep["totalStakeRewards"]) / 1e9; sub = int(ep["totalStakeSubsidies"]) / 1e9
        gas = int(ep["totalGasFees"]) / 1e9
        if t0.year < 2021 or rew == 0:  # skip genesis epochs before stake subsidies began
            continue
        rows.append(row("sui", t0.date(), stake / TOTAL, "total supply (10B SUI fixed)",
                        rew * per / stake, sub * per / TOTAL, (rew - sub) / rew if rew else None,
                        f"{u} epoch({e}) totalStakeRewards/totalStakeSubsidies/validatorSet.total_stake",
                        f"epoch {e}; staked includes locked/non-circulating SUI; fee_share = (stake rewards - subsidy)/stake rewards"))
        time.sleep(0.3)
    return rows


# ---------------------------------------------------------------- Aptos
def _aptos_point(base, ver=None):
    p = {"ledger_version": ver} if ver else None
    sc = get(f"{base}/accounts/0x1/resource/0x1::staking_config::StakingConfig", params=p)["data"]
    br = get(f"{base}/accounts/0x1/resource/0x1::block::BlockResource", params=p)["data"]
    vs = get(f"{base}/accounts/0x1/resource/0x1::stake::ValidatorSet", params=p)["data"]
    body = {"function": "0x1::coin::supply", "type_arguments": ["0x1::aptos_coin::AptosCoin"], "arguments": []}
    url = f"{base}/view" + (f"?ledger_version={ver}" if ver else "")
    try:
        sup = int(post(url, body, tries=2)[0]["vec"][0]) / 1e8
    except Exception:  # early ledger versions: read CoinInfo<AptosCoin>.supply directly
        ci = get(f"{base}/accounts/0x1/resource/0x1::coin::CoinInfo<0x1::aptos_coin::AptosCoin>", params=p)["data"]
        v = ci["supply"]["vec"][0]
        sup = int(v["integer"]["vec"][0]["value"] if v.get("integer", {}).get("vec") else
                  v["aggregator"]["vec"][0]["handle"]) / 1e8
    epochs = 365.25 * 86400e6 / int(br["epoch_interval"])
    try:  # AIP-30 periodic decrease: StakingRewardsConfig (FixedPoint64) supersedes StakingConfig.rewards_rate
        src = get(f"{base}/accounts/0x1/resource/0x1::staking_config::StakingRewardsConfig", params=p, tries=1)["data"]
        per_epoch = int(src["rewards_rate"]["value"]) / 2 ** 64
    except Exception:
        per_epoch = int(sc["rewards_rate"]) / int(sc["rewards_rate_denominator"])
    rate = per_epoch * epochs
    stake = int(vs["total_voting_power"]) / 1e8
    return stake / sup, rate, rate * stake / sup


def aptos():
    base = "https://api.mainnet.aptoslabs.com/v1"
    th, r, E = _aptos_point(base)
    rows = [row("aptos", TODAY, th, "total supply (0x1::coin::supply AptosCoin)", r, E, None,
                f"{base} StakingConfig, BlockResource, ValidatorSet, view coin::supply",
                "r = StakingRewardsConfig.rewards_rate (FixedPoint64, falls back to StakingConfig) per epoch x epochs/yr (max, full performance); E = r*theta; gas fees burned")]
    arch = "https://archive.mainnet.aptoslabs.com/v1"
    try:
        info = get(arch)
        hh = int(info["block_height"]); ht = int(info["ledger_timestamp"]) / 1e6
        for d in quarter_starts("2023-01"):
            ts = d.timestamp()
            lo, hi = 1, hh
            # interpolation search on block height
            for _ in range(40):
                if hi - lo < 50:
                    break
                mid = (lo + hi) // 2
                b = get(f"{arch}/blocks/by_height/{mid}")
                bt = int(b["block_timestamp"]) / 1e6
                if bt < ts:
                    lo = mid
                else:
                    hi = mid
            b = get(f"{arch}/blocks/by_height/{hi}")
            ver = int(b["last_version"])
            try:
                th, r, E = _aptos_point(arch, ver)
            except Exception as ex:
                print("aptos point", d.date(), repr(ex)[:150]); continue
            rows.append(row("aptos", d.date(), th, "total supply (0x1::coin::supply AptosCoin)", r, E, None,
                            f"{arch} resources at ledger_version {ver}", "archival; r max at full performance"))
    except Exception as ex:
        import traceback; traceback.print_exc()
        print("aptos hist failed", repr(ex)[:200])
    return rows


# ---------------------------------------------------------------- Sonic / Fantom (SFC)
SFC = "0xFC00FACE00000000000000000000000000000000"


def _sfc_point(url, blk):
    stake = u256(eth_call(url, SFC, "0x8b0e9f3f", blk)) / 1e18
    sup = u256(eth_call(url, SFC, "0x18160ddd", blk)) / 1e18
    try:
        consts = "0x" + eth_call(url, SFC, "0xd46fa518", blk)[-40:]
        brps = u256(eth_call(url, consts, "0xd9a7c1f9", blk)) / 1e18
    except Exception:
        try:  # older SFC versions keep baseRewardPerSecond in the SFC itself
            brps = u256(eth_call(url, SFC, "0xd9a7c1f9", blk)) / 1e18
        except Exception:
            brps = None
    return stake, sup, brps


def _sfc_chain(chain, url, start):
    rows = []
    head = int(rpc(url, "eth_blockNumber"), 16)
    lo = 1
    pts = [d for d in quarter_starts(start)] + [None]
    for d in pts:
        try:
            if d is None:
                blk = head; date = TODAY
            else:
                blk = evm_block_at(url, d.timestamp(), lo, head)
                if blk is None:
                    continue
                lo = blk; date = d.date()
            stake, sup, brps = _sfc_point(url, blk)
            if stake == 0:
                continue
            ann = brps * 365.25 * 86400 if brps is not None else None
            rows.append(row(chain, date, stake / sup, "total supply (SFC.totalSupply)",
                            ann / stake if ann is not None else None, ann / sup if ann is not None else None, None,
                            f"{url} eth_call SFC {SFC} totalStake/totalSupply, ConstantsManager.baseRewardPerSecond @block {blk}",
                            "r = baseRewardPerSecond*yr/totalStake (base reward only, excludes validators' fee share; "
                            "uptime-weighted in practice)"))
        except Exception as ex:
            print(chain, d, str(ex)[:150])
    return rows


def sonic():
    return _sfc_chain("sonic", "https://rpc.soniclabs.com", "2025-01")


def fantom():
    return _sfc_chain("fantom", "https://rpcapi.fantom.network", "2021-01")


# ---------------------------------------------------------------- CHAIN REGISTRY
CHAINS = {
    "cardano": cardano, "tezos": tezos, "cosmos-hub": cosmos_hub, "osmosis": osmosis,
    "injective": injective, "sei": sei, "hedera": hedera, "multiversx": multiversx,
    "algorand": algorand, "avalanche": avalanche, "solana": solana, "tron": tron,
    "near": near, "sui": sui, "aptos": aptos, "sonic": sonic, "fantom": fantom,
}

try:
    from fetch_staking_part2 import CHAINS2  # noqa: E402
    CHAINS.update(CHAINS2)
except ImportError:
    pass


def build():
    import csv
    allrows = []
    for f in sorted(os.listdir(RAW)):
        if f.endswith(".json"):
            allrows += json.load(open(os.path.join(RAW, f)))
    allrows.sort(key=lambda r: (r["chain"], r["date"]))
    with open(os.path.join(HERE, "staking_panel.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in allrows:
            w.writerow({k: r.get(k) for k in COLS})
    print("wrote", len(allrows), "rows")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--build" not in sys.argv:
        for name in (args or list(CHAINS)):
            t = time.time()
            try:
                rows = CHAINS[name]()
                json.dump(rows, open(os.path.join(RAW, f"{name}.json"), "w"), indent=1)
                print(f"{name}: {len(rows)} rows ({time.time()-t:.0f}s)")
            except Exception as ex:
                print(f"{name}: FAILED {type(ex).__name__}: {str(ex)[:200]}")
    build()
