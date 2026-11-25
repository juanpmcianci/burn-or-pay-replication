"""Fetch daily chain fee revenue (DefiLlama, USD) and market cap (Coin Metrics community, CapMrktEstUSD) for PoS L1s."""
import json, time, urllib.request, pandas as pd
PAIRS = {  # defillama chain slug -> coin metrics asset
    "ethereum": "eth", "solana": "sol", "avalanche": "avax", "cosmos-hub": "atom", "cardano": "ada",
    "polkadot": "dot", "tron": "trx", "bsc": "bnb", "near": "near", "algorand": "algo", "tezos": "xtz",
    "sui": "sui", "aptos": "apt", "polygon": "pol", "osmosis": "osmo", "injective": "inj", "sei": "sei",
    "celo": "celo", "fantom": "ftm", "hedera": "hbar", "ton": "ton", "flow": "flow", "kava": "kava",
    "multiversx": "egld", "mantra": "om", "berachain": "bera", "sonic": "s",
}
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client"}), timeout=60) as r:
        return json.load(r)
rows = []
for slug, a in PAIRS.items():
    try:
        f = get(f"https://api.llama.fi/summary/fees/{slug}?dataType=dailyFees")["totalDataChart"]
        fee = pd.DataFrame(f, columns=["t", "fee"]); fee["date"] = pd.to_datetime(fee.t, unit="s").dt.date
    except Exception as e:
        print(slug, "fees fail", e); continue
    caps, url = [], f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets={a}&metrics=CapMrktEstUSD&frequency=1d&page_size=10000"
    try:
        while url:
            d = get(url); caps += d["data"]; url = d.get("next_page_url"); time.sleep(0.3)
    except Exception as e:
        print(slug, a, "cap fail", e); continue
    cap = pd.DataFrame(caps); cap["date"] = pd.to_datetime(cap.time).dt.date; cap["mcap"] = cap.CapMrktEstUSD.astype(float)
    m = fee[["date", "fee"]].merge(cap[["date", "mcap"]], on="date"); m["chain"] = slug
    print(slug, len(fee), len(cap), len(m), m.date.min() if len(m) else None)
    rows.append(m); time.sleep(0.5)
pd.concat(rows).to_csv("fees_mcap_daily.csv", index=False)
