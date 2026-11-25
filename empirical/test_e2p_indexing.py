"""Unit test for hour indexing: Uniswap's 2024-02-23 14:00 UTC post coincides with the 13:00->14:00-close jump
(close 7.19 at the 13:00 candle, 11.12 at the 14:00 candle). That return must land in hour 0, not hour -1."""
import numpy as np
import pandas as pd
import e2p_capture as m

def test_uniswap_jump_in_hour0():
    r = m.one("t", "UNIUSDT", pd.Timestamp("2024-02-23 14:00"))
    jump = np.log(11.12 / 7.19)
    assert r["car4"] > 0.8 * jump, r
    assert abs(r["pre24"]) < 0.1, r

if __name__ == "__main__":
    test_uniswap_jump_in_hour0(); print("indexing test passed")
