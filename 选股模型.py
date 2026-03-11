import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from WindPy import w

w.start()
w.isconnected()

def _to_wind_date(s: str) -> str:
    # 接受 'YYYY-MM-DD' 或 'YYYYMMDD'
    s = str(s)
    if "-" in s:
        return s
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s


def _chunks(lst, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _wsd_field_to_df(
    field_data: object,
    times: pd.DatetimeIndex,
    codes: list[str],
) -> pd.DataFrame:
    """
    Normalize one-field WSD payload to DataFrame(index=times, columns=codes).
    Wind may return field_data in different layouts depending on account/server.
    """
    nT, nC = len(times), len(codes)

    # 1) Flattened vector: length = nT * nC
    arr = np.array(field_data, dtype=object)
    if arr.ndim == 1 and arr.size == nT * nC and nC > 0:
        return pd.DataFrame(arr.reshape(nT, nC), index=times, columns=codes)

    # 2) Single-code vector: length = nT
    if arr.ndim == 1 and arr.size == nT and nC == 1:
        return pd.DataFrame(arr.reshape(nT, 1), index=times, columns=codes)

    # 3) Code-major nested: shape like [code][time] -> (nC, nT)
    try:
        df = pd.DataFrame(field_data, index=codes).T
        if df.shape == (nT, nC):
            df.index = times
            df.columns = codes
            return df
    except Exception:
        pass

    # 4) Time-major nested: shape like [time][code] -> (nT, nC)
    try:
        df = pd.DataFrame(field_data, index=times, columns=codes)
        if df.shape == (nT, nC):
            return df
    except Exception:
        pass

    # 5) Generic fallback
    try:
        df = pd.DataFrame(field_data)
        if df.shape == (nC, nT):
            df = df.T
        if df.shape[0] == nT and df.shape[1] == nC:
            df.index = times
            df.columns = codes
            return df
    except Exception:
        pass

    raw_len = len(field_data) if hasattr(field_data, "__len__") else None
    raise ValueError(
        f"Unexpected WSD field shape: nT={nT}, nC={nC}, raw_len={raw_len}"
    )


def _wind_wsd_panel(
    codes: list[str],
    fields: str,
    start: str,
    end: str,
    options: str = "",
    batch: int = 300,
    *,
    string_fields: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Return dict[field] = DataFrame(index=DatetimeIndex, columns=codes)
    """
    start = _to_wind_date(start)
    end = _to_wind_date(end)
    if string_fields is None:
        string_fields = set()

    string_fields = {str(f).strip().lower() for f in string_fields}
    field_list = [f.strip() for f in fields.split(",") if f.strip()]
    out: dict[str, list[pd.DataFrame]] = {f.lower(): [] for f in field_list}

    def _append_field_df(field_key: str, payload: object, times: pd.DatetimeIndex, codes_ret: list[str]) -> None:
        key = field_key.lower()
        df = _wsd_field_to_df(payload, times, codes_ret)
        if key not in string_fields:
            df = df.apply(pd.to_numeric, errors="coerce")
        if key not in out:
            out[key] = []
        out[key].append(df)

    def _fetch_single_field(code_blk: list[str], field: str, s: str, e: str) -> None:
        codes_str = ",".join(code_blk)
        rf = w.wsd(codes_str, field, s, e, options)

        if rf.ErrorCode == 0:
            times = pd.to_datetime(rf.Times)
            codes_ret = list(rf.Codes)
            payload = rf.Data[0] if len(rf.Data) == 1 else rf.Data
            _append_field_df(field, payload, times, codes_ret)
            return

        # Quota-per-request exceeded: split this code block and retry recursively.
        if rf.ErrorCode == -40522017 and len(code_blk) > 1:
            mid = len(code_blk) // 2
            _fetch_single_field(code_blk[:mid], field, s, e)
            _fetch_single_field(code_blk[mid:], field, s, e)
            return

        # If still quota-exceeded on a single-code request, split date range.
        if rf.ErrorCode == -40522017 and len(code_blk) == 1:
            sd = pd.Timestamp(s)
            ed = pd.Timestamp(e)
            if sd < ed:
                mid_dt = sd + (ed - sd) / 2
                mid_dt = pd.Timestamp(mid_dt.date())
                if mid_dt < sd:
                    mid_dt = sd
                left_s = sd.strftime("%Y-%m-%d")
                left_e = mid_dt.strftime("%Y-%m-%d")
                right_s_dt = mid_dt + pd.Timedelta(days=1)
                if right_s_dt <= ed:
                    right_s = right_s_dt.strftime("%Y-%m-%d")
                    right_e = ed.strftime("%Y-%m-%d")
                    _fetch_single_field(code_blk, field, left_s, left_e)
                    _fetch_single_field(code_blk, field, right_s, right_e)
                    return

        raise RuntimeError(f"WSD error {rf.ErrorCode} on field {field}: {rf.Data}")

    def _fetch_block(code_blk: list[str]) -> None:
        codes_str = ",".join(code_blk)
        r = w.wsd(codes_str, ",".join(field_list), start, end, options)

        if r.ErrorCode == 0:
            times = pd.to_datetime(r.Times)
            codes_ret = list(r.Codes)
            for fi, f in enumerate(r.Fields):
                _append_field_df(str(f), r.Data[fi], times, codes_ret)
            return

        # Wind error -40522018: multi-codes with multi-indicators is not supported.
        if r.ErrorCode == -40522018 and len(field_list) > 1:
            for f in field_list:
                _fetch_single_field(code_blk, f, start, end)
            return

        # Quota-per-request exceeded for this block: split and retry.
        if r.ErrorCode == -40522017:
            if len(code_blk) > 1:
                mid = len(code_blk) // 2
                _fetch_block(code_blk[:mid])
                _fetch_block(code_blk[mid:])
                return
            # Single code but multi-fields may still fail; degrade to one-field calls.
            for f in field_list:
                _fetch_single_field(code_blk, f, start, end)
            return

        raise RuntimeError(f"WSD error {r.ErrorCode}: {r.Data}")

    for blk in _chunks(codes, batch):
        _fetch_block(blk)

    merged: dict[str, pd.DataFrame] = {}
    for f, parts in out.items():
        if not parts:
            merged[f] = pd.DataFrame()
        else:
            # Parts can come from both code-splitting and date-splitting.
            # combine_first safely unions both index and columns.
            df = parts[0].copy()
            for p in parts[1:]:
                df = df.combine_first(p)
            df = df.sort_index().sort_index(axis=1)
            merged[f] = df

    return merged
def build_market_close_vol_mktcap(
    start: str,
    end: str,
    *,
    asof_date: str | None = None,     # 'YYYYMMDD'，为空则使用 end 对应日期
    batch_hq: int = 300,
    batch_shares: int = 200,
    cps: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | pd.Series, pd.DataFrame | None]:
    """
    返回 close, vol, mktcap, st_flag, turn
    - 股票池：全A（沪深北）用 w.wset 获取
    - close/volume/turn/mktcap_ashare/sec_name：WSD 拉日序列
    """
    if w.isconnected() != 1:
        w.start()
        if w.isconnected() != 1:
            raise RuntimeError("WindPy is not connected. Please login Wind terminal first.")

    if asof_date is None:
        asof_date = pd.Timestamp(end).strftime("%Y%m%d")

    # 全A股票列表
    r = w.wset("sectorconstituent", f"date={asof_date};sectorid=a001010100000000")
    if r.ErrorCode != 0:
        raise RuntimeError(f"WSET sectorconstituent error {r.ErrorCode}: {r.Data}")

    codes = None
    for col in r.Data:
        if isinstance(col, list) and col and isinstance(col[0], str) and (
            ".SH" in col[0] or ".SZ" in col[0] or ".BJ" in col[0]
        ):
            codes = col
            break
    if not codes:
        raise RuntimeError("Failed to parse stock codes from wset result.")

    codes = sorted(set(codes))

    # 拉取 close/volume/turn/mktcap_ashare（sec_name 改用 WSS 静态拉取）
    fields = "close,volume,turn,mkt_cap_ashare"
    data = _wind_wsd_panel(
        codes=codes,
        fields=fields,
        start=start,
        end=end,
        options="PriceAdj=F;unit=1",
        batch=batch_hq,
    )

    close = data["close"]
    vol = data["volume"]
    turn = data["turn"]
    mktcap = data["mkt_cap_ashare"]

    # sec_name 是静态字段，按批次通过 WSS 拉取，避免 WSD 配额超限
    sec_name_parts: list[pd.Series] = []
    for blk in _chunks(codes, batch_shares):
        rs = w.wss(",".join(blk), "sec_name")
        if rs.ErrorCode != 0:
            raise RuntimeError(f"WSS error {rs.ErrorCode} on field sec_name: {rs.Data}")

        codes_ret = list(rs.Codes)
        vals = None
        if rs.Data and len(rs.Data) == 1 and isinstance(rs.Data[0], list):
            vals = rs.Data[0]
        else:
            arr = np.array(rs.Data, dtype=object)
            if arr.shape == (len(codes_ret), 1):
                vals = arr[:, 0].tolist()
            elif arr.shape == (1, len(codes_ret)):
                vals = arr[0].tolist()
            else:
                vals = arr.reshape(-1).tolist()

        if len(vals) != len(codes_ret):
            raise RuntimeError(
                f"WSS sec_name shape mismatch: len(vals)={len(vals)}, len(codes)={len(codes_ret)}"
            )
        sec_name_parts.append(pd.Series(vals, index=codes_ret, dtype=object))

    if sec_name_parts:
        sec_name_s = pd.concat(sec_name_parts)
        sec_name_s = sec_name_s[~sec_name_s.index.duplicated(keep="last")].reindex(codes)
    else:
        sec_name_s = pd.Series(index=codes, dtype=object)

    # ST 标志：证券简称中包含 ST
    st_col = sec_name_s.astype(str).str.contains("ST", case=False, na=False)
    if close.empty:
        st_flag = pd.DataFrame(columns=codes)
    else:
        st_arr = np.repeat(st_col.to_numpy()[None, :], repeats=len(close.index), axis=0)
        st_flag = pd.DataFrame(st_arr, index=close.index, columns=st_col.index)

    # turn 如果是百分比，转成比例（0-1）
    if not turn.empty:
        sample = turn.stack().dropna()
        if not sample.empty and sample.median() > 1:
            turn = turn / 100.0

    # 对齐索引（只保留 close/vol/mktcap 共同交易日）
    idx = close.index.intersection(vol.index).intersection(mktcap.index)#取三个数据index的交集
    close = close.loc[idx].sort_index()
    vol = vol.loc[idx].sort_index()
    mktcap = mktcap.loc[idx].sort_index()
    if not turn.empty:
        turn = turn.loc[idx].sort_index()
    if not st_flag.empty:
        st_flag = st_flag.loc[idx].sort_index()

    return close, vol, mktcap, st_flag, (turn if not turn.empty else None)
        
def nearest_trading_day(idx: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp:
    """取不晚于 dt 的最近交易日"""
    dt = pd.Timestamp(dt)
    pos = idx.searchsorted(dt, side="right") - 1
    if pos < 0:
        raise ValueError("dt earlier than first trading day in data")
    return idx[pos]

def next_trading_day(idx: pd.DatetimeIndex, dt: pd.Timestamp) -> pd.Timestamp | None:
    dt = pd.Timestamp(dt)
    pos = idx.searchsorted(dt, side='right')#取严格晚于 dt 的最近交易日
    if pos >= len(idx):
        return None
    return idx[pos]  

def weekly_ema_crossunder_to_daily_signal(
    close: pd.DataFrame,
    daily_idx: pd.DatetimeIndex,
    fast_span_weeks: int = 14,  # 14周EMA
    slow_span_weeks: int = 60,  # 60周EMA
) -> pd.DataFrame:
    """
    用周线收盘计算 EMA(14)/EMA(60)，并将“14周EMA下穿60周EMA”映射回日频信号。
    仅在该周最后一个交易日标记 True，其余日期为 False。
    """
    # 周线收盘：若周五非交易日，last() 会取该周最后一个可用交易日
    weekly_close = close.resample("W-FRI").last()

    ema_fast = weekly_close.ewm(span=fast_span_weeks, adjust=False).mean()#14周EMA
    ema_slow = weekly_close.ewm(span=slow_span_weeks, adjust=False).mean()#60周EMA

    # 14周EMA 由 >= 60周EMA 变为 < 60周EMA
    wk_cross = (ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)

    # 映射到日频：只在“该周最后一个交易日”打信号
    sig = pd.DataFrame(False, index=daily_idx, columns=close.columns)

    for w in wk_cross.index:
        # w 是周标签（周五），先映射到不晚于 w 的最近交易日
        sd = nearest_trading_day(daily_idx, w)
        if sd in sig.index:
            # 只在 sd 这一天置 True（不向后扩散）
            
            sig.loc[sd] = sig.loc[sd] | wk_cross.loc[w].reindex(sig.columns).fillna(False)

    return sig


def select_stocks(
    close: pd.DataFrame,
    vol: pd.DataFrame,
    mktcap: pd.DataFrame,
    st_flag: pd.DataFrame | pd.Series | None,
    turn: pd.DataFrame | None,
    t: pd.Timestamp,
    ma_win: int = 120,
    ma_dev: float = 0.10,# 均线偏离度约束：不超过10%
    spike_mult: float = 2.0,  # 放量阈值：> spike_mult * baseline
    min_spike_days: int = 4,  # 放量天数阈值
    liq_window: int = 20,
    min_turnover: float = 0.001,  # 日均换手率约束：不小于0.1%
) -> list[str]:

    idx = close.index
    t = nearest_trading_day(idx, pd.Timestamp(t))  # t 不一定是交易日，先对齐到最近交易日

    # 锚点 a = t - 6 个月
    a_raw = t - pd.DateOffset(months=6)
    
    a = nearest_trading_day(idx, a_raw)  

    # 锚点窗口：前后各 1 个月
    w_start = nearest_trading_day(idx, a - pd.DateOffset(months=1))
    w_end   = nearest_trading_day(idx, a + pd.DateOffset(months=1))

    baseline_vol = vol.loc[w_start : w_end].mean()  # 锚点窗口内日均成交量

    vol_half_year = vol.loc[a:t]
    spike_cnt = (vol_half_year.gt(spike_mult * baseline_vol, axis=1)).sum()  # 半年内放量天数

    pass_vol = spike_cnt >= min_spike_days

    # ST 过滤：st_flag=True 视为 ST
    if st_flag is None:
        pass_st = pd.Series(True, index=close.columns, dtype=bool)
    elif isinstance(st_flag, pd.Series):
        pass_st = ~st_flag.reindex(close.columns).fillna(False)
    else:
        st_t = st_flag.loc[t] if t in st_flag.index else st_flag.reindex([t]).iloc[0]
        pass_st = ~st_t.reindex(close.columns).fillna(False)

    # 停牌过滤：当日成交量 <= 0 或 NaN 视为不可交易
    vol_t = vol.loc[t]
    pass_trade = vol_t.notna() & (vol_t > 0)

    # 均线偏离度
    ma_windows = (20, 60, 120, 250)  # 四条均线，20，60，120，250日线
    
    p_t = close.loc[t]#t天时的收盘价
    
    pass_ma = pd.Series(True, index=close.columns, dtype=bool)
    for w in ma_windows:
        ma_w = close.rolling(w, min_periods=w).mean().loc[t]
        dev_w = (p_t.sub(ma_w).abs()).div(ma_w)#计算t天时的收盘价和四条均线的偏离度
        pass_ma = pass_ma & (dev_w <= ma_dev) & ma_w.notna()

    # 金叉条件：MA20 在 t 日上穿 MA60
    fast_win, slow_win = 20, 60
    ma_fast = close.rolling(fast_win, min_periods=fast_win).mean()
    ma_slow = close.rolling(slow_win, min_periods=slow_win).mean()

    pos = idx.get_indexer([t])[0]
    if pos <= 0:
        pass_golden = pd.Series(False, index=close.columns, dtype=bool)
    else:
        t_prev = idx[pos - 1]
        pass_golden = (ma_fast.loc[t_prev] <= ma_slow.loc[t_prev]) & (ma_fast.loc[t] > ma_slow.loc[t])
        pass_golden = pass_golden.fillna(False)

    # 流动性过滤：近 liq_window 日均换手率
    if turn is not None:
        avg_turn = turn.rolling(liq_window, min_periods=liq_window).mean().loc[t]
    else:
        turnover = (vol * close) / mktcap
        turnover = turnover.replace([np.inf, -np.inf], np.nan)
        avg_turn = turnover.rolling(liq_window, min_periods=liq_window).mean().loc[t]
    pass_liq = avg_turn.notna() & (avg_turn >= min_turnover)

    ok = (
        pass_vol
        & pass_ma
        & pass_golden
        & pass_st
        & pass_trade
        & pass_liq
        & p_t.notna()
        & baseline_vol.notna()
    )  # 排除 NA、停牌和低流动性样本

    return ok.loc[ok].index.tolist()

def top_n_by_mktcap(
    picked: list[str],
    mktcap_row: pd.Series,
    n: int = 20 #最高市值的20只股票
) -> list[str]:
    if len(picked) <= n:
        return picked
    s = mktcap_row.reindex(picked)
    s = s.dropna()
    if s.empty:  # 市值全缺失时，退化为取前 n 个
        return picked[:n]
    return s.nlargest(n).index.tolist()

def backtest_cycle_stock_selection(
    close: pd.DataFrame,
    vol: pd.DataFrame,
    mktcap: pd.DataFrame,
    st_flag: pd.DataFrame | pd.Series | None,
    turn: pd.DataFrame | None,
    ma_win: int,
    ma_dev: float,
    spike_mult: float,
    min_spike_days: int,
    init_cash: float = 1_000_000.0,
    top_n = 20,
) -> tuple[pd.Series, pd.DataFrame]:
    if not isinstance(close.index, pd.DatetimeIndex) or not isinstance(vol.index, pd.DatetimeIndex):
        raise TypeError("close.index / vol.index 必须是 DatetimeIndex")

    idx = close.index.intersection(vol.index).intersection(mktcap.index)  # 三者共同交易日
    close = close.loc[idx].sort_index()
    vol = vol.loc[idx].sort_index()
    mktcap = mktcap.loc[idx].sort_index()
    
    close_ff = close.sort_index().ffill()#时间序列对齐后的dataframe可能仍会有close的缺失，我们这里将缺失的数据向前对齐

    ema_cross_down = weekly_ema_crossunder_to_daily_signal(
        close=close,
        daily_idx=idx,
        fast_span_weeks=14,
        slow_span_weeks=60,
    )

    # --- 回测状态变量 ---
    cash = float(init_cash)
    positions: dict[str, float] = {}  # 当前持仓：ticker -> shares
    nav_list = []
    trades = []
    
    # 周期管理
    cycle_id = 0
    cycle_target_n: int = 0
    cycle_signal_day: pd.Timestamp | None = None      # 本周期选股日（信号日）
    cycle_deadline: pd.Timestamp | None = None        # 本周期 6 个月截止日（对齐交易日）
    cycle_initial_set: set[str] = set()               # 本周期首次买入集合
    cycle_sold_set: set[str] = set()                  # 本周期已卖出集合（仅统计初始集合）
    banned_in_cycle: set[str] = set()                 # 周期内卖出后禁止回补

    # 挂单：统一在“下一交易日收盘”执行，避免未来函数
    pending_buys: dict[str, pd.Timestamp] = {}        # 待买：ticker -> exec_day
    pending_sells: dict[str, pd.Timestamp] = {}       # 待卖：ticker -> exec_day

    # 选股起点：保证 SMA/EMA 计算有足够历史
    start_i = max(ma_win + 160, 300)  # 至少预留 300 个交易日（60周EMA，每周五个交易日，对应300；往前推6个月大约是160个交易日）
    if start_i >= len(idx):
        raise ValueError("数据太短，不足以支持 ma_win、6个月窗口与EMA计算")

    # 初始化第一周期：在 idx[start_i] 这天产生选股信号
    cycle_signal_day = idx[start_i]
    cycle_deadline = nearest_trading_day(idx, cycle_signal_day + pd.DateOffset(months=6))

    # 信号日选股，下一个交易日执行买入
    picked_all = select_stocks(
        close, vol, mktcap, st_flag, turn, cycle_signal_day,
        ma_win=ma_win, ma_dev=ma_dev, 
        spike_mult=spike_mult, min_spike_days=min_spike_days
    )
    
    mktcap_row = mktcap.loc[cycle_signal_day]
    if len(picked_all) > top_n:
        picked = top_n_by_mktcap(picked_all, mktcap_row, n=top_n)
    else:
        picked = picked_all
    
    cycle_target_n = min(top_n, len(picked))  # 本周期目标持仓数
    
    exec_day = next_trading_day(idx, cycle_signal_day)
    if exec_day is not None:
        for tk in picked:
            pending_buys[tk] = exec_day# pending_buys: ticker -> 预定执行日，是一个dict

    # 主循环
    for d in idx[start_i:]:
        # 1) 执行到期卖单（收盘执行）
        to_sell_now = []
        for tk, ed in pending_sells.items():
            if ed == d:
                to_sell_now.append(tk)
        
        for tk in to_sell_now:
            if tk in close.columns:
                price = close.at[d, tk]
            else:
                price = np.nan
            if tk in positions and pd.notna(price) and price > 0:
                shares = positions.pop(tk)
                cash = cash + shares * float(price)

                trades.append({
                    "date": d, "cycle": cycle_id, "ticker": tk, "side": "SELL",
                    "price": float(price), "shares": float(shares),
                })

                # 统计本周期卖出数量（仅统计初始买入集合）
                if tk in cycle_initial_set:
                    cycle_sold_set.add(tk)

                # 周期内卖出后禁止回补
                banned_in_cycle.add(tk)

                pending_sells.pop(tk, None)
            else:
                nd2 = next_trading_day(idx, d)
                if nd2 is None:
                    pending_sells.pop(tk, None)
                else:
                    pending_sells[tk] = nd2

        # 2) 执行到期买单（收盘执行）
        to_buy_now = []
        for tk, ed in pending_buys.items():
            if ed == d:
                to_buy_now.append(tk)
        if to_buy_now:
             # 仅买未持仓且未被本周期禁买的标的
            candidates = []
            for tk in to_buy_now:
                if tk in positions or tk in banned_in_cycle:
                    continue
                if tk in close.columns:
                    price = close.at[d, tk]
                else:
                    price = np.nan
                if pd.notna(price) and price > 0:
                    candidates.append(tk)

            # 现金等权一次性买入：买后不加仓，卖后现金留存
            bought = set()
            bought_cnt = len(bought)#记录这一天（d）实际买入了多少只票
            if candidates and cash > 0:
                alloc = cash / len(candidates)#等权买入，alloc计算每只股票分到的资金预算
                for tk in candidates:
                    bought_cnt = bought_cnt + 1
                    price = float(close.at[d, tk])
                    shares = alloc / price
                    positions[tk] = shares
                    cash = cash - shares * price
                    bought.add(tk)
                    trades.append({
                        "date": d, "cycle": cycle_id, "ticker": tk, "side": "BUY",
                        "price": price, "shares": float(shares),
                    })
            
            if bought:
                cycle_target_n = max(cycle_target_n, len(positions))

            # 清理挂单,买到的就删掉，买不到的则顺延
            for tk in to_buy_now:
                if tk in bought:
                    pending_buys.pop(tk, None)
                else:
                    nd2 = next_trading_day(idx, d)
                    if nd2 is None:
                        pending_buys.pop(tk, None)
                    else:
                        pending_buys[tk] = nd2

            # 若这是本周期第一次真实买入，记录 initial_set
            if not cycle_initial_set:
                cycle_initial_set = set(positions.keys())
                cycle_target_n = len(cycle_initial_set)  # 记录本周期目标只数

        # 3) 生成卖出挂单：当日死叉，下一交易日卖出
        # 仅对当前持仓且未挂卖单的标的生效
        nd = next_trading_day(idx, d)
        
        if nd is not None:
            for tk in list(positions.keys()):
                if tk in pending_sells:
                    continue
                # 死叉信号在 d 发生，则在 nd 执行卖出
                if tk in ema_cross_down.columns and bool(ema_cross_down.at[d, tk]):
                    pending_sells[tk] = nd
        # 4) 计算当日 NAV（收盘后状态）
        # 组合净值 = 现金 + 持仓市值
        port_val = cash
        if positions:
            prices = close_ff.loc[d].reindex(list(positions.keys()))
            for tk, sh in positions.items():
                px = prices.get(tk, np.nan)  # 若该票当日无价格则返回 NaN
                if pd.notna(px) and px > 0:
                    port_val = port_val + float(px) * float(sh)
                else:
                    # 个别股票当日缺价时，保守按 0 计
                    port_val = port_val + 0.0

        nav_list.append((d, port_val))

        # 5) 周期结束条件：满 6 个月，或已卖出达到一半
        time_end = (cycle_deadline is not None and d >= cycle_deadline)

        half_end = False
        if cycle_target_n > 0:
            need = int(np.ceil((cycle_target_n) / 2))
            if len(cycle_sold_set) >= need:
                half_end = True

        if time_end or half_end:# 周期结束，重置状态
            pending_buys.clear()
            pending_sells.clear()
            cycle_id = cycle_id + 1
            
            cycle_signal_day = d
            cycle_deadline = nearest_trading_day(idx, cycle_signal_day + pd.DateOffset(months=6))
            cycle_initial_set = set()
            cycle_target_n = 0
            cycle_sold_set = set()
            banned_in_cycle = set()
            
            picked_all = select_stocks(
                close, vol, mktcap, st_flag, turn, cycle_signal_day,
                ma_win=ma_win, ma_dev=ma_dev,
                spike_mult=spike_mult, min_spike_days=min_spike_days
            )
            
            # 从新入选且当前未持仓的标的中补仓
            new_candidates = [tk for tk in picked_all if tk not in positions]
            
            # 需要补买的数量
            need_buy = max(top_n - len(positions), 0)
            
            if need_buy <= 0:
                new_names = []
            else:
                if len(positions) + len(new_candidates) <= top_n:  # 总数不超过 top_n
                    new_names = new_candidates
                else:
                    mktcap_row = mktcap.loc[cycle_signal_day]
                    new_names = top_n_by_mktcap(new_candidates, mktcap_row, n=need_buy)
            
            cycle_target_n = min(top_n, len(positions) + len(new_names))
            
            exec_day = next_trading_day(idx, cycle_signal_day)
            if exec_day is not None:
                for tk in new_names:
                    pending_buys[tk] = exec_day
                    
    nav = pd.Series(dict(nav_list)).sort_index()
    nav = nav / nav.iloc[0]  # 归一化净值

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["date", "cycle", "ticker"]).reset_index(drop=True)

    return nav, trades_df

def perf_summary(nav: pd.Series, freq: int = 252) -> dict:  # 绩效统计
    ret = nav.pct_change().dropna()
    if ret.empty:
        return{'CAGR': np.nan, 'VOL': np.nan, 'SHARPE': np.nan, 'MAXDD': np.nan}
    
    years = (nav.index[-1] - nav.index[0]).days / 365.25  # 按自然年换算，每四年一个闰年
    if years > 0:
        cagr = float(nav.iloc[-1] ** (1 / years)) - 1
    else:
        cagr = np.nan
    vol = float(ret.std() * np.sqrt(freq))
    if ret.std() > 0:    
        sharpe = float((ret.mean() / ret.std()) * np.sqrt(freq))
    else:
        sharpe = np.nan
    dd = nav / nav.cummax() - 1.0
    maxdd = float(dd.min())
    return{'CAGR': cagr, 'VOL': vol, 'SHARPE': sharpe, 'MAXDD': maxdd}

def make_fixed_param_rolling_splits(
    idx: pd.DatetimeIndex,
    warmup_months: int = 36,
    test_months: int = 6,
    step_months: int | None = None,
    min_test_days: int = 40,
    fixed_test_end: str | pd.Timestamp | None = None,
) -> list[dict[str, pd.Timestamp]]:
    """构建固定参数的滚动窗口（walk-forward）切分。"""
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("idx must be DatetimeIndex")
    if len(idx) == 0:
        return []
    if warmup_months <= 0 or test_months <= 0:
        raise ValueError("warmup_months and test_months must be positive")
    if min_test_days <= 0:
        raise ValueError("min_test_days must be positive")

    if step_months is None:
        step_months = test_months
    if step_months <= 0:
        raise ValueError("step_months must be positive")

    idx = pd.DatetimeIndex(idx).sort_values().unique()
    splits: list[dict[str, pd.Timestamp]] = [] # splits：又dict构成的list

    fixed_end_pos: int | None = None
    if fixed_test_end is not None:
        fixed_end_target = pd.Timestamp(fixed_test_end)
        fixed_end_pos = int(idx.searchsorted(fixed_end_target, side="right")) - 1
        if fixed_end_pos < 0:
            raise ValueError("fixed_test_end earlier than first trading day in idx")

    first_test_target = idx[0] + pd.DateOffset(months=warmup_months)
    test_pos = int(idx.searchsorted(first_test_target, side="left"))#表当前滚动窗口测试段起点在idx交易日索引里的位置

    while test_pos < len(idx):
        test_start = idx[test_pos]

        history_target = test_start - pd.DateOffset(months=warmup_months)# 理论上的训练起始日（可能非交易日）
        history_pos = int(idx.searchsorted(history_target, side="left"))# 训练起始日的时间下标
        history_start = idx[history_pos]# 实际训练起始日（history_target映射至交易日）

        if fixed_end_pos is None:
            test_end_target = test_start + pd.DateOffset(months=test_months)
            test_end_pos = int(idx.searchsorted(test_end_target, side="right")) - 1#表示测试终点位置
        else:
            test_end_pos = fixed_end_pos
        if test_end_pos < test_pos:
            break#测试起点在终点之后，此时测试无效，虽然理论上基本不会发生

        test_end_pos = min(test_end_pos, len(idx) - 1)
        if (test_end_pos - test_pos + 1) < min_test_days:
            break
        test_end = idx[test_end_pos]

        splits.append(
            {
                "history_start": history_start,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

        next_test_target = test_start + pd.DateOffset(months=step_months)
        next_pos = int(idx.searchsorted(next_test_target, side="left"))
        if next_pos <= test_pos:
            next_pos = test_pos + 1
        test_pos = next_pos

    return splits


def backtest_fixed_param_rolling_windows(
    close: pd.DataFrame,
    vol: pd.DataFrame,
    mktcap: pd.DataFrame,
    st_flag: pd.DataFrame | pd.Series | None,
    turn: pd.DataFrame | None,
    ma_win: int,
    ma_dev: float,
    spike_mult: float,
    min_spike_days: int,
    *,
    warmup_months: int = 36,
    test_months: int = 6,
    step_months: int | None = None,
    min_test_days: int = 40,# 最短回测时长，避免失真
    fixed_test_end: str | pd.Timestamp | None = None,
    init_cash: float = 1_000_000.0,
    top_n: int = 20,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    idx = close.index.intersection(vol.index).intersection(mktcap.index).sort_values()
    if len(idx) == 0:
        raise ValueError("No common dates among close/vol/mktcap")

    splits = make_fixed_param_rolling_splits(
        idx=idx,
        warmup_months=warmup_months,
        test_months=test_months,
        step_months=step_months,
        min_test_days=min_test_days,
        fixed_test_end=fixed_test_end,
    )
    if not splits:
        raise ValueError("No rolling splits generated. Try reducing warmup/test windows.")

    nav_parts: list[pd.Series] = []
    trade_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    stitched_level = 1.0
    min_hist_days = max(ma_win + 160, 300) + 1

    for fold_id, sp in enumerate(splits):
        slc = slice(sp["history_start"], sp["test_end"]) # 创建回测用到时间区间的切片对象
        close_fold = close.loc[slc]
        vol_fold = vol.loc[slc]
        mktcap_fold = mktcap.loc[slc]
        if len(close_fold) < min_hist_days:
            continue

        if isinstance(st_flag, pd.DataFrame):
            st_fold = st_flag.loc[slc]
        else:
            st_fold = st_flag

        if isinstance(turn, pd.DataFrame):
            turn_fold = turn.loc[slc]
        else:
            turn_fold = turn

        nav_fold, trades_fold = backtest_cycle_stock_selection(
            close_fold,
            vol_fold,
            mktcap_fold,
            st_fold,
            turn_fold,
            ma_win=ma_win,
            ma_dev=ma_dev,
            spike_mult=spike_mult,
            min_spike_days=min_spike_days,
            init_cash=init_cash,
            top_n=top_n,
        )

        nav_test = nav_fold.loc[sp["test_start"]:sp["test_end"]]
        if nav_test.empty:
            continue

        nav_test_rebased = nav_test / nav_test.iloc[0]
        fold_stats = perf_summary(nav_test_rebased)

        fold_rows.append(
            {
                "fold": fold_id,
                "history_start": sp["history_start"],
                "test_start": sp["test_start"],
                "test_end": sp["test_end"],
                "test_days": int(len(nav_test_rebased)),
                "test_return": float(nav_test_rebased.iloc[-1] - 1.0),
                "CAGR": fold_stats["CAGR"],
                "VOL": fold_stats["VOL"],
                "SHARPE": fold_stats["SHARPE"],
                "MAXDD": fold_stats["MAXDD"],
            }
        )

        if not trades_fold.empty:
            trades_oos = trades_fold[
                (trades_fold["date"] >= sp["test_start"])
                & (trades_fold["date"] <= sp["test_end"])
            ].copy()
            if not trades_oos.empty:
                trades_oos["fold"] = fold_id
                trade_parts.append(trades_oos)

        nav_stitched = nav_test_rebased * stitched_level
        if nav_parts:
            last_day = nav_parts[-1].index[-1]
            nav_stitched = nav_stitched.loc[nav_stitched.index > last_day]
        if nav_stitched.empty:
            continue

        stitched_level = float(nav_stitched.iloc[-1])
        nav_parts.append(nav_stitched)

    if not nav_parts:
        raise ValueError("No valid rolling fold produced NAV. Check data length and permissions.")

    nav_oos = pd.concat(nav_parts).sort_index()
    nav_oos = nav_oos.loc[~nav_oos.index.duplicated(keep="first")]

    if trade_parts:
        trades_all = pd.concat(trade_parts, ignore_index=True)
        trades_all = trades_all.sort_values(["date", "fold", "ticker"]).reset_index(drop=True)
    else:
        trades_all = pd.DataFrame()

    fold_report = pd.DataFrame(fold_rows)
    if not fold_report.empty:
        fold_report = fold_report.sort_values("fold").reset_index(drop=True)

    return nav_oos, trades_all, fold_report


def plot_nav(nav: pd.Series):  # 绘制净值曲线
    plt.figure()
    nav.plot()
    plt.title('NAV')
    plt.xlabel('DATE')
    plt.ylabel('NET VALUE')
    plt.tight_layout()
    plt.show()


def main(
    ma_win,
    ma_dev,
    spike_mult,
    min_spike_days,
    warmup_months=36,
    test_months=6,
    step_months=6,
    min_test_days=40,
    fixed_test_end=None,
):
    start = '2021-01-01'
    end = '2026-02-04'

    close, vol, mktcap, st_flag, turn = build_market_close_vol_mktcap(
        start,
        end,
        asof_date=pd.Timestamp(end).strftime('%Y%m%d'),
        batch_hq=300,
        batch_shares=200,
        cps=1,
    )

    nav, trades_df, fold_report = backtest_fixed_param_rolling_windows(
        close,
        vol,
        mktcap,
        st_flag,
        turn,
        ma_win=ma_win,
        ma_dev=ma_dev,
        spike_mult=spike_mult,
        min_spike_days=min_spike_days,
        warmup_months=warmup_months,
        test_months=test_months,
        step_months=step_months,
        min_test_days=min_test_days,
        fixed_test_end=fixed_test_end,
        init_cash=1_000_000.0,
        top_n=20,
    )

    stats = perf_summary(nav)

    print('=== PERFORMANCE SUMMARY ===')
    print(stats)
    print("\n=== NAV tail ===")
    print(nav.tail())

    print("\n=== Trades tail ===")
    print(trades_df.tail(10) if not trades_df.empty else trades_df)

    print("\n=== Fold report tail ===")
    print(fold_report.tail(10) if not fold_report.empty else fold_report)

    plot_nav(nav)
    return nav, trades_df, fold_report


nav, trades, folds = main(
    ma_win=120,
    ma_dev=0.10,
    spike_mult=2.0,
    min_spike_days=4,
    warmup_months=36,
    test_months=6,
    step_months=6,
    min_test_days=40,
    fixed_test_end="2026-01-01",
)

print("\n=== Trades last 20 ===")
print(trades.tail(20))

print("\n=== Folds last 20 ===")
print(folds.tail(20))