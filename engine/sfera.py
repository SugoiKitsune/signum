"""Sfera DB adapter for Signum charts.

Thin facade over sfera_db — returns DataFrames / Series that plug directly into
Chart methods (candlestick, line, area, forecast, …).

The Sfera DB is a PostgreSQL database with multiple schemas.  Use the discovery
methods first to understand what's available, then use ``read()`` or ``query()``
to pull the data you need.

Discovery workflow
------------------
    from signum import sfera

    sfera.schemas()                          # list all schemas
    sfera.tables("bbgidx")                   # list tables in a schema
    sfera.columns("index_prices", "bbgidx")  # list columns of a table

Generic read (works against any schema / table)
-----------------------------------------------
    df = sfera.read("index_total_return", schema="bbgidx",
                    ticker="CACT", start="2015-01-01")

    df = sfera.read("my_table", schema="my_schema")   # no ticker filter

Raw SQL (full power — no assumptions)
--------------------------------------
    df = sfera.query("SELECT * FROM bbgidx.index_prices WHERE ticker = %s", ["CAC"])

Named convenience helpers (only for tables we know exist in bbgidx)
---------------------------------------------------------------------
    ohlc = sfera.ohlc("CAC",  start="2020-01-01")   # → candlestick
    tr   = sfera.total_return("CACT", start="2020-01-01")  # → line / forecast anchor
    ivol = sfera.ivol("CAC",  start="2020-01-01")    # → sub-pane line

Note: sfera_db is imported lazily — signum can be used without it.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def _sfera_db():
    """Lazy import of sfera_db with a clear error if it is not on sys.path."""
    try:
        import sfera_db as _sd
        return _sd
    except ImportError:
        raise ImportError(
            "sfera_db is not installed or not on sys.path.\n"
            "Add it before importing signum:\n"
            "    import sys\n"
            "    sys.path.insert(0, '/path/to/sfera-db')\n"
            "or install it with:  uv pip install -e /path/to/sfera-db"
        )


class SferaData:
    """Sfera DB data adapter — returns DataFrames ready for Signum charts.

    Use the module-level singleton:
        from signum import sfera
    """

    # ── Discovery ──────────────────────────────────────────────────────────

    def schemas(self) -> pd.DataFrame:
        """List all non-system schemas in the database.

        Example
        -------
            sfera.schemas()
            #   schema_name
            # 0       bbgidx
            # 1     equities
            # ...
        """
        return _sfera_db().schemas()

    def tables(self, schema: str = "bbgidx") -> pd.DataFrame:
        """List tables in *schema*.

        Example
        -------
            sfera.tables("bbgidx")
        """
        return _sfera_db().tables(schema=schema)

    def columns(self, table: str, schema: str = "bbgidx") -> pd.DataFrame:
        """List columns (name + data_type + position) for *schema.table*.

        Example
        -------
            sfera.columns("index_prices", "bbgidx")
        """
        return _sfera_db().columns(table, schema=schema)

    # ── Generic read ───────────────────────────────────────────────────────

    def query(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        """Execute arbitrary SQL and return a DataFrame.

        Example
        -------
            sfera.query(
                "SELECT trade_date, close_price FROM bbgidx.index_total_return "
                "WHERE ticker = %s ORDER BY trade_date",
                ["CACT"],
            )
        """
        return _sfera_db().query(sql, params)

    def read(
        self,
        table: str,
        schema: str = "bbgidx",
        ticker: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        date_col: str = "trade_date",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """Generic table reader — works against any schema / table.

        Returns a DataFrame indexed by *date_col* (when it exists).
        No hardcoded column names — whatever is in the table comes back as-is.

        Parameters
        ----------
        table      : Table name (without schema prefix).
        schema     : Schema name.  Default ``"bbgidx"``.
        ticker     : Optional ticker filter applied to *ticker_col*.
        start / end: Optional date range filters applied to *date_col*.
        date_col   : Date column to filter and use as index.  Default ``"trade_date"``.
        ticker_col : Column used for the ticker filter.  Default ``"ticker"``.

        Examples
        --------
            sfera.read("index_total_return", ticker="CACT", start="2015-01-01")
            sfera.read("index_prices",       ticker="CAC",  schema="bbgidx")
            sfera.read("my_table", schema="my_schema")   # no filters
        """
        return _sfera_db().read_table(
            table,
            schema=schema,
            ticker=ticker,
            ticker_col=ticker_col,
            start=start,
            end=end,
            date_col=date_col,
        )

    # ── Named helpers (bbgidx only — confirmed tables) ─────────────────────

    def ohlc(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """OHLCV from ``bbgidx.index_prices`` — feeds ``Chart.candlestick()``.

        Returns a DataFrame indexed by ``trade_date`` with columns
        ``open``, ``high``, ``low``, ``close``, ``volume``.

        Example
        -------
            Chart(theme="dark").candlestick(sfera.ohlc("CAC", start="2015-01-01"))
        """
        df = _sfera_db().index_prices(ticker=ticker, start=start, end=end)
        return df.drop(columns=["ticker"], errors="ignore")

    def total_return(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """Close from ``bbgidx.index_total_return`` — feeds ``Chart.line()`` / forecast anchor.

        Returns a Series indexed by ``trade_date``.

        Example
        -------
            cact = sfera.total_return("CACT", start="2015-01-01")
            Chart(theme="dark").line(cact, name="CACT TR")
        """
        df = _sfera_db().read_table(
            "index_total_return", schema="bbgidx",
            ticker=ticker, start=start, end=end,
        )
        # pick close_price if present, otherwise the first numeric column
        if "close_price" in df.columns:
            return df["close_price"].rename("close")
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            raise ValueError(
                f"No numeric columns found in bbgidx.index_total_return for ticker={ticker!r}. "
                f"Columns: {list(df.columns)}"
            )
        return df[numeric_cols[0]].rename("close")

    def ivol(
        self,
        ticker: str,
        col: str = "3m_50d_ivol",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """Implied vol from ``bbgidx.index_implied_vol`` — feeds a sub-pane line.

        Returns a Series indexed by ``trade_date``.

        Parameters
        ----------
        col : Column name.  Default ``"3m_50d_ivol"``.
              Use ``sfera.columns("index_implied_vol")`` to see all available columns.

        Example
        -------
            ivol_pane = Chart(theme="dark", height=150).line(
                sfera.ivol("CAC", start="2015-01-01"), name="IVol 3m50d"
            )
        """
        df = _sfera_db().index_ivol(ticker=ticker, start=start, end=end)
        # index_ivol() already aliases "3m_50d_ivol" → "ivol"
        if col in df.columns:
            return df[col].rename(col)
        if "ivol" in df.columns:
            return df["ivol"].rename("ivol")
        raise ValueError(
            f"Column {col!r} not found in bbgidx.index_implied_vol. "
            f"Available: {list(df.columns)}.  "
            f"Use sfera.columns('index_implied_vol') to inspect."
        )


# Module-level singleton:
#   from signum import sfera
#   sfera.schemas()
#   sfera.ohlc("CAC", start="2020-01-01")
sfera = SferaData()
