"""
LiveData.py - Updated for Options Chain Snapshot
This module uses the official Polygon REST client to retrieve options snapshot data,
and includes functions to return a narrowed list of option details.
"""

import logging
from polygon import RESTClient

# Replace with your actual Polygon.io API key.
API_KEY = "_7uZSzoHV3_SQZNamkhtn5pOT4OHWy8f"

def get_narrowed_options_chain(ticker, expiration_start, expiration_end, strike_min, strike_max, limit=10):
    """
    Retrieve a snapshot options chain for the given ticker with filtering parameters,
    then narrow the results to only return a subset of fields.

    Parameters:
        ticker (str): The underlying ticker (e.g., "SPY")
        expiration_start (str): The earliest expiration date to consider (YYYY-MM-DD)
        expiration_end (str): The latest expiration date to consider (YYYY-MM-DD)
        strike_min (float or int): The minimum strike price
        strike_max (float or int): The maximum strike price
        limit (int): Maximum number of contracts to retrieve (default 10)

    Returns:
        List[dict]: A list of dictionaries where each dictionary contains:
            - option_ticker: The option contract symbol.
            - expiration_date: The expiration date.
            - strike_price: The strike price.
            - contract_type: "call" or "put".
            - delta: The option delta.
            - theta: The option theta.
            - open_interest: Open interest.
            - last_close: The last close price from the day snapshot.
    """
    client = RESTClient(api_key=API_KEY, trace=False)
    params = {
        "expiration_date.gte": expiration_start,
        "expiration_date.lte": expiration_end,
        "strike_price.gte": strike_min,
        "strike_price.lte": strike_max,
        "limit": limit
    }

    try:
        # list_snapshot_options_chain returns a generator; convert it to a list.
        options_chain = list(client.list_snapshot_options_chain(ticker, params=params))
    except Exception as e:
        logging.error("Error retrieving options chain for %s: %s", ticker, e)
        return []

    narrowed = []
    for option in options_chain:
        try:
            # Each option is returned as an OptionContractSnapshot object.
            details = option.details
            greeks = option.greeks
            day = option.day
            narrowed.append({
                "option_ticker": details.ticker,
                "expiration_date": details.expiration_date,
                "strike_price": details.strike_price,
                "contract_type": details.contract_type,
                "delta": greeks.delta if greeks else None,
                "theta": greeks.theta if greeks else None,
                "open_interest": option.open_interest,
                "last_close": day.close if day else None
            })
        except Exception as ex:
            logging.error("Error processing option snapshot for %s: %s", ticker, ex)

    return narrowed

def get_live_options_snapshot(ticker):
    """
    Retrieve a "live" options snapshot for the given ticker using preset filtering parameters.
    
    Parameters:
        ticker (str): The underlying ticker (e.g., "SPY")
    
    Returns:
        List[dict]: The narrowed list of option contract details.
    """
    # Define default filtering parameters.
    expiration_start = "2025-07-01"  # Future expiration start date
    expiration_end   = "2025-12-31"  # Future expiration end date
    strike_min       = 390          # Example strike range (adjust as needed)
    strike_max       = 410
    limit            = 20

    return get_narrowed_options_chain(ticker, expiration_start, expiration_end, strike_min, strike_max, limit)

def process_etf_options(ticker, snapshot):
    """
    Process the snapshot (a list of option dictionaries) for the given ticker.
    For simplicity, this function returns the first option contract from the snapshot.
    
    Parameters:
        ticker (str): The underlying ticker.
        snapshot (List[dict]): The list of option contracts returned by get_live_options_snapshot.
    
    Returns:
        dict or None: The first option contract dictionary, or None if no contracts are found.
    """
    if not snapshot or not isinstance(snapshot, list) or len(snapshot) == 0:
        return None
    return snapshot[0]

# Create an alias if you want; in this example process_etf_options is defined separately.
# process_etf_options = get_narrowed_options_chain  # (Do not use alias in this case.)

# Example usage for testing (wrap in __main__ for production code)
if __name__ == "__main__":
    ticker = "SPY"
    options = get_live_options_snapshot(ticker)
    logging.info("Live options snapshot for %s: Found %d contracts.", ticker, len(options))
    processed = process_etf_options(ticker, options)
    logging.info("Processed option data for %s:", ticker)
    logging.info(processed)
