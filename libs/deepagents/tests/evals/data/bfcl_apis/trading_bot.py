import random
from copy import deepcopy
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Union

from .long_context import (
    AUTOMOBILE_EXTENSION,
    MA_5_EXTENSION,
    MA_20_EXTENSION,
    ORDER_DETAIL_EXTENSION,
    TECHNOLOGY_EXTENSION,
    TRANSACTION_HISTORY_EXTENSION,
    WATCH_LIST_EXTENSION,
)

CURRENT_TIME = datetime(2024, 9, 1, 10, 30)

DEFAULT_STATE = {
    "orders": {
        12345: {
            "id": 12345,
            "order_type": "Buy",
            "symbol": "AAPL",
            "price": 210.65,
            "amount": 10,
            "status": "Completed",
        },
        12446: {
            "id": 12446,
            "order_type": "Sell",
            "symbol": "GOOG",
            "price": 2840.56,
            "amount": 5,
            "status": "Pending",
        },
    },
    "account_info": {
        "account_id": 12345,
        "balance": 10000.0,
        "binding_card": 1974202140965533,
    },
    "authenticated": False,
    "market_status": "Closed",
    "order_counter": 12446,
    "stocks": {
        "AAPL": {
            "price": 227.16,
            "percent_change": 0.17,
            "volume": 2.552,
            "MA(5)": 227.11,
            "MA(20)": 227.09,
        },
        "GOOG": {
            "price": 2840.34,
            "percent_change": 0.24,
            "volume": 1.123,
            "MA(5)": 2835.67,
            "MA(20)": 2842.15,
        },
        "TSLA": {
            "price": 667.92,
            "percent_change": -0.12,
            "volume": 1.654,
            "MA(5)": 671.15,
            "MA(20)": 668.20,
        },
        "MSFT": {
            "price": 310.23,
            "percent_change": 0.09,
            "volume": 3.234,
            "MA(5)": 309.88,
            "MA(20)": 310.11,
        },
        "NVDA": {
            "price": 220.34,
            "percent_change": 0.34,
            "volume": 1.234,
            "MA(5)": 220.45,
            "MA(20)": 220.67,
        },
        "ALPH": {
            "price": 1320.45,
            "percent_change": -0.08,
            "volume": 1.567,
            "MA(5)": 1321.12,
            "MA(20)": 1325.78,
        },
        "OMEG": {
            "price": 457.23,
            "percent_change": 0.12,
            "volume": 2.345,
            "MA(5)": 456.78,
            "MA(20)": 458.12,
        },
        "QUAS": {
            "price": 725.89,
            "percent_change": -0.03,
            "volume": 1.789,
            "MA(5)": 726.45,
            "MA(20)": 728.00,
        },
        "NEPT": {
            "price": 88.34,
            "percent_change": 0.19,
            "volume": 0.654,
            "MA(5)": 88.21,
            "MA(20)": 88.67,
        },
        "SYNX": {
            "price": 345.67,
            "percent_change": 0.11,
            "volume": 2.112,
            "MA(5)": 345.34,
            "MA(20)": 346.12,
        },
        "ZETA": {
            "price": 22.09,
            "percent_change": -0.05,
            "volume": 0.789,
            "MA(5)": 22.12,
            "MA(20)": 22.34,
        },
    },
    "watch_list": ["NVDA"],
    "transaction_history": [],
    "random_seed": 1053520,
}


class TradingBot:
    """
    A class representing a trading bot for executing stock trades and managing a trading account.

    Attributes:
        orders (Dict[int, Dict[str, Union[str, float, int]]]): A dictionary of orders for purchasing and selling of stock, keyed by order ID.
        account_info (Dict[str, Union[int, float]]): Information about the trading account.
        authenticated (bool): Whether the user is currently authenticated.
        market_status (str): The current status of the market ('Open' or 'Closed').
        order_counter (int): A counter for generating unique order IDs.
        stocks (Dict[str, Dict[str, Union[float, int]]]): Information about various stocks.
        watch_list (List[str]): A list of stock symbols being watched.
        transaction_history (List[Dict[str, Union[str, float, int]]]): A history of trading account related transactions.
    """

    def __init__(self):
        """
        Initialize the TradingBot instance.
        """
        self.orders: Dict[int, Dict[str, Union[str, float, int]]]
        self.account_info: Dict[str, Union[int, float]]
        self.authenticated: bool
        self.market_status: str
        self.order_counter: int
        self.stocks: Dict[str, Dict[str, Union[float, int]]]
        self.watch_list: List[str]
        self.transaction_history: List[Dict[str, Union[str, float, int]]]
        self._api_description = "This tool belongs to the trading system, which allows users to trade stocks, manage their account, and view stock information."

    def _load_scenario(self, scenario: dict, long_context=False) -> None:
        """
        Load a scenario into the TradingBot.

        Args:
            scenario (dict): A scenario dictionary containing data to load.
        """
        DEFAULT_STATE_COPY = deepcopy(DEFAULT_STATE)
        self.orders = scenario.get("orders", DEFAULT_STATE_COPY["orders"])
        # Convert all string keys that can be interpreted as integers to integer keys
        self.orders = {
            int(k) if isinstance(k, str) and k.isdigit() else k: v
            for k, v in self.orders.items()
        }
        self.account_info = scenario.get("account_info", DEFAULT_STATE_COPY["account_info"])
        self.authenticated = scenario.get(
            "authenticated", DEFAULT_STATE_COPY["authenticated"]
        )
        self.market_status = scenario.get(
            "market_status", DEFAULT_STATE_COPY["market_status"]
        )
        self.order_counter = scenario.get(
            "order_counter", DEFAULT_STATE_COPY["order_counter"]
        )  # Start counter from the next order ID
        self.stocks = scenario.get("stocks", DEFAULT_STATE_COPY["stocks"])
        self.watch_list = scenario.get("watch_list", DEFAULT_STATE_COPY["watch_list"])
        self.transaction_history = scenario.get(
            "transaction_history", DEFAULT_STATE_COPY["transaction_history"]
        )
        self.long_context = long_context
        self._random = random.Random(
            (scenario.get("random_seed", DEFAULT_STATE_COPY["random_seed"]))
        )

    def _generate_transaction_timestamp(self) -> str:
        """
        Generate a timestamp for a transaction.

        Returns:
            timestamp (str): A formatted timestamp string.
        """
        # Define the start and end dates for the range
        start_date = CURRENT_TIME
        end_date = CURRENT_TIME + timedelta(days=1)

        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        # Generate a random timestamp within the range
        random_timestamp = self._random.randint(start_timestamp, end_timestamp)

        # Convert the random timestamp to a datetime object
        random_date = datetime.fromtimestamp(random_timestamp)

        return random_date.strftime("%Y-%m-%d %H:%M:%S")

    def get_current_time(self) -> Dict[str, str]:
        """
        Get the current time.

        Returns:
            current_time (str): Current time in HH:MM AM/PM format.
        """
        return {"current_time": CURRENT_TIME.strftime("%I:%M %p")}

    def get_symbol_by_name(self, name: str) -> Dict[str, str]:
        """
        Get the symbol of a stock by company name.

        Args:
            name (str): Name of the company.

        Returns:
            symbol (str): Symbol of the stock or "Stock not found" if not available.
        """
        symbol_map = {
            "Apple": "AAPL",
            "Google": "GOOG",
            "Tesla": "TSLA",
            "Microsoft": "MSFT",
            "Nvidia": "NVDA",
            "Zeta Corp": "ZETA",
            "Alpha Tech": "ALPH",
            "Omega Industries": "OMEG",
            "Quasar Ltd.": "QUAS",
            "Neptune Systems": "NEPT",
            "Synex Solutions": "SYNX",
            "Amazon": "AMZN",
            "Gorilla": "GORI",
        }

        return {"symbol": symbol_map.get(name, "Stock not found")}

    def get_stock_info(self, symbol: str) -> Dict[str, Union[float, int, str]]:
        """
        Get the details of a stock.

        Args:
            symbol (str): Symbol that uniquely identifies the stock.

        Returns:
            price (float): Current price of the stock.
            percent_change (float): Percentage change in stock price.
            volume (float): Trading volume of the stock.
            MA(5) (float): 5-day Moving Average of the stock.
            MA(20) (float): 20-day Moving Average of the stock.
        """
        if symbol not in self.stocks:
            return {"error": f"Stock with symbol '{symbol}' not found."}
        if self.long_context:
            stock = self.stocks[symbol].copy()
            stock["MA(5)"] = MA_5_EXTENSION
            stock["MA(20)"] = MA_20_EXTENSION
            return stock
        return self.stocks[symbol]

    def get_order_details(self, order_id: int) -> Dict[str, Union[str, float, int]]:
        """
        Get the details of an order.

        Args:
            order_id (int): ID of the order.

        Returns:
            id (int): ID of the order.
            order_type (str): Type of the order.
            symbol (str): Symbol of the stock in the order.
            price (float): Price at which the order was placed.
            amount (int): Number of shares in the order.
            status (str): Current status of the order. [Enum]: ["Open", "Pending", "Completed", "Cancelled"]
        """
        if order_id not in self.orders:
            return {
                "error": f"Order with ID {order_id} not found."
                + "Here is the list of orders_id: "
                + str(list(self.orders.keys()))
            }

        if self.long_context:
            order = self.orders[order_id].copy()
            symbol = order["symbol"]

            formatted_extension = {}
            for key, value in ORDER_DETAIL_EXTENSION.items():
                try:
                    formatted_extension[key] = value.format(symbol=symbol)
                except KeyError as e:
                    return {"error": f"KeyError during formatting: {str(e)}"}

            # Add formatted extension to the order metadata
            order["metadata"] = formatted_extension
            return order

        return self.orders[order_id]

    def cancel_order(self, order_id: int) -> Dict[str, Union[int, str]]:
        """
        Cancel an order.

        Args:
            order_id (int): ID of the order to cancel.

        Returns:
            order_id (int): ID of the cancelled order.
            status (str): New status of the order after cancellation attempt.
        """
        if order_id not in self.orders:
            return {"error": f"Order with ID {order_id} not found."}
        if self.orders[order_id]["status"] == "Completed":
            return {"error": f"Can't cancel order {order_id}. Order is already completed."}
        self.orders[order_id]["status"] = "Cancelled"
        return {"order_id": order_id, "status": "Cancelled"}

    def place_order(
        self, order_type: str, symbol: str, price: float, amount: int
    ) -> Dict[str, Union[int, str, float]]:
        """
        Place an order.

        Args:
            order_type (str): Type of the order (Buy/Sell).
            symbol (str): Symbol of the stock to trade.
            price (float): Price at which to place the order.
            amount (int): Number of shares to trade.

        Returns:
            order_id (int): ID of the newly placed order.
            order_type (str): Type of the order (Buy/Sell).
            status (str): Initial status of the order.
            price (float): Price at which the order was placed.
            amount (int): Number of shares in the order.
        """
        if not self.authenticated:
            return {"error": "User not authenticated. Please log in to place an order."}
        if symbol not in self.stocks:
            return {"error": f"Invalid stock symbol: {symbol}"}
        if price <= 0 or amount <= 0:
            return {"error": "Price and amount must be positive values."}

        # Ensure sufficient funds for buy orders
        if order_type.lower() == "buy":
            total_cost = float(price) * int(amount)
            if total_cost > self.account_info.get("balance", 0):
                return {
                    "error": (
                        "Insufficient funds: required "
                        f"${total_cost:.2f} but only ${self.account_info.get('balance', 0):.2f} available."
                    )
                }

        price = float(price)
        order_id = self.order_counter
        self.orders[order_id] = {
            "id": order_id,
            "order_type": order_type,
            "symbol": symbol,
            "price": price,
            "amount": amount,
            "status": "Open",
        }
        self.order_counter += 1
        # We return the status as "Pending" to indicate that the order has been placed but not yet executed
        # When polled later, the status will show as 'Open'
        # This is to simulate the delay between placing an order and it being executed
        return {
            "order_id": order_id,
            "order_type": order_type,
            "status": "Pending",
            "price": price,
            "amount": amount,
        }

    def withdraw_funds(self, amount: float) -> Dict[str, Union[str, float]]:
        """
        Withdraw funds from the account balance.

        Args:
            amount (float): Amount to withdraw from the account.

        Returns:
            status (str): Status of the transaction.
            new_balance (float): Updated account balance after the transaction.
        """
        if not self.authenticated:
            return {"error": "User not authenticated. Please log in to make a transaction."}
        if self.market_status != "Open":
            return {"error": "Market is closed. Transactions are not allowed."}
        if amount <= 0:
            return {"error": "Transaction amount must be positive."}

        if amount > self.account_info["balance"]:
            return {"error": "Insufficient funds for withdrawal."}

        self.account_info["balance"] -= amount
        self.transaction_history.append(
            {
                "type": "withdrawal",
                "amount": amount,
                "timestamp": self._generate_transaction_timestamp(),
            }
        )
        return {
            "status": "Withdrawal successful",
            "new_balance": self.account_info["balance"],
        }

    def get_account_info(self) -> Dict[str, Union[int, float]]:
        """
        Get account information.

        Returns:
            account_id (int): ID of the account.
            balance (float): Current balance of the account.
            binding_card (int): Card number associated with the account.
        """
        if not self.authenticated:
            return {
                "error": "User not authenticated. Please log in to view account information."
            }
        return self.account_info

    def trading_login(self, username: str, password: str) -> Dict[str, str]:
        """
        Handle user login.

        Args:
            username (str): Username for authentication.
            password (str): Password for authentication.

        Returns:
            status (str): Login status message.
        """
        if self.authenticated:
            return {"status": "Already logged in"}
        # In a real system, we would validate the username and password here
        self.authenticated = True
        return {"status": "Logged in successfully"}

    def trading_get_login_status(self) -> Dict[str, bool]:
        """
        Get the login status.

        Returns:
            status (bool): Login status.
        """

        return {"status": bool(self.authenticated)}

    def trading_logout(self) -> Dict[str, str]:
        """
        Handle user logout for trading system.

        Returns:
            status (str): Logout status message.
        """
        if not self.authenticated:
            return {"status": "No user is currently logged in"}
        self.authenticated = False
        return {"status": "Logged out successfully"}

    def fund_account(self, amount: float) -> Dict[str, Union[str, float]]:
        """
        Fund the account with the specified amount.

        Args:
            amount (float): Amount to fund the account with.

        Returns:
            status (str): Status of the funding operation.
            new_balance (float): Updated account balance after funding.
        """
        if not self.authenticated:
            return {"error": "User not authenticated. Please log in to fund the account."}
        if amount <= 0:
            return {"error": "Funding amount must be positive."}
        self.account_info["balance"] += amount
        self.transaction_history.append(
            {
                "type": "deposit",
                "amount": amount,
                "timestamp": self._generate_transaction_timestamp(),
            }
        )
        return {
            "status": "Account funded successfully",
            "new_balance": self.account_info["balance"],
        }

    def remove_stock_from_watchlist(self, symbol: str) -> Dict[str, str]:
        """
        Remove a stock from the watchlist.

        Args:
            symbol (str): Symbol of the stock to remove.

        Returns:
            status (str): Status of the removal operation.
        """
        if not self.authenticated:
            return {
                "error": "User not authenticated. Please log in to modify the watchlist."
            }
        if symbol not in self.watch_list:
            return {"error": f"Stock {symbol} not found in watchlist."}
        self.watch_list.remove(symbol)
        return {"status": f"Stock {symbol} removed from watchlist successfully."}

    def get_watchlist(self) -> Dict[str, List[str]]:
        """
        Get the watchlist.

        Returns:
            watchlist (List[str]): List of stock symbols in the watchlist.
        """
        if not self.authenticated:
            return ["Error: User not authenticated. Please log in to view the watchlist."]

        if self.long_context:
            watch_list = self.watch_list.copy()
            watch_list.extend(WATCH_LIST_EXTENSION)
            return watch_list
        return {"watchlist": self.watch_list}

    def get_order_history(self) -> Dict[str, List[Dict[str, Union[str, int, float]]]]:
        """
        Get the stock order ID history.

        Returns:
            order_history (List[int]): List of orders ID in the order history.
        """
        if not self.authenticated:
            return [
                {"error": "User not authenticated. Please log in to view order history."}
            ]

        return {"history": list(self.orders.keys())}

    def get_transaction_history(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """
        Get the transaction history within a specified date range.

        Args:
            start_date (str): [Optional] Start date for the history (format: 'YYYY-MM-DD').
            end_date (str): [Optional] End date for the history (format: 'YYYY-MM-DD').

        Returns:
            transaction_history (List[Dict]): List of transactions within the specified date range.
                - type (str): Type of transaction. [Enum]: ["deposit", "withdrawal"]
                - amount (float): Amount involved in the transaction.
                - timestamp (str): Timestamp of the transaction, formatted as 'YYYY-MM-DD HH:MM:SS'.
        """
        if not self.authenticated:
            return [
                {
                    "error": "User not authenticated. Please log in to view transaction history."
                }
            ]

        if start_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start = datetime.min

        if end_date:
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = datetime.max

        filtered_history = [
            transaction
            for transaction in self.transaction_history
            if start
            <= datetime.strptime(transaction["timestamp"], "%Y-%m-%d %H:%M:%S")
            <= end
        ]

        if self.long_context:
            filtered_history.extend(TRANSACTION_HISTORY_EXTENSION)

        return {"transaction_history": filtered_history}

    # below contains a list of functions to be nested
    def get_available_stocks(self, sector: str) -> Dict[str, List[str]]:
        """
        Get a list of stock symbols in the given sector.

        Args:
            sector (str): The sector to retrieve stocks from (e.g., 'Technology').

        Returns:
            stock_list (List[str]): List of stock symbols in the specified sector.
        """
        sector_map = {
            "Technology": ["AAPL", "GOOG", "MSFT", "NVDA"],
            "Automobile": ["TSLA", "F", "GM"],
        }

        if self.long_context:
            sector_map["Technology"].extend(TECHNOLOGY_EXTENSION)
            sector_map["Automobile"].extend(AUTOMOBILE_EXTENSION)
        return {"stock_list": sector_map.get(sector, [])}

    def filter_stocks_by_price(
        self, stocks: List[str], min_price: float, max_price: float
    ) -> Dict[str, List[str]]:
        """
        Filter stocks based on a price range.

        Args:
            stocks (List[str]): List of stock symbols to filter.
            min_price (float): Minimum stock price.
            max_price (float): Maximum stock price.

        Returns:
            filtered_stocks (List[str]): Filtered list of stock symbols within the price range.
        """
        filtered_stocks = [
            symbol
            for symbol in stocks
            if self.stocks.get(symbol, {}).get("price", 0) >= min_price
            and self.stocks.get(symbol, {}).get("price", 0) <= max_price
        ]
        return {"filtered_stocks": filtered_stocks}

    def add_to_watchlist(self, stock: str) -> Dict[str, List[str]]:
        """
        Add a stock to the watchlist.

        Args:
            stock (str): the stock symbol to add to the watchlist.

        Returns:
            watchlist (List[str]): the watchlist.
        """
        if stock not in self.watch_list:
            if stock in self.stocks:  # Ensure symbol is valid
                self.watch_list.append(stock)
        return {"watchlist": self.watch_list}

    def notify_price_change(self, stocks: List[str], threshold: float) -> Dict[str, str]:
        """
        Notify if there is a significant price change in the stocks.

        Args:
            stocks (List[str]): List of stock symbols to check.
            threshold (float): Percentage change threshold to trigger a notification.

        Returns:
            notification (str): Notification message about the price changes.
        """
        changed_stocks = [
            symbol
            for symbol in stocks
            if symbol in self.stocks
            and abs(self.stocks[symbol]["percent_change"]) >= threshold
        ]

        if changed_stocks:
            return {
                "notification": f"Stocks {', '.join(changed_stocks)} have significant price changes."
            }
        else:
            return {"notification": "No significant price changes in the selected stocks."}
