import random
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from .long_context import (
    BOOKING_RECORD_EXTENSION,
    CREDIT_CARD_EXTENSION,
)

DEFAULT_STATE = {
    "random_seed": 141053,
    "credit_card_list": {},
    "booking_record": {},
    "access_token": None,
    "token_type": None,
    "token_expires_in": None,
    "token_scope": None,
    "user_first_name": None,
    "user_last_name": None,
    "budget_limit": None,
}


class TravelAPI:
    # Adapted from source : https://developer.concur.com/api-reference/
    def __init__(self):
        super().__init__()
        self.credit_card_list: Dict[str, Dict[str, Union[str, int, float]]]
        self.booking_record: Dict[str, Dict[str, Union[str, float]]]
        self.access_token: Optional[str]
        self.token_type: Optional[str]
        self.token_expires_in: Optional[int]
        self.token_scope: Optional[str]
        self.user_first_name: Optional[str]
        self.user_last_name: Optional[str]
        self.budget_limit: Optional[float]
        self._api_description = "This tool belongs to the travel system, which allows users to book flights, manage credit cards, and view budget information."
        self._flight_cost_lookup: Dict[str, Dict[str, float]] = {}

    def _load_scenario(
        self,
        scenario: Dict[str, Union[Dict, str, int, float]],
        long_context: bool = False,
    ) -> None:
        """
        Load a scenario from the scenarios folder
        Args:
            scenario (Dict[str, str]): The scenario to load
        """
        DEFAULT_STATE_COPY = deepcopy(DEFAULT_STATE)
        self._random = random.Random(
            (scenario.get("random_seed", DEFAULT_STATE_COPY["random_seed"]))
        )
        self.credit_card_list = scenario.get(
            "credit_card_list", DEFAULT_STATE_COPY["credit_card_list"]
        )
        self.booking_record = scenario.get(
            "booking_record", DEFAULT_STATE_COPY["booking_record"]
        )
        self.access_token = scenario.get("access_token", DEFAULT_STATE_COPY["access_token"])
        self.token_type = scenario.get("token_type", DEFAULT_STATE_COPY["token_type"])
        self.token_expires_in = scenario.get(
            "token_expires_in", DEFAULT_STATE_COPY["token_expires_in"]
        )
        self.token_scope = scenario.get("token_scope", DEFAULT_STATE_COPY["token_scope"])
        self.user_first_name = scenario.get(
            "user_first_name", DEFAULT_STATE_COPY["user_first_name"]
        )
        self.user_last_name = scenario.get(
            "user_last_name", DEFAULT_STATE_COPY["user_last_name"]
        )
        self.budget_limit = scenario.get("budget_limit", DEFAULT_STATE_COPY["budget_limit"])
        self.long_context = long_context

        if self.long_context:
            self._add_credit_cards()  # Add credit card extension for long context
            self._add_booking_records()  # Add booking record extension

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TravelAPI):
            return False

        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            model_attr = getattr(self, attr_name)
            ground_truth_attr = getattr(value, attr_name)

            if model_attr != ground_truth_attr:
                return False

        return True

    def _add_credit_cards(self) -> None:
        """
        Merge the credit card list with predefined credit cards from long_context.py.
        Existing cards in the scenario won't be overwritten.
        """
        for card_id, card_info in CREDIT_CARD_EXTENSION.items():
            if card_id not in self.credit_card_list:
                self.credit_card_list[card_id] = card_info

    def _add_booking_records(self) -> None:
        """
        Merge the booking record list with predefined booking records from long_context.py.
        Existing bookings in the scenario won't be overwritten.
        """
        for booking_id, booking_info in BOOKING_RECORD_EXTENSION.items():
            if booking_id not in self.booking_record:
                self.booking_record[booking_id] = booking_info

    def _cache_flight_cost_entry(
        self, travel_from, travel_to, cost, travel_class, travel_date
    ):
        key = f"{travel_from}|{travel_to}|{travel_class}|{travel_date}"
        self._flight_cost_lookup[key] = {"cost": cost}

    def authenticate_travel(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        grant_type: str,
        user_first_name: str,
        user_last_name: str,
    ) -> Dict[str, Union[int, str]]:
        """
        Authenticate the user with the travel API

        Args:
            client_id (str): The client applications client_id supplied by App Management
            client_secret (str): The client applications client_secret supplied by App Management
            refresh_token (str): The refresh token obtained from the initial authentication
            grant_type (str): The grant type of the authentication request. Here are the options: read_write, read, write
            user_first_name (str): The first name of the user
            user_last_name (str): The last name of the user
        Returns:
            expires_in (int): The number of time it can use until the access token expires
            access_token (str): The access token to be used in the Authorization header of future requests
            token_type (str): The type of token
            scope (str): The scope of the token
        """
        self.token_expires_in = 2
        self.access_token = str(self._random.randint(100000, 999999))  # 6 digits
        self.token_type = "Bearer"
        self.token_scope = grant_type
        self.user_first_name = user_first_name
        self.user_last_name = user_last_name
        return {
            "expires_in": 2,
            "access_token": self.access_token,
            "token_type": "Bearer",
            "scope": grant_type,
        }

    def travel_get_login_status(self) -> Dict[str, bool]:
        """
        Get the status of the login

        Returns:
            status (bool): The status of the login
        """
        is_not_loggedin = self.token_expires_in is None or self.token_expires_in == 0
        return {"status": not is_not_loggedin}

    def get_budget_fiscal_year(
        self,
        lastModifiedAfter: Optional[str] = None,
        includeRemoved: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Get the budget fiscal year

        Args:
            lastModifiedAfter (str): [Optional] Use this field if you only want Fiscal Years that were changed after the supplied date. The supplied date will be interpreted in the UTC time zone. If lastModifiedAfter is not supplied, the service will return all Fiscal Years, regardless of modified date. Example: 2016-03-29T16:12:20. Return in the format of YYYY-MM-DDTHH:MM:SS.
            includeRemoved (str): [Optional] If true, the service will return all Fiscal Years, including those that were previously removed. If not supplied, this field defaults to false.
        Returns:
            budget_fiscal_year (str): The budget fiscal year
        """
        return {"budget_fiscal_year": "2018"}

    def register_credit_card(
        self,
        access_token: str,
        card_number: str,
        expiration_date: str,
        cardholder_name: str,
        card_verification_number: int,
    ) -> Dict[str, Union[str, Dict[str, str]]]:
        """
        Register a credit card

        Args:
            access_token (str): The access token obtained from the authenticate method
            card_number (str): The credit card number
            expiration_date (str): The expiration date of the credit card in the format MM/YYYY
            cardholder_name (str): The name of the cardholder
            card_verification_number (int): The card verification number
        Returns:
            card_id (str): The ID of the registered credit card
        """
        if self.token_expires_in is None:
            return {"error": "Token not initialized"}
        if self.token_expires_in == 0:
            return {"error": "Token expired"}
        if access_token != self.access_token:
            return {"error": "Invalid access token"}
        if card_number in self.credit_card_list:
            return {"error": "Card already registered"}
        card_id = str(self._random.randint(100000000000, 999999999999))  # 12 digits
        self.credit_card_list[card_id] = {
            "card_number": card_number,
            "expiration_date": expiration_date,
            "cardholder_name": cardholder_name,
            "card_verification_number": card_verification_number,
            "balance": self._random.randint(10000, 99999),  # 5 digits
        }
        return {"card_id": card_id}

    def _set_card_balance(self, card_id: str, balance: float) -> None:
        """
        Set the balance of a credit card

        Args:
            card_id (str): The ID of the credit card
            balance (float): The balance of the credit card
        """
        self.credit_card_list[card_id]["balance"] = balance

    def get_flight_cost(
        self, travel_from: str, travel_to: str, travel_date: str, travel_class: str
    ) -> Dict[str, List[float]]:
        """
        Get the list of cost of a flight in USD based on location, date, and class

        Args:
            travel_from (str): The 3 letter code of the departing airport
            travel_to (str): The 3 letter code of the arriving airport
            travel_date (str): The date of the travel in the format 'YYYY-MM-DD'
            travel_class (str): The class of the travel. Options are: economy, business, first.
        Returns:
            travel_cost_list (List[float]): The list of cost of the travel
        """
        base_costs: Dict[Tuple[str, str], int] = {
            ("SFO", "LAX"): 200,
            ("SFO", "JFK"): 500,
            ("SFO", "ORD"): 400,
            ("SFO", "BOS"): 450,
            ("SFO", "RMS"): 300,
            ("SFO", "SBK"): 350,
            ("SFO", "MPC"): 370,
            ("SFO", "SVP"): 320,
            ("SFO", "SHD"): 330,
            ("SFO", "SSV"): 340,
            ("SFO", "OKD"): 360,
            ("SFO", "WLB"): 310,
            ("SFO", "CRH"): 380,
            ("SFO", "ATV"): 390,
            ("SFO", "PHV"): 420,
            ("SFO", "GFD"): 430,
            ("SFO", "CIA"): 700,
            ("LAX", "SFO"): 100,
            ("LAX", "JFK"): 600,
            ("LAX", "ORD"): 500,
            ("LAX", "BOS"): 550,
            ("LAX", "RMS"): 310,
            ("LAX", "SBK"): 320,
            ("LAX", "MPC"): 330,
            ("LAX", "SVP"): 340,
            ("LAX", "SHD"): 350,
            ("LAX", "SSV"): 360,
            ("LAX", "OKD"): 370,
            ("LAX", "WLB"): 380,
            ("LAX", "CRH"): 390,
            ("LAX", "ATV"): 400,
            ("LAX", "PHV"): 410,
            ("LAX", "GFD"): 420,
            ("LAX", "HND"): 430,
            ("JFK", "ORD"): 300,
            ("JFK", "BOS"): 250,
            ("JFK", "RMS"): 450,
            ("JFK", "SBK"): 460,
            ("JFK", "MPC"): 470,
            ("JFK", "SVP"): 480,
            ("JFK", "SHD"): 490,
            ("JFK", "SSV"): 500,
            ("JFK", "OKD"): 510,
            ("JFK", "WLB"): 520,
            ("JFK", "CRH"): 530,
            ("JFK", "ATV"): 540,
            ("JFK", "PHV"): 550,
            ("JFK", "GFD"): 560,
            ("JFK", "LAX"): 570,
            ("JFK", "HND"): 800,
            ("JFK", "PVG"): 950,
            ("JFK", "PEK"): 1000,
            ("ORD", "LAX"): 180,
            ("ORD", "BOS"): 200,
            ("ORD", "RMS"): 350,
            ("ORD", "SBK"): 360,
            ("ORD", "MPC"): 370,
            ("ORD", "SVP"): 380,
            ("ORD", "SHD"): 390,
            ("ORD", "SSV"): 400,
            ("ORD", "OKD"): 410,
            ("ORD", "WLB"): 420,
            ("ORD", "CRH"): 430,
            ("ORD", "ATV"): 440,
            ("ORD", "PHV"): 450,
            ("ORD", "GFD"): 460,
            ("BOS", "RMS"): 400,
            ("BOS", "SBK"): 410,
            ("BOS", "MPC"): 420,
            ("BOS", "SVP"): 430,
            ("BOS", "SHD"): 440,
            ("BOS", "SSV"): 450,
            ("BOS", "OKD"): 460,
            ("BOS", "WLB"): 470,
            ("BOS", "CRH"): 480,
            ("BOS", "ATV"): 490,
            ("BOS", "PHV"): 500,
            ("BOS", "GFD"): 510,
            ("RMS", "BOS"): 200,
            ("RMS", "JFK"): 210,
            ("RMS", "SBK"): 220,
            ("RMS", "MPC"): 230,
            ("RMS", "SVP"): 240,
            ("RMS", "SHD"): 250,
            ("RMS", "SSV"): 260,
            ("RMS", "OKD"): 270,
            ("RMS", "WLB"): 280,
            ("RMS", "CRH"): 290,
            ("RMS", "ATV"): 300,
            ("RMS", "PHV"): 310,
            ("RMS", "GFD"): 320,
            ("RMS", "LAX"): 330,
            ("SBK", "MPC"): 200,
            ("SBK", "SVP"): 210,
            ("SBK", "SHD"): 220,
            ("SBK", "SSV"): 230,
            ("SBK", "OKD"): 240,
            ("SBK", "WLB"): 250,
            ("SBK", "CRH"): 260,
            ("SBK", "ATV"): 270,
            ("SBK", "PHV"): 280,
            ("SBK", "GFD"): 290,
            ("MPC", "SVP"): 210,
            ("MPC", "SHD"): 220,
            ("MPC", "SSV"): 230,
            ("MPC", "OKD"): 240,
            ("MPC", "WLB"): 250,
            ("MPC", "CRH"): 260,
            ("MPC", "ATV"): 270,
            ("MPC", "PHV"): 280,
            ("MPC", "GFD"): 290,
            ("SVP", "SHD"): 230,
            ("SVP", "SSV"): 240,
            ("SVP", "OKD"): 250,
            ("SVP", "WLB"): 260,
            ("SVP", "CRH"): 270,
            ("SVP", "ATV"): 280,
            ("SVP", "PHV"): 290,
            ("SVP", "GFD"): 300,
            ("SHD", "SSV"): 220,
            ("SHD", "OKD"): 230,
            ("SHD", "WLB"): 240,
            ("SHD", "CRH"): 250,
            ("SHD", "ATV"): 260,
            ("SHD", "PHV"): 270,
            ("SHD", "GFD"): 280,
            ("SSV", "OKD"): 240,
            ("SSV", "WLB"): 250,
            ("SSV", "CRH"): 260,
            ("SSV", "ATV"): 270,
            ("SSV", "PHV"): 280,
            ("SSV", "GFD"): 290,
            ("OKD", "WLB"): 230,
            ("OKD", "CRH"): 240,
            ("OKD", "ATV"): 250,
            ("OKD", "PHV"): 260,
            ("OKD", "GFD"): 270,
            ("WLB", "CRH"): 250,
            ("WLB", "ATV"): 260,
            ("WLB", "PHV"): 270,
            ("WLB", "GFD"): 280,
            ("CRH", "ATV"): 240,
            ("CRH", "PHV"): 250,
            ("CRH", "GFD"): 260,
            ("CRH", "SFO"): 270,
            ("CRH", "RMS"): 280,
            ("CRH", "HKG"): 290,
            ("CRH", "JFK"): 300,
            ("ATV", "PHV"): 230,
            ("ATV", "GFD"): 240,
            ("PHV", "GFD"): 220,
            ("LHR", "CDG"): 100,
            ("OKD", "LAX"): 220,
        }

        # Ensure the travel_from and travel_to is a tuple in the correct order (from, to)
        travel_pair = (travel_from, travel_to)

        # Get the base cost, raise an error if the route is not available
        if travel_pair in base_costs:
            base_cost = base_costs[travel_pair]
        else:
            raise ValueError("No available route for the given airports.")

        # Determine the multiplier based on the travel class
        if travel_class == "economy":
            factor = 1
        elif travel_class == "business":
            factor = 2
        elif travel_class == "first":
            factor = 5
        else:
            raise ValueError("Invalid travel class. Options are: economy, business, first.")

        # Determine the multiplier based on the travel date
        digit_sum = sum(int(char) for char in travel_date if char.isdigit())
        travel_date_multiplier = 2 if digit_sum % 2 == 0 else 1

        # Calculate the total cost
        travel_cost = float(base_cost * factor * travel_date_multiplier)

        travel_cost_list = []
        if self.long_context:
            self._flight_cost_lookup = {}  # reset cache
            for (frm, to), base in base_costs.items():
                cost = float(base * factor * travel_date_multiplier)
                self._cache_flight_cost_entry(frm, to, cost, travel_class, travel_date)
                travel_cost_list.append(cost)
        else:
            cost = float(base_costs[travel_pair] * factor * travel_date_multiplier)
            travel_cost_list = [cost]
            self._flight_cost_lookup = {
                f"{travel_from}|{travel_to}|{travel_class}|{travel_date}": {"cost": cost}
            }

        return {"travel_cost_list": travel_cost_list}

    def get_credit_card_balance(
        self, access_token: str, card_id: str
    ) -> Dict[str, Union[float, str]]:
        """
        Get the balance of a credit card
        Args:
            access_token (str): The access token obtained from the authenticate
            card_id (str): The ID of the credit card
        Returns:
            card_balance (float): The balance of the credit card
        """
        if self.token_expires_in == 0:
            return {"error": "Token expired"}
        if access_token != self.access_token:
            return {"error": "Invalid access token"}
        if card_id not in self.credit_card_list:
            return {
                "error": "Card not registered. Here are a list of card_id's: "
                + str(list(self.credit_card_list.keys()))
            }
        return {"card_balance": self.credit_card_list[card_id]["balance"]}

    def book_flight(
        self,
        access_token: str,
        card_id: str,
        travel_date: str,
        travel_from: str,
        travel_to: str,
        travel_class: str,
    ) -> Dict[str, Union[str, bool, Dict]]:
        """
        Book a flight given the travel information. From and To should be the airport codes in the IATA format.

        Args:
            access_token (str): The access token obtained from the authenticate
            card_id (str): The ID of the credit card to use for the booking
            travel_date (str): The date of the travel in the format YYYY-MM-DD
            travel_from (str): The location the travel is from
            travel_to (str): The location the travel is to
            travel_class (str): The class of the travel
        Returns:
            booking_id (str): The ID of the booking
            transaction_id (str): The ID of the transaction
            booking_status (bool): The status of the booking, True if successful, False if failed
            booking_history (Dict): The booking history. This field might be empty even though the history is non-empty.
                - booking_id (str): The ID of the booking
                - transaction_id (str): The ID of the transaction
                - travel_date (str): The date of the travel
                - travel_from (str): The location the travel is from
                - travel_to (str): The location the travel is to
                - travel_class (str): The class of the travel
                - travel_cost (float): The cost of the travel
        """
        if self.token_expires_in == 0:
            return {"booking_status": False, "error": "Token expired"}
        if access_token != self.access_token:
            return {"booking_status": False, "error": "Invalid access token"}
        if card_id not in self.credit_card_list:
            return {"booking_status": False, "error": "Card not registered"}
        if "balance" not in self.credit_card_list[card_id]:
            return {"booking_status": False, "error": "Balance not found"}

        all_airports = self.list_all_airports()
        if travel_from not in all_airports:
            return {
                "booking_status": False,
                "error": f"Invalid departure airport code: {travel_from}",
            }
        if travel_to not in all_airports:
            return {
                "booking_status": False,
                "error": f"Invalid destination airport code: {travel_to}",
            }

        try:
            datetime.strptime(travel_date, "%Y-%m-%d")
        except ValueError:
            return {
                "booking_status": False,
                "error": "Invalid date format. Use YYYY-MM-DD.",
            }

        valid_classes = {"economy", "business", "first"}
        if travel_class not in valid_classes:
            return {
                "booking_status": False,
                "error": f"Invalid travel class. Must be one of {valid_classes}",
            }

        try:
            self.get_flight_cost(
                travel_from=travel_from,
                travel_to=travel_to,
                travel_date=travel_date,
                travel_class=travel_class,
            )
            key = f"{travel_from}|{travel_to}|{travel_class}|{travel_date}"
            travel_cost_entry = self._flight_cost_lookup.get(key)
            if travel_cost_entry is None:
                return {
                    "booking_status": False,
                    "error": "No available route for the given parameters",
                }
            travel_cost = travel_cost_entry["cost"]
        except ValueError as e:
            return {"booking_status": False, "error": str(e)}

        if self.credit_card_list[card_id]["balance"] < travel_cost:
            return {"booking_status": False, "error": "Insufficient funds"}
        if (
            self.budget_limit is not None
            and self.credit_card_list[card_id]["balance"] < self.budget_limit
        ):
            return {
                "booking_status": False,
                "error": "Balance is less than budget limit",
            }

        self.credit_card_list[card_id]["balance"] -= travel_cost
        booking_id = str(self._random.randint(1000000, 9999999))  # 7 digits
        transaction_id = str(self._random.randint(10000000, 99999999))  # 8 digits
        self.booking_record[booking_id] = {
            "card_id": card_id,
            "travel_date": travel_date,
            "travel_from": travel_from,
            "travel_to": travel_to,
            "travel_class": travel_class,
            "travel_cost": travel_cost,
            "transaction_id": transaction_id,
        }
        if self.long_context:
            return {
                "booking_id": booking_id,
                "transaction_id": transaction_id,
                "booking_status": True,
                "booking_history": self.booking_record,
            }
        return {
            "booking_id": booking_id,
            "transaction_id": transaction_id,
            "booking_status": True,
            "booking_history": {},
        }

    def retrieve_invoice(
        self,
        access_token: str,
        booking_id: Optional[str] = None,
        insurance_id: Optional[str] = None,
    ) -> Dict[str, Union[Dict[str, Union[str, float]], str]]:
        """
        Retrieve the invoice for a booking.

        Args:
            access_token (str): The access token obtained from the authenticate
            booking_id (str): [Optional] The ID of the booking
            insurance_id (str): [Optional] The ID of the insurance
        Returns:
            invoice (Dict): The invoice for the booking
                - booking_id (str): The ID of the booking
                - travel_date (str): The date of the travel
                - travel_from (str): The location the travel is from
                - travel_to (str): The location the travel is to
                - travel_class (str): The class of the travel
                - travel_cost (float): The cost of the travel
                - transaction_id (str): The ID of the transaction
        """
        if self.token_expires_in == 0:
            return {"error": "Token expired"}
        if access_token != self.access_token:
            return {"error": "Invalid access token"}
        if booking_id not in self.booking_record:
            return {"error": "Booking not found"}
        invoice = {
            "booking_id": booking_id,
            "travel_date": self.booking_record[booking_id]["travel_date"],
            "travel_from": self.booking_record[booking_id]["travel_from"],
            "travel_to": self.booking_record[booking_id]["travel_to"],
            "travel_class": self.booking_record[booking_id]["travel_class"],
            "travel_cost": self.booking_record[booking_id]["travel_cost"],
            "transaction_id": self.booking_record[booking_id]["transaction_id"],
        }
        return {"invoice": invoice}

    def get_booking_history(
        self,
        access_token: str,
    ) -> Dict[str, Dict[str, Dict[str, Union[str, float]]]]:
        """
        Retrieve all booking history for the user.

        Args:
            access_token (str): The access token obtained from the authenticate method.

        Returns:
            booking_history (Dict): A dictionary keyed by booking_id where each value contains the booking details.
                - transaction_id (str): The ID of the transaction
                - travel_date (str): The date of the travel
                - travel_from (str): The location the travel is from
                - travel_to (str): The location the travel is to
                - travel_class (str): The class of the travel
                - travel_cost (float): The cost of the travel
        """
        # Validate token state similar to other secured endpoints
        if self.token_expires_in == 0:
            return {"error": "Token expired"}
        if access_token != self.access_token:
            return {"error": "Invalid access token"}

        # Simply return a copy of the booking records to avoid accidental mutation
        return {"booking_history": deepcopy(self.booking_record)}

    def list_all_airports(self) -> List[str]:
        """
        List all available airports

        Returns:
            airports (List[str]): A list of all available airports
        """
        return [
            "RMS",
            "SBK",
            "MPC",
            "SVP",
            "SHD",
            "CDG",
            "LHR",
            "SSV",
            "OKD",
            "WLB",
            "PEK",
            "HND",
            "HKG",
            "CIA",
            "CRH",
            "ATV",
            "PHV",
            "GFD",
            "SFO",
            "LAX",
            "JFK",
            "ORD",
            "BOS",
        ]

    def cancel_booking(
        self, access_token: str, booking_id: str
    ) -> Dict[str, Union[bool, str]]:
        """
        Cancel a booking

        Args:
            access_token (str): The access token obtained from the authenticate
            booking_id (str): The ID of the booking
        Returns:
            cancel_status (bool): The status of the cancellation, True if successful, False if failed
        """
        if self.token_expires_in == 0:
            return {"cancel_status": False, "error": "Token expired"}
        if access_token != self.access_token:
            return {"cancel_status": False, "error": "Invalid access token"}
        if booking_id not in self.booking_record:
            return {"cancel_status": False, "error": "Booking not found"}
        card_id = self.booking_record[booking_id]["card_id"]
        travel_cost = self.booking_record[booking_id]["travel_cost"]
        self.credit_card_list[card_id]["balance"] += travel_cost
        del self.booking_record[booking_id]
        return {"cancel_status": True}

    def compute_exchange_rate(
        self, base_currency: str, target_currency: str, value: float
    ) -> float:
        """
        Compute the exchange rate between two currencies

        Args:
            base_currency (str): The base currency. [Enum]: USD, RMB, EUR, JPY, GBP, CAD, AUD, INR, RUB, BRL, MXN
            target_currency (str): The target currency. [Enum]: USD, RMB, EUR, JPY, GBP, CAD, AUD, INR, RUB, BRL, MXN
            value (float): The value to convert
        Returns:
            exchanged_value (float): The value after the exchange

        """
        exchange_rates = {
            ("USD", "RMB"): 7,
            ("USD", "EUR"): 0.8,
            ("USD", "JPY"): 110,
            ("USD", "GBP"): 0.7,
            ("USD", "CAD"): 1.3,
            ("USD", "AUD"): 1.4,
            ("USD", "INR"): 70,
            ("USD", "RUB"): 60,
            ("USD", "BRL"): 3.8,
            ("USD", "MXN"): 20,
        }
        for key, val in exchange_rates.items():
            if base_currency == key[0] and target_currency == key[1]:
                return {"exchanged_value": float(value * val)}
            elif base_currency == key[1] and target_currency == key[0]:
                return {"exchanged_value": round(value / val, 2)}
        raise ValueError("No available exchange rate for the given currencies.")

    def verify_traveler_information(
        self, first_name: str, last_name: str, date_of_birth: str, passport_number: str
    ) -> Dict[str, Union[bool, str]]:
        """
        Verify the traveler information

        Args:
            first_name (str): The first name of the traveler
            last_name (str): The last name of the traveler
            date_of_birth (str): The date of birth of the traveler in the format YYYY-MM-DD
            passport_number (str): The passport number of the traveler
        Returns:
            verification_status (bool): The status of the verification, True if successful, False if failed
            verification_failure (str): The reason for the verification failure
        """
        if self.user_first_name != first_name or self.user_last_name != last_name:
            return {
                "verification_status": False,
                "verification_failure": "Cannot book flight information for another user."
                + f"Expected {self.user_first_name} {self.user_last_name}, got {first_name} {last_name}",
            }

        # Calculate age
        try:
            birth_date = datetime.strptime(date_of_birth, "%Y-%m-%d")
            today = datetime.today()
            age = (
                today.year
                - birth_date.year
                - ((today.month, today.day) < (birth_date.month, birth_date.day))
            )
        except ValueError:
            return {
                "verification_status": False,
                "verification_failure": "Invalid date of birth format. Please use YYYY-MM-DD.",
            }

        # Check if the traveler is at least 18 years old
        if age < 18:
            return {
                "verification_status": False,
                "verification_failure": "Traveler must be at least 18 years old.",
            }

        # Check if the passport number starts with 'US' (assuming this indicates a US passport)
        if not passport_number.startswith("US"):
            return {
                "verification_status": False,
                "verification_failure": "Passport must be issued by the United States.",
            }

        # If all checks pass
        return {"verification_status": True}

    def set_budget_limit(
        self, access_token: str, budget_limit: float
    ) -> Dict[str, Union[float, str]]:
        """
        Set the budget limit for the user

        Args:
            access_token (str): The access token obtained from the authentication process or initial configuration.
            budget_limit (float): The budget limit to set in USD
        Returns:
            budget_limit (float): The budget limit set in USD
        """
        if self.token_expires_in == 0:
            return {"error": "Token expired"}
        if access_token != self.access_token:
            return {"error": "Invalid access token"}
        budget_limit = float(budget_limit)
        self.budget_limit = budget_limit
        return {"budget_limit": budget_limit}

    def get_nearest_airport_by_city(self, location: str) -> Dict[str, str]:
        """
        Get the nearest airport to the given location

        Args:
            location (str): The name of the location. [Enum]: Rivermist, Stonebrook, Maplecrest, Silverpine, Shadowridge, London, Paris, Sunset Valley, Oakendale, Willowbend, Crescent Hollow, Autumnville, Pinehaven, Greenfield, San Francisco, Los Angeles, New York, Chicago, Boston, Beijing, Hong Kong, Rome, Tokyo
        Returns:
            nearest_airport (str): The nearest airport to the given location
        """
        airport_map = {
            "Rivermist": "RMS",
            "Stonebrook": "SBK",
            "Maplecrest": "MPC",
            "Silverpine": "SVP",
            "Shadowridge": "SHD",
            "London": "LHR",
            "Paris": "CDG",
            "Sunset Valley": "SSV",
            "Oakendale": "OKD",
            "Willowbend": "WLB",
            "Crescent Hollow": "CRH",
            "Autumnville": "ATV",
            "Pinehaven": "PHV",
            "Greenfield": "GFD",
            "San Francisco": "SFO",
            "Los Angeles": "LAX",
            "New York": "JFK",
            "Chicago": "ORD",
            "Boston": "BOS",
            "Beijing": "PEK",
            "Hong Kong": "HKG",
            "Rome": "CIA",
            "Tokyo": "HND",
        }

        return {"nearest_airport": airport_map.get(location, "Unknown")}

    def purchase_insurance(
        self,
        access_token: str,
        insurance_type: str,
        booking_id: str,
        insurance_cost: float,
        card_id: str,
    ) -> Dict[str, Union[str, bool]]:
        """
        Purchase insurance

        Args:
            access_token (str): The access token obtained from the authenticate
            insurance_type (str): The type of insurance to purchase
            insurance_cost (float): The cost of the insurance
            booking_id (str): The ID of the booking
            card_id (str): The ID of the credit card to use for the
        Returns:
            insurance_id (str): The ID of the insurance
            insurance_status (bool): The status of the insurance purchase, True if successful, False if failed
        """
        if self.token_expires_in == 0:
            return {"insurance_status": False, "error": "Token expired"}
        if access_token != self.access_token:
            return {"insurance_status": False, "error": "Invalid access token"}
        if self.budget_limit is not None and self.budget_limit < insurance_cost:
            return {"insurance_status": False, "error": "Exceeded budget limit"}
        if booking_id not in self.booking_record:
            return {"insurance_status": False, "error": "Booking not found"}
        if card_id not in self.credit_card_list:
            return {"insurance_status": False, "error": "Credit card not registered"}
        self.credit_card_list[card_id]["balance"] -= insurance_cost
        return {
            "insurance_id": str(self._random.randint(100000000, 999999999)),  # 9 digits
            "insurance_status": True,
        }

    def contact_customer_support(self, booking_id: str, message: str) -> Dict[str, str]:
        """
        Contact travel booking customer support, get immediate support on an issue with an online call.

        Args:
            booking_id (str): The ID of the booking
            message (str): The message to send to customer support
        Returns:
            customer_support_message (str): The message from customer support
        """
        if booking_id not in self.booking_record:
            return {"error": "Booking not found"}

        return {
            "customer_support_message": "Thank you for contacting customer support. Your message has been received and we will get back to you shortly."
        }

    def get_all_credit_cards(self) -> Dict[str, Dict[str, Union[str, int, float]]]:
        """
        Get all registered credit cards

        Returns:
            credit_card_list (Dict): A dictionary containing all registered credit cards
                - card_number (str): The number of the credit card
                - expiration_date (str): The expiration date of the credit card in the format YYYY-MM-DD
                - cardholder_name (str): The name of the cardholder
                - card_verification_value (int): The verification value of the credit card
                - balance (float): The balance of the credit card
        """
        return {"credit_card_list": self.credit_card_list}
