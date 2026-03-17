"""Airline domain models and LangChain tool wrappers for tau2 benchmark evaluation.

Reimplements the essential tau2 airline domain (data models, database, tools) as
self-contained code so the evals run without a cross-repo dependency.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FlightType = Literal["round_trip", "one_way"]
CabinClass = Literal["business", "economy", "basic_economy"]
Insurance = Literal["yes", "no"]
MembershipLevel = Literal["gold", "silver", "regular"]

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class AirportCode(BaseModel):
    iata: str
    city: str


class Name(BaseModel):
    first_name: str
    last_name: str


class Address(BaseModel):
    address1: str
    address2: str | None = None
    city: str
    country: str
    state: str
    zip: str


class Payment(BaseModel):
    payment_id: str
    amount: int


class CreditCard(BaseModel):
    source: Literal["credit_card"]
    id: str
    brand: str
    last_four: str


class GiftCard(BaseModel):
    source: Literal["gift_card"]
    id: str
    amount: float


class Certificate(BaseModel):
    source: Literal["certificate"]
    id: str
    amount: float


PaymentMethod = CreditCard | GiftCard | Certificate


class Passenger(BaseModel):
    first_name: str
    last_name: str
    dob: str


class FlightDateStatusAvailable(BaseModel):
    status: Literal["available"]
    available_seats: dict[str, int]
    prices: dict[str, int]


class FlightDateStatusDelayed(BaseModel):
    status: Literal["delayed"]
    estimated_departure_time_est: str
    estimated_arrival_time_est: str


class FlightDateStatusOnTime(BaseModel):
    status: Literal["on time"]
    estimated_departure_time_est: str
    estimated_arrival_time_est: str


class FlightDateStatusFlying(BaseModel):
    status: Literal["flying"]
    actual_departure_time_est: str
    estimated_arrival_time_est: str


class FlightDateStatusLanded(BaseModel):
    status: Literal["landed"]
    actual_departure_time_est: str
    actual_arrival_time_est: str


class FlightDateStatusCancelled(BaseModel):
    status: Literal["cancelled"]


FlightDateStatus = (
    FlightDateStatusAvailable
    | FlightDateStatusDelayed
    | FlightDateStatusOnTime
    | FlightDateStatusFlying
    | FlightDateStatusLanded
    | FlightDateStatusCancelled
)


class Flight(BaseModel):
    flight_number: str
    origin: str
    destination: str
    scheduled_departure_time_est: str
    scheduled_arrival_time_est: str
    dates: dict[str, FlightDateStatus]


class DirectFlight(BaseModel):
    flight_number: str
    origin: str
    destination: str
    status: Literal["available"] = "available"
    scheduled_departure_time_est: str
    scheduled_arrival_time_est: str
    date: str | None = None
    available_seats: dict[str, int]
    prices: dict[str, int]


class ReservationFlight(BaseModel):
    flight_number: str
    origin: str
    destination: str
    date: str
    price: int


class FlightInfo(BaseModel):
    flight_number: str = Field(description="Flight number, such as 'HAT001'.")
    date: str = Field(description="The date for the flight in the format 'YYYY-MM-DD', such as '2024-05-01'.")


class User(BaseModel):
    user_id: str
    name: Name
    address: Address
    email: str
    dob: str
    payment_methods: dict[str, PaymentMethod]
    saved_passengers: list[Passenger]
    membership: MembershipLevel
    reservations: list[str]


class Reservation(BaseModel):
    reservation_id: str
    user_id: str
    origin: str
    destination: str
    flight_type: FlightType
    cabin: CabinClass
    flights: list[ReservationFlight]
    passengers: list[Passenger]
    payment_history: list[Payment]
    created_at: str
    total_baggages: int
    nonfree_baggages: int
    insurance: Insurance
    status: Literal["cancelled"] | None = None


class FlightDB(BaseModel):
    flights: dict[str, Flight]
    users: dict[str, User]
    reservations: dict[str, Reservation]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"


def load_db() -> FlightDB:
    """Load a fresh FlightDB from the bundled db.json."""
    with (_DATA_DIR / "db.json").open() as fp:
        return FlightDB.model_validate_json(fp.read())


def load_policy() -> str:
    """Load the airline customer service policy."""
    with (_DATA_DIR / "policy.md").open() as fp:
        return fp.read()


def load_task(task_id: str) -> dict[str, Any]:
    """Load a single task by ID from tasks.json.

    Args:
        task_id: The string task ID (e.g. "2", "14").

    Returns:
        The raw task dict.

    Raises:
        KeyError: If the task ID is not found.
    """
    with (_DATA_DIR / "tasks.json").open() as fp:
        tasks = json.load(fp)
    for task in tasks:
        if str(task.get("id")) == str(task_id):
            return task
    msg = f"Task {task_id} not found in tasks.json"
    raise KeyError(msg)


# ---------------------------------------------------------------------------
# Tool call logging
# ---------------------------------------------------------------------------


@dataclass
class ToolCallEntry:
    """Record of a single tool invocation."""

    name: str
    args: dict[str, Any]
    result: str
    error: bool = False


# ---------------------------------------------------------------------------
# Internal helpers (shared by tool implementations)
# ---------------------------------------------------------------------------

_CURRENT_DATETIME = "2024-05-15T15:00:00"
_NEW_RESERVATION_IDS = ("HATHAT", "HATHAU", "HATHAV")
_NEW_PAYMENT_IDS = (3221322, 3221323, 3221324)


def _get_user(db: FlightDB, user_id: str) -> User:
    if user_id not in db.users:
        msg = f"User {user_id} not found"
        raise ToolException(msg)
    return db.users[user_id]


def _get_reservation(db: FlightDB, reservation_id: str) -> Reservation:
    if reservation_id not in db.reservations:
        msg = f"Reservation {reservation_id} not found"
        raise ToolException(msg)
    return db.reservations[reservation_id]


def _get_flight(db: FlightDB, flight_number: str) -> Flight:
    if flight_number not in db.flights:
        msg = f"Flight {flight_number} not found"
        raise ToolException(msg)
    return db.flights[flight_number]


def _get_flight_instance(db: FlightDB, flight_number: str, date: str) -> FlightDateStatus:
    flight = _get_flight(db, flight_number)
    if date not in flight.dates:
        msg = f"Flight {flight_number} not found on date {date}"
        raise ToolException(msg)
    return flight.dates[date]


def _search_direct_flights(
    db: FlightDB,
    date: str,
    origin: str | None = None,
    destination: str | None = None,
    leave_after: str | None = None,
) -> list[DirectFlight]:
    results: list[DirectFlight] = []
    for flight in db.flights.values():
        if origin is not None and flight.origin != origin:
            continue
        if destination is not None and flight.destination != destination:
            continue
        if date not in flight.dates:
            continue
        if flight.dates[date].status != "available":
            continue
        if leave_after is not None and flight.scheduled_departure_time_est < leave_after:
            continue
        date_data = flight.dates[date]
        if not isinstance(date_data, FlightDateStatusAvailable):
            continue
        results.append(
            DirectFlight(
                flight_number=flight.flight_number,
                origin=flight.origin,
                destination=flight.destination,
                scheduled_departure_time_est=flight.scheduled_departure_time_est,
                scheduled_arrival_time_est=flight.scheduled_arrival_time_est,
                available_seats=date_data.available_seats,
                prices=date_data.prices,
            )
        )
    return results


def _get_new_reservation_id(db: FlightDB) -> str:
    for rid in _NEW_RESERVATION_IDS:
        if rid not in db.reservations:
            return rid
    msg = "Too many reservations"
    raise ToolException(msg)


def _payment_for_update(user: User, payment_id: str, total_price: int) -> Payment | None:
    if payment_id not in user.payment_methods:
        msg = "Payment method not found"
        raise ToolException(msg)
    pm = user.payment_methods[payment_id]
    if pm.source == "certificate":
        msg = "Certificate cannot be used to update reservation"
        raise ToolException(msg)
    if pm.source == "gift_card" and pm.amount < total_price:
        msg = "Gift card balance is not enough"
        raise ToolException(msg)
    if pm.source == "gift_card":
        pm.amount -= total_price
    if total_price != 0:
        return Payment(payment_id=payment_id, amount=total_price)
    return None


def _serialize(obj: BaseModel | list[BaseModel]) -> str:
    """Serialize a Pydantic model or list to JSON string."""
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=2)

    def _default(o: object) -> dict[str, object]:
        if isinstance(o, BaseModel):
            return o.model_dump()
        msg = f"Object of type {type(o)} is not JSON serializable"
        raise TypeError(msg)

    return json.dumps(obj, indent=2, default=_default)


# ---------------------------------------------------------------------------
# Pydantic schemas for complex tool inputs
# ---------------------------------------------------------------------------


class BookReservationInput(BaseModel):
    """Input schema for the book_reservation tool."""

    user_id: str = Field(description="The ID of the user, such as 'sara_doe_496'.")
    origin: str = Field(description="The IATA code for the origin city, such as 'SFO'.")
    destination: str = Field(description="The IATA code for the destination city, such as 'JFK'.")
    flight_type: FlightType = Field(description="The type of flight: 'one_way' or 'round_trip'.")
    cabin: CabinClass = Field(description="The cabin class: 'basic_economy', 'economy', or 'business'.")
    flights: list[FlightInfo] = Field(description="List of flight segments with flight_number and date.")
    passengers: list[Passenger] = Field(description="List of passengers with first_name, last_name, and dob.")
    payment_methods: list[Payment] = Field(description="List of payment methods with payment_id and amount.")
    total_baggages: int = Field(description="Total number of checked bags.")
    nonfree_baggages: int = Field(description="Number of paid (non-free) checked bags.")
    insurance: Insurance = Field(description="Whether to purchase travel insurance: 'yes' or 'no'.")


class UpdateReservationFlightsInput(BaseModel):
    """Input schema for the update_reservation_flights tool."""

    reservation_id: str = Field(description="The reservation ID, such as 'ZFA04Y'.")
    cabin: CabinClass = Field(description="The cabin class for the reservation.")
    flights: list[FlightInfo] = Field(description="All flight segments in the updated reservation (include unchanged segments too).")
    payment_id: str = Field(description="Payment method ID from user profile, such as 'credit_card_7815826'.")


class UpdateReservationPassengersInput(BaseModel):
    """Input schema for the update_reservation_passengers tool."""

    reservation_id: str = Field(description="The reservation ID, such as 'ZFA04Y'.")
    passengers: list[Passenger] = Field(description="Updated list of passengers.")


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def create_airline_tools(  # noqa: C901, PLR0915
    db: FlightDB,
) -> tuple[list[StructuredTool], list[ToolCallEntry]]:
    """Create LangChain tools backed by the given FlightDB instance.

    Each tool mutates the shared `db` and logs its invocation to the returned log.

    Args:
        db: The mutable FlightDB instance.

    Returns:
        A tuple of (tools list, shared tool call log).
    """
    log: list[ToolCallEntry] = []

    def _log_and_return(name: str, args: dict[str, Any], result: str) -> str:
        log.append(ToolCallEntry(name=name, args=args, result=result))
        return result

    def _log_error(name: str, args: dict[str, Any], error: str) -> str:
        log.append(ToolCallEntry(name=name, args=args, result=error, error=True))
        return error

    # --- get_user_details ---
    def get_user_details(user_id: str) -> str:
        try:
            user = _get_user(db, user_id)
        except ToolException as exc:
            return _log_error("get_user_details", {"user_id": user_id}, str(exc))
        return _log_and_return("get_user_details", {"user_id": user_id}, _serialize(user))

    # --- get_reservation_details ---
    def get_reservation_details(reservation_id: str) -> str:
        try:
            res = _get_reservation(db, reservation_id)
        except ToolException as exc:
            return _log_error("get_reservation_details", {"reservation_id": reservation_id}, str(exc))
        return _log_and_return("get_reservation_details", {"reservation_id": reservation_id}, _serialize(res))

    # --- search_direct_flight ---
    def search_direct_flight(origin: str, destination: str, date: str) -> str:
        args = {"origin": origin, "destination": destination, "date": date}
        results = _search_direct_flights(db, date=date, origin=origin, destination=destination)
        return _log_and_return("search_direct_flight", args, _serialize(results))

    # --- search_onestop_flight ---
    def search_onestop_flight(origin: str, destination: str, date: str) -> str:
        args = {"origin": origin, "destination": destination, "date": date}
        results: list[list[DirectFlight]] = []
        for r1 in _search_direct_flights(db, date=date, origin=origin):
            r1.date = date
            date2 = f"2024-05-{int(date[-2:]) + 1}" if "+1" in r1.scheduled_arrival_time_est else date
            for r2 in _search_direct_flights(
                db,
                date=date2,
                origin=r1.destination,
                destination=destination,
                leave_after=r1.scheduled_arrival_time_est,
            ):
                r2.date = date2
                results.append([r1, r2])
        return _log_and_return("search_onestop_flight", args, _serialize(results))

    # --- cancel_reservation ---
    def cancel_reservation(reservation_id: str) -> str:
        try:
            reservation = _get_reservation(db, reservation_id)
        except ToolException as exc:
            return _log_error("cancel_reservation", {"reservation_id": reservation_id}, str(exc))
        refunds = [Payment(payment_id=p.payment_id, amount=-p.amount) for p in reservation.payment_history]
        reservation.payment_history.extend(refunds)
        reservation.status = "cancelled"
        return _log_and_return("cancel_reservation", {"reservation_id": reservation_id}, _serialize(reservation))

    # --- book_reservation ---
    def book_reservation(  # noqa: C901, PLR0912
        user_id: str,
        origin: str,
        destination: str,
        flight_type: FlightType,
        cabin: CabinClass,
        flights: list[dict[str, str] | FlightInfo],
        passengers: list[dict[str, str] | Passenger],
        payment_methods: list[dict[str, Any] | Payment],
        total_baggages: int,
        nonfree_baggages: int,
        insurance: Insurance,
    ) -> str:
        raw_args: dict[str, Any] = {
            "user_id": user_id,
            "origin": origin,
            "destination": destination,
            "flight_type": flight_type,
            "cabin": cabin,
            "flights": [f.model_dump() if isinstance(f, BaseModel) else f for f in flights],
            "passengers": [p.model_dump() if isinstance(p, BaseModel) else p for p in passengers],
            "payment_methods": [p.model_dump() if isinstance(p, BaseModel) else p for p in payment_methods],
            "total_baggages": total_baggages,
            "nonfree_baggages": nonfree_baggages,
            "insurance": insurance,
        }
        try:
            parsed_flights = [FlightInfo(**f) if isinstance(f, dict) else f for f in flights]
            parsed_passengers = [Passenger(**p) if isinstance(p, dict) else p for p in passengers]
            parsed_payments = [Payment(**p) if isinstance(p, dict) else p for p in payment_methods]
            user = _get_user(db, user_id)
            rid = _get_new_reservation_id(db)

            reservation = Reservation(
                reservation_id=rid,
                user_id=user_id,
                origin=origin,
                destination=destination,
                flight_type=flight_type,
                cabin=cabin,
                flights=[],
                passengers=deepcopy(parsed_passengers),
                payment_history=deepcopy(parsed_payments),
                created_at=_CURRENT_DATETIME,
                total_baggages=total_baggages,
                nonfree_baggages=nonfree_baggages,
                insurance=insurance,
            )

            total_price = 0
            all_flight_dates: list[FlightDateStatusAvailable] = []

            for fi in parsed_flights:
                flight = _get_flight(db, fi.flight_number)
                fds = _get_flight_instance(db, fi.flight_number, fi.date)
                if not isinstance(fds, FlightDateStatusAvailable):
                    msg = f"Flight {fi.flight_number} not available on date {fi.date}"
                    raise ToolException(msg)  # noqa: TRY301
                if fds.available_seats.get(cabin, 0) < len(parsed_passengers):
                    msg = f"Not enough seats on flight {fi.flight_number}"
                    raise ToolException(msg)  # noqa: TRY301
                price = fds.prices[cabin]
                reservation.flights.append(
                    ReservationFlight(
                        origin=flight.origin,
                        destination=flight.destination,
                        flight_number=fi.flight_number,
                        date=fi.date,
                        price=price,
                    )
                )
                all_flight_dates.append(fds)
                total_price += price * len(parsed_passengers)

            if insurance == "yes":
                total_price += 30 * len(parsed_passengers)
            total_price += 50 * nonfree_baggages

            for pm in parsed_payments:
                if pm.payment_id not in user.payment_methods:
                    msg = f"Payment method {pm.payment_id} not found"
                    raise ToolException(msg)  # noqa: TRY301
                upm = user.payment_methods[pm.payment_id]
                if upm.source in {"gift_card", "certificate"} and upm.amount < pm.amount:
                    msg = f"Not enough balance in payment method {pm.payment_id}"
                    raise ToolException(msg)  # noqa: TRY301

            total_payment = sum(p.amount for p in parsed_payments)
            if total_payment != total_price:
                msg = f"Payment amount mismatch: total price is {total_price}, paid {total_payment}"
                raise ToolException(msg)  # noqa: TRY301

            for pm in parsed_payments:
                upm = user.payment_methods[pm.payment_id]
                if upm.source == "gift_card":
                    upm.amount -= pm.amount
                elif upm.source == "certificate":
                    user.payment_methods.pop(pm.payment_id)

            for fds in all_flight_dates:
                fds.available_seats[cabin] -= len(parsed_passengers)

            db.reservations[rid] = reservation
            db.users[user_id].reservations.append(rid)

        except ToolException as exc:
            return _log_error("book_reservation", raw_args, str(exc))
        return _log_and_return("book_reservation", raw_args, _serialize(reservation))

    # --- update_reservation_flights ---
    def update_reservation_flights(
        reservation_id: str,
        cabin: CabinClass,
        flights: list[dict[str, str] | FlightInfo],
        payment_id: str,
    ) -> str:
        raw_args: dict[str, Any] = {
            "reservation_id": reservation_id,
            "cabin": cabin,
            "flights": [f.model_dump() if isinstance(f, BaseModel) else f for f in flights],
            "payment_id": payment_id,
        }
        try:
            parsed_flights = [FlightInfo(**f) if isinstance(f, dict) else f for f in flights]
            reservation = _get_reservation(db, reservation_id)
            user = _get_user(db, reservation.user_id)

            total_price = 0
            new_flights: list[ReservationFlight] = []

            for fi in parsed_flights:
                existing = next(
                    (rf for rf in reservation.flights if rf.flight_number == fi.flight_number and rf.date == fi.date and cabin == reservation.cabin),
                    None,
                )
                if existing:
                    total_price += existing.price * len(reservation.passengers)
                    new_flights.append(existing)
                    continue

                flight = _get_flight(db, fi.flight_number)
                fds = _get_flight_instance(db, fi.flight_number, fi.date)
                if not isinstance(fds, FlightDateStatusAvailable):
                    msg = f"Flight {fi.flight_number} not available on date {fi.date}"
                    raise ToolException(msg)  # noqa: TRY301
                if fds.available_seats.get(cabin, 0) < len(reservation.passengers):
                    msg = f"Not enough seats on flight {fi.flight_number}"
                    raise ToolException(msg)  # noqa: TRY301
                rf = ReservationFlight(
                    flight_number=fi.flight_number,
                    date=fi.date,
                    price=fds.prices[cabin],
                    origin=flight.origin,
                    destination=flight.destination,
                )
                total_price += rf.price * len(reservation.passengers)
                new_flights.append(rf)

            total_price -= sum(f.price for f in reservation.flights) * len(reservation.passengers)
            payment = _payment_for_update(user, payment_id, total_price)
            if payment is not None:
                reservation.payment_history.append(payment)

            reservation.flights = new_flights
            reservation.cabin = cabin

        except ToolException as exc:
            return _log_error("update_reservation_flights", raw_args, str(exc))
        return _log_and_return("update_reservation_flights", raw_args, _serialize(reservation))

    # --- update_reservation_baggages ---
    def update_reservation_baggages(
        reservation_id: str,
        total_baggages: int,
        nonfree_baggages: int,
        payment_id: str,
    ) -> str:
        args = {
            "reservation_id": reservation_id,
            "total_baggages": total_baggages,
            "nonfree_baggages": nonfree_baggages,
            "payment_id": payment_id,
        }
        try:
            reservation = _get_reservation(db, reservation_id)
            user = _get_user(db, reservation.user_id)
            cost = 50 * max(0, nonfree_baggages - reservation.nonfree_baggages)
            payment = _payment_for_update(user, payment_id, cost)
            if payment is not None:
                reservation.payment_history.append(payment)
            reservation.total_baggages = total_baggages
            reservation.nonfree_baggages = nonfree_baggages
        except ToolException as exc:
            return _log_error("update_reservation_baggages", args, str(exc))
        return _log_and_return("update_reservation_baggages", args, _serialize(reservation))

    # --- update_reservation_passengers ---
    def update_reservation_passengers(
        reservation_id: str,
        passengers: list[dict[str, str] | Passenger],
    ) -> str:
        raw_args: dict[str, Any] = {
            "reservation_id": reservation_id,
            "passengers": [p.model_dump() if isinstance(p, BaseModel) else p for p in passengers],
        }
        try:
            parsed = [Passenger(**p) if isinstance(p, dict) else p for p in passengers]
            reservation = _get_reservation(db, reservation_id)
            if len(parsed) != len(reservation.passengers):
                msg = "Number of passengers does not match"
                raise ToolException(msg)  # noqa: TRY301
            reservation.passengers = deepcopy(parsed)
        except ToolException as exc:
            return _log_error("update_reservation_passengers", raw_args, str(exc))
        return _log_and_return("update_reservation_passengers", raw_args, _serialize(reservation))

    # --- send_certificate ---
    def send_certificate(user_id: str, amount: int) -> str:
        args = {"user_id": user_id, "amount": amount}
        try:
            user = _get_user(db, user_id)
            for pid in [f"certificate_{i}" for i in _NEW_PAYMENT_IDS]:
                if pid not in user.payment_methods:
                    new_cert = Certificate(id=pid, amount=amount, source="certificate")
                    user.payment_methods[pid] = new_cert
                    result = f"Certificate {pid} added to user {user_id} with amount {amount}."
                    return _log_and_return("send_certificate", args, result)
            msg = "Too many certificates"
            raise ToolException(msg)  # noqa: TRY301
        except ToolException as exc:
            return _log_error("send_certificate", args, str(exc))

    # --- calculate ---
    def calculate(expression: str) -> str:
        args = {"expression": expression}
        if not all(c in "0123456789+-*/(). " for c in expression):
            return _log_error("calculate", args, "Invalid characters in expression")
        result = str(round(float(eval(expression, {"__builtins__": None}, {})), 2))  # noqa: S307
        return _log_and_return("calculate", args, result)

    # --- transfer_to_human_agents ---
    def transfer_to_human_agents(summary: str) -> str:
        args = {"summary": summary}
        return _log_and_return("transfer_to_human_agents", args, "Transfer successful")

    # --- list_all_airports ---
    def list_all_airports() -> str:
        airports = [
            AirportCode(iata="SFO", city="San Francisco"),
            AirportCode(iata="JFK", city="New York"),
            AirportCode(iata="LAX", city="Los Angeles"),
            AirportCode(iata="ORD", city="Chicago"),
            AirportCode(iata="DFW", city="Dallas"),
            AirportCode(iata="DEN", city="Denver"),
            AirportCode(iata="SEA", city="Seattle"),
            AirportCode(iata="ATL", city="Atlanta"),
            AirportCode(iata="MIA", city="Miami"),
            AirportCode(iata="BOS", city="Boston"),
            AirportCode(iata="PHX", city="Phoenix"),
            AirportCode(iata="IAH", city="Houston"),
            AirportCode(iata="LAS", city="Las Vegas"),
            AirportCode(iata="MCO", city="Orlando"),
            AirportCode(iata="EWR", city="Newark"),
            AirportCode(iata="CLT", city="Charlotte"),
            AirportCode(iata="MSP", city="Minneapolis"),
            AirportCode(iata="DTW", city="Detroit"),
            AirportCode(iata="PHL", city="Philadelphia"),
            AirportCode(iata="LGA", city="LaGuardia"),
        ]
        return _log_and_return("list_all_airports", {}, _serialize(airports))

    # --- get_flight_status ---
    def get_flight_status(flight_number: str, date: str) -> str:
        args = {"flight_number": flight_number, "date": date}
        try:
            fds = _get_flight_instance(db, flight_number, date)
        except ToolException as exc:
            return _log_error("get_flight_status", args, str(exc))
        return _log_and_return("get_flight_status", args, fds.status)

    # --- Assemble StructuredTools ---
    tools = [
        StructuredTool.from_function(
            func=get_user_details,
            name="get_user_details",
            description="Get the details of a user, including their reservations.",
        ),
        StructuredTool.from_function(
            func=get_reservation_details,
            name="get_reservation_details",
            description="Get the details of a reservation.",
        ),
        StructuredTool.from_function(
            func=search_direct_flight,
            name="search_direct_flight",
            description="Search for direct flights between two cities on a specific date.",
        ),
        StructuredTool.from_function(
            func=search_onestop_flight,
            name="search_onestop_flight",
            description="Search for one-stop flights between two cities on a specific date.",
        ),
        StructuredTool.from_function(
            func=cancel_reservation,
            name="cancel_reservation",
            description="Cancel a reservation. Returns the updated reservation.",
        ),
        StructuredTool.from_function(
            func=book_reservation,
            name="book_reservation",
            description=(
                "Book a new reservation. Requires user_id, origin, destination, "
                "flight_type, cabin, flights, passengers, payment_methods, "
                "total_baggages, nonfree_baggages, and insurance."
            ),
            args_schema=BookReservationInput,
        ),
        StructuredTool.from_function(
            func=update_reservation_flights,
            name="update_reservation_flights",
            description=(
                "Update the flights of a reservation. All flight segments must be "
                "included (even unchanged ones). Requires a payment method for any price difference."
            ),
            args_schema=UpdateReservationFlightsInput,
        ),
        StructuredTool.from_function(
            func=update_reservation_baggages,
            name="update_reservation_baggages",
            description="Update the baggage information of a reservation.",
        ),
        StructuredTool.from_function(
            func=update_reservation_passengers,
            name="update_reservation_passengers",
            description="Update passenger information of a reservation. Cannot change the number of passengers.",
            args_schema=UpdateReservationPassengersInput,
        ),
        StructuredTool.from_function(
            func=send_certificate,
            name="send_certificate",
            description="Send a compensation certificate to a user.",
        ),
        StructuredTool.from_function(
            func=calculate,
            name="calculate",
            description=("Calculate the result of a mathematical expression. Supports +, -, *, /, parentheses."),
        ),
        StructuredTool.from_function(
            func=transfer_to_human_agents,
            name="transfer_to_human_agents",
            description=(
                "Transfer the user to a human agent with a summary. "
                "Only use when the request cannot be handled within policy scope, "
                "or the user explicitly asks for a human agent."
            ),
        ),
        StructuredTool.from_function(
            func=list_all_airports,
            name="list_all_airports",
            description="List all available airports with IATA codes and city names.",
        ),
        StructuredTool.from_function(
            func=get_flight_status,
            name="get_flight_status",
            description="Get the status of a flight on a specific date.",
        ),
    ]

    return tools, log
