import random
from copy import deepcopy
from typing import Dict, List, Union

from .long_context import (
    CAR_STATUS_METADATA_EXTENSION,
    INTERMEDIARY_CITIES,
    LONG_WEATHER_EXTENSION,
    PARKING_BRAKE_INSTRUCTION,
)

MAX_FUEL_LEVEL = 50
MIN_FUEL_LEVEL = 0.0
MILE_PER_GALLON = 20.0
MAX_BATTERY_VOLTAGE = 14.0
MIN_BATTERY_VOLTAGE = 10.0

DEFAULT_STATE = {
    "random_seed": 141053,
    "fuelLevel": 0.0,
    "batteryVoltage": 12.6,
    "engine_state": "stopped",
    "remainingUnlockedDoors": 4,
    "doorStatus": {
        "driver": "unlocked",
        "passenger": "unlocked",
        "rear_left": "unlocked",
        "rear_right": "unlocked",
    },
    "acTemperature": 25.0,
    "fanSpeed": 50,
    "acMode": "auto",
    "humidityLevel": 50.0,
    "headLightStatus": "off",
    "parkingBrakeStatus": "released",
    "_parkingBrakeForce": 0.0,
    "_slopeAngle": 0.0,
    "brakePedalStatus": "released",
    "brakePedalForce": 0.0,
    "distanceToNextVehicle": 50.0,
    "cruiseStatus": "inactive",
    "destination": "None",
    "frontLeftTirePressure": 32.0,
    "frontRightTirePressure": 32.0,
    "rearLeftTirePressure": 30.0,
    "rearRightTirePressure": 30.0,
}


class VehicleControlAPI:
    def __init__(self):
        """
        Initializes the vehicle control API with default values.
        """
        self.fuelLevel: float
        self.batteryVoltage: float
        self.engine_state: str
        self.remainingUnlockedDoors: int
        self.doorStatus: Dict[str, str]

        self.acTemperature: float
        self.fanSpeed: int
        self.acMode: str
        self.humidityLevel: float
        self.headLightStatus: str
        self.parkingBrakeStatus: str
        self._parkingBrakeForce: float
        self._slopeAngle: float
        self.brakePedalStatus: str
        self._brakePedalForce: float
        self.distanceToNextVehicle: float
        self.cruiseStatus: str
        self.destination: str
        self.frontLeftTirePressure: float
        self.frontRightTirePressure: float
        self.rearLeftTirePressure: float
        self.rearRightTirePressure: float
        self._api_description = "This tool belongs to the vehicle control system, which allows users to control various aspects of the car such as engine, doors, climate control, lights, and more."

    def _load_scenario(self, scenario: dict, long_context=False) -> None:
        """
        Loads the scenario for the vehicle control.
        Args:
            scenario (Dict): The scenario to load.
        """
        DEFAULT_STATE_COPY = deepcopy(DEFAULT_STATE)
        self._random = random.Random(
            (scenario.get("random_seed", DEFAULT_STATE_COPY["random_seed"]))
        )
        self.fuelLevel = scenario.get(
            "fuelLevel", DEFAULT_STATE_COPY["fuelLevel"]
        )  # in gallons
        self.batteryVoltage = scenario.get(
            "batteryVoltage", DEFAULT_STATE_COPY["batteryVoltage"]
        )  # in volts
        self.engine_state = scenario.get(
            "engineState", DEFAULT_STATE_COPY["engine_state"]
        )  # running, stopped
        self.remainingUnlockedDoors = scenario.get(
            "remainingUnlockedDoors", DEFAULT_STATE_COPY["remainingUnlockedDoors"]
        )  # driver, passenger, rear_left, rear_right
        self.doorStatus = scenario.get(
            "doorStatus",
            DEFAULT_STATE_COPY["doorStatus"],
        )
        self.remainingUnlockedDoors = 4 - len(
            [1 for door in self.doorStatus.keys() if self.doorStatus[door] == "locked"]
        )
        self.acTemperature = scenario.get(
            "acTemperature", DEFAULT_STATE_COPY["acTemperature"]
        )  # in degree Celsius
        self.fanSpeed = scenario.get(
            "fanSpeed", DEFAULT_STATE_COPY["fanSpeed"]
        )  # 0 to 100
        self.acMode = scenario.get(
            "acMode", DEFAULT_STATE_COPY["acMode"]
        )  # auto, cool, heat, defrost
        self.humidityLevel = scenario.get(
            "humidityLevel", DEFAULT_STATE_COPY["humidityLevel"]
        )  # in percentage
        self.headLightStatus = scenario.get(
            "headLightStatus", DEFAULT_STATE_COPY["headLightStatus"]
        )  # on, off
        self.parkingBrakeStatus = scenario.get(
            "parkingBrakeStatus", DEFAULT_STATE_COPY["parkingBrakeStatus"]
        )  # released, engaged
        self._parkingBrakeForce = scenario.get(
            "parkingBrakeForce", DEFAULT_STATE_COPY["_parkingBrakeForce"]
        )  # in Newtons
        self._slopeAngle = scenario.get(
            "slopeAngle", DEFAULT_STATE_COPY["_slopeAngle"]
        )  # in degrees
        self.brakePedalStatus = scenario.get(
            "brakePedalStatus", DEFAULT_STATE_COPY["brakePedalStatus"]
        )  # pressed, released
        self._brakePedalForce = scenario.get(
            "brakePedalForce", DEFAULT_STATE_COPY["brakePedalForce"]
        )  # in Newtons
        self.distanceToNextVehicle = scenario.get(
            "distanceToNextVehicle", DEFAULT_STATE_COPY["distanceToNextVehicle"]
        )  # in meters
        self.cruiseStatus = scenario.get(
            "cruiseStatus", DEFAULT_STATE_COPY["cruiseStatus"]
        )  # active, inactive
        self.destination = scenario.get(
            "destination", DEFAULT_STATE_COPY["destination"]
        )
        self.frontLeftTirePressure = scenario.get(
            "frontLeftTirePressure", DEFAULT_STATE_COPY["frontLeftTirePressure"]
        )
        self.frontRightTirePressure = scenario.get(
            "frontRightTirePressure", DEFAULT_STATE_COPY["frontRightTirePressure"]
        )
        self.rearLeftTirePressure = scenario.get(
            "rearLeftTirePressure", DEFAULT_STATE_COPY["rearLeftTirePressure"]
        )
        self.rearRightTirePressure = scenario.get(
            "rearRightTirePressure", DEFAULT_STATE_COPY["rearRightTirePressure"]
        )

        self.long_context = long_context

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, VehicleControlAPI):
            return False

        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            model_attr = getattr(self, attr_name)
            ground_truth_attr = getattr(value, attr_name)

            if model_attr != ground_truth_attr:
                return False

        return True

    def startEngine(self, ignitionMode: str) -> Dict[str, Union[str, float]]:
        """
        Starts the engine of the vehicle.
        Args:
            ignitionMode (str): The ignition mode of the vehicle. [Enum]: ["START", "STOP"]
        Returns:
            engineState (str): The state of the engine. [Enum]: ["running", "stopped"]
            fuelLevel (float): The fuel level of the vehicle in gallons.
            batteryVoltage (float): The battery voltage of the vehicle in volts.
        """
        if ignitionMode == "STOP":
            self.engine_state = "stopped"
        if self.remainingUnlockedDoors > 0:
            return {
                "error": "All doors must be locked before starting the engine. Here are the unlocked doors: "
                + ", ".join(
                    [
                        door
                        for door, status in self.doorStatus.items()
                        if status == "unlocked"
                    ]
                )
            }
        if self.brakePedalStatus != "pressed":
            return {
                "error": "Brake pedal needs to be pressed when starting the engine."
            }
        if self._brakePedalForce != 1000.0:
            return {"error": "Must press the brake fully before starting the engine."}
        if self.fuelLevel < MIN_FUEL_LEVEL:
            return {"error": "Fuel tank is empty."}
        if ignitionMode == "START":
            self.engine_state = "running"
        else:
            return {"error": "Invalid ignition mode."}

        return {
            "engineState": self.engine_state,
            "fuelLevel": self.fuelLevel,
            "batteryVoltage": self.batteryVoltage,
        }

    def fillFuelTank(self, fuelAmount: float) -> Dict[str, Union[str, float]]:
        """
        Fills the fuel tank of the vehicle. The fuel tank can hold up to 50 gallons.
        Args:
            fuelAmount (float): The amount of fuel to fill in gallons; this is the additional fuel to add to the tank.
        Returns:
            fuelLevel (float): The fuel level of the vehicle in gallons.
        """
        if fuelAmount < 0:
            return {"error": "Fuel amount cannot be negative."}
        if self.fuelLevel + fuelAmount > MAX_FUEL_LEVEL:
            return {"error": "Cannot fill gas above the tank capacity."}
        if self.fuelLevel + fuelAmount < MIN_FUEL_LEVEL:
            return {"error": "Fuel tank is empty. Min fuel level is 0 gallons."}
        self.fuelLevel += fuelAmount
        return {"fuelLevel": self.fuelLevel}

    def lockDoors(self, unlock: bool, door: list[str]) -> Dict[str, Union[str, int]]:
        """
        Locks the doors of the vehicle.
        Args:
            unlock (bool): True if the doors are to be unlocked, False otherwise.
            door (List[str]): The list of doors to lock or unlock. [Enum]: ["driver", "passenger", "rear_left", "rear_right"]
        Returns:
            lockStatus (str): The status of the lock. [Enum]: ["locked", "unlocked"]
            remainingUnlockedDoors (int): The number of remaining unlocked doors.
        """
        if unlock:
            for d in door:
                if self.doorStatus[d] == "unlocked":
                    continue
                self.doorStatus[d] = "unlocked"
                self.remainingUnlockedDoors += 1
            return {
                "lockStatus": "unlocked",
                "remainingUnlockedDoors": self.remainingUnlockedDoors,
            }
        else:
            for d in door:
                if self.doorStatus[d] == "locked":
                    continue
                self.doorStatus[d] = "locked"
                self.remainingUnlockedDoors -= 1
            return {
                "lockStatus": "locked",
                "remainingUnlockedDoors": self.remainingUnlockedDoors,
            }

    def adjustClimateControl(
        self,
        temperature: float,
        unit: str = "celsius",
        fanSpeed: int = 50,
        mode: str = "auto",
    ) -> Dict[str, Union[str, float]]:
        """
        Adjusts the climate control of the vehicle.
        Args:
            temperature (float): The temperature to set in degree. Default to be celsius.
            unit (str): [Optional] The unit of temperature. [Enum]: ["celsius", "fahrenheit"]
            fanSpeed (int): [Optional] The fan speed to set from 0 to 100. Default is 50.
            mode (str): [Optional] The climate mode to set. [Enum]: ["auto", "cool", "heat", "defrost"]
        Returns:
            currentTemperature (float): The current temperature set in degree Celsius.
            climateMode (str): The current climate mode set.
            humidityLevel (float): The humidity level in percentage.
        """
        if not (0 <= fanSpeed <= 100):
            return {"error": "Fan speed must be between 0 and 100."}
        self.acTemperature = temperature
        if unit == "fahrenheit":
            self.acTemperature = (temperature - 32) * 5 / 9
        self.fanSpeed = fanSpeed
        self.acMode = mode
        return {
            "currentACTemperature": temperature,
            "climateMode": mode,
            "humidityLevel": self.humidityLevel,
        }

    def get_outside_temperature_from_google(self) -> Dict[str, float]:
        """
        Gets the outside temperature.
        Returns:
            outsideTemperature (float): The outside temperature in degree Celsius.
        """
        if self.long_context:
            LONG_WEATHER_EXTENSION["outsideTemperature"] = self._random.uniform(
                -10.0, 40.0
            )
            return LONG_WEATHER_EXTENSION
        return {"outsideTemperature": self._random.uniform(-10.0, 40.0)}

    def get_outside_temperature_from_weather_com(self) -> Dict[str, float]:
        """
        Gets the outside temperature.
        Returns:
            outsideTemperature (float): The outside temperature in degree Celsius.
        """
        return {"error": 404}

    def setHeadlights(self, mode: str) -> Dict[str, str]:
        """
        Sets the headlights of the vehicle.
        Args:
            mode (str): The mode of the headlights. [Enum]: ["on", "off", "auto"]
        Returns:
            headlightStatus (str): The status of the headlights. [Enum]: ["on", "off"]
        """
        if mode not in ["on", "off", "auto"]:
            return {"error": "Invalid headlight mode."}
        if mode == "on":
            self.headLightStatus = "on"
            return {"headlightStatus": "on"}
        else:
            self.headLightStatus = "off"
            return {"headlightStatus": "off"}

    def displayCarStatus(
        self, option: str
    ) -> Dict[str, Union[str, float, Dict[str, str]]]:
        """
        Displays the status of the vehicle based on the provided display option.
        Args:
            option (str): The option to display. [Enum]: ["fuel", "battery", "doors", "climate", "headlights", "parkingBrake", "brakePedal", "engine"]
        Returns:
            status (Dict): The status of the vehicle based on the option.
                - fuelLevel (float): [Optional] The fuel level of the vehicle in gallons.
                - batteryVoltage (float): [Optional] The battery voltage of the vehicle in volts.
                - doorStatus (Dict): [Optional] The status of the doors.
                    - driver (str): The status of the driver door. [Enum]: ["locked", "unlocked"]
                    - passenger (str): The status of the passenger door. [Enum]: ["locked", "unlocked"]
                    - rear_left (str): The status of the rear left door. [Enum]: ["locked", "unlocked"]
                    - rear_right (str): The status of the rear right door. [Enum]: ["locked", "unlocked"]
                - currentACTemperature (float): [Optional] The current temperature set in degree Celsius.
                - fanSpeed (int): [Optional] The fan speed set from 0 to 100.
                - climateMode (str): [Optional] The climate mode set. [Enum]: ["auto", "cool", "heat", "defrost"]
                - humidityLevel (float): [Optional] The humidity level in percentage.
                - headlightStatus (str): [Optional] The status of the headlights. [Enum]: ["on", "off"]
                - parkingBrakeStatus (str): [Optional] The status of the brake. [Enum]: ["engaged", "released"]
                - parkingBrakeForce (float): [Optional] The force applied to the brake in Newtons.
                - slopeAngle (float): [Optional] The slope angle in degrees.
                - brakePedalStatus (str): [Optional] The status of the brake pedal. [Enum]: ["pressed", "released"]
                - brakePedalForce (float): [Optional] The force applied to the brake pedal in Newtons.
                - engineState (str): [Optional] The state of the engine. [Enum]: ["running", "stopped"]
                - metadata (str): [Optional] The metadata of the car.
        """
        status = {}
        if self.long_context:
            status["metadata"] = CAR_STATUS_METADATA_EXTENSION
        if option == "fuel":
            status["fuelLevel"] = self.fuelLevel
        elif option == "battery":
            status["batteryVoltage"] = self.batteryVoltage
        elif option == "doors":
            status["doorStatus"] = self.doorStatus
        elif option == "climate":
            status["currentACTemperature"] = self.acTemperature
            status["fanSpeed"] = self.fanSpeed
            status["climateMode"] = self.acMode
            status["humidityLevel"] = self.humidityLevel
        elif option == "headlights":
            status["headlightStatus"] = self.headLightStatus
        elif option == "parkingBrake":
            status["parkingBrakeStatus"] = self.parkingBrakeStatus
            status["parkingBrakeForce"] = self._parkingBrakeForce
            status["slopeAngle"] = self._slopeAngle
        elif option == "brakePedal":
            status["brakePedalStatus"] = self.brakePedalStatus
            status["brakePedalForce"] = self._brakePedalForce
        elif option == "engine":
            status["engineState"] = self.engine_state
        else:
            status["error"] = "Invalid option"
        return status

    def activateParkingBrake(self, mode: str) -> Dict[str, Union[str, float]]:
        """
        Activates the parking brake of the vehicle.
        Args:
            mode (str): The mode to set. [Enum]: ["engage", "release"]
        Returns:
            parkingBrakeStatus (str): The status of the brake. [Enum]: ["engaged", "released"]
            _parkingBrakeForce (float): The force applied to the brake in Newtons.
            _slopeAngle (float): The slope angle in degrees.
        """
        if mode not in ["engage", "release"]:
            return {"error": "Invalid mode"}
        if mode == "engage":
            self.parkingBrakeStatus = "engaged"
            self._parkingBrakeForce = 500.0
            self._slopeAngle = 10.0
            if self.long_context:
                return {
                    "parkingBrakeInstruction": PARKING_BRAKE_INSTRUCTION,
                    "parkingBrakeStatus": "engaged",
                    "_parkingBrakeForce": 500.0,
                    "_slopeAngle": 10.0,
                }
            return {
                "parkingBrakeStatus": "engaged",
                "_parkingBrakeForce": 500.0,
                "_slopeAngle": 10.0,
            }
        else:
            self.parkingBrakeStatus = "released"
            self._parkingBrakeForce = 0.0
            self._slopeAngle = 10.0
            if self.long_context:
                return {
                    "parkingBrakeInstruction": PARKING_BRAKE_INSTRUCTION,
                    "parkingBrakeStatus": "released",
                    "_parkingBrakeForce": 0.0,
                    "_slopeAngle": 10.0,
                }
            return {
                "parkingBrakeStatus": "released",
                "_parkingBrakeForce": 0.0,
                "_slopeAngle": 10.0,
            }

    def pressBrakePedal(self, pedalPosition: float) -> Dict[str, Union[str, float]]:
        """
        Presses the brake pedal based on pedal position. The brake pedal will be kept pressed until released.

        Args:
            pedalPosition (float): Position of the brake pedal, between 0 (not pressed) and 1 (fully pressed).
        Returns:
            brakePedalStatus (str): The status of the brake pedal. [Enum]: ["pressed", "released"]
            brakePedalForce (float): The force applied to the brake pedal in Newtons.
        """
        # Validate pedal position is within 0 to 1
        if not (0 <= pedalPosition <= 1):
            return {"error": "Pedal position must be between 0 and 1."}

        # Release the brake if pedal position is zero
        if pedalPosition == 0:
            self.brakePedalStatus = "released"
            self._brakePedalForce = 0.0
            return {"brakePedalStatus": "released", "brakePedalForce": 0.0}

        # Calculate force based on pedal position
        max_brake_force = 1000  # Max force in Newtons
        force = pedalPosition * max_brake_force

        # Update the brake pedal status and force
        self.brakePedalStatus = "pressed"
        self._brakePedalForce = force
        return {"brakePedalStatus": "pressed", "brakePedalForce": float(force)}

    def releaseBrakePedal(self) -> Dict[str, Union[str, float]]:
        """
        Releases the brake pedal of the vehicle.
        Returns:
            brakePedalStatus (str): The status of the brake pedal. [Enum]: ["pressed", "released"]
            brakePedalForce (float): The force applied to the brake pedal in Newtons.
        """
        self.brakePedalStatus = "released"
        self._brakePedalForce = 0.0
        return {"brakePedalStatus": "released", "brakePedalForce": 0.0}

    def setCruiseControl(
        self, speed: float, activate: bool, distanceToNextVehicle: float
    ) -> Dict[str, Union[str, float]]:
        """
        Sets the cruise control of the vehicle.
        Args:
            speed (float): The speed to set in m/h. The speed should be between 0 and 120 and a multiple of 5.
            activate (bool): True to activate the cruise control, False to deactivate.
            distanceToNextVehicle (float): The distance to the next vehicle in meters.
        Returns:
            cruiseStatus (str): The status of the cruise control. [Enum]: ["active", "inactive"]
            currentSpeed (float): The current speed of the vehicle in km/h.
            distanceToNextVehicle (float): The distance to the next vehicle in meters.
        """
        distanceToNextVehicle = float(distanceToNextVehicle)
        speed = float(speed)

        if self.engine_state == "stopped":
            return {"error": "Start the engine before activating the cruise control."}
        if activate:
            self.distanceToNextVehicle = distanceToNextVehicle
            if speed < 0 or speed > 120 or speed % 5 != 0:
                return {"error": "Invalid speed"}
            self.cruiseStatus = "active"
            return {
                "cruiseStatus": "active",
                "currentSpeed": speed,
                "distanceToNextVehicle": distanceToNextVehicle,
            }
        else:
            self.cruiseStatus = "inactive"
            self.distanceToNextVehicle = distanceToNextVehicle
            return {
                "cruiseStatus": "inactive",
                "currentSpeed": speed,
                "distanceToNextVehicle": distanceToNextVehicle,
            }

    def get_current_speed(self) -> Dict[str, float]:
        """
        Gets the current speed of the vehicle.
        Returns:
            currentSpeed (float): The current speed of the vehicle in km/h.
        """
        return {"currentSpeed": self._random.uniform(0.0, 120.0)}

    def display_log(self, messages: List[str]):
        """
        Displays the log messages.
        Args:
            messages (List[str]): The list of messages to display.
        Returns:
            log (List[str]): The list of messages displayed.
        """
        return {"log": messages}

    def estimate_drive_feasibility_by_mileage(self, distance: float) -> Dict[str, bool]:
        """
        Estimates the milage of the vehicle given the distance needed to drive.
        Args:
            distance (float): The distance to travel in miles.
        Returns:
            canDrive (bool): True if the vehicle can drive the distance, False otherwise.
        """
        if self.fuelLevel * MILE_PER_GALLON < distance:
            return {"canDrive": False}
        else:
            return {"canDrive": True}

    def liter_to_gallon(self, liter: float) -> Dict[str, float]:
        """
        Converts the liter to gallon.
        Args:
            liter (float): The amount of liter to convert.
        Returns:
            gallon (float): The amount of gallon converted.
        """
        return {"gallon": liter * 0.264172}

    def gallon_to_liter(self, gallon: float) -> Dict[str, float]:
        """
        Converts the gallon to liter.
        Args:
            gallon (float): The amount of gallon to convert.
        Returns:
            liter (float): The amount of liter converted.
        """
        return {"liter": gallon * 3.78541}

    def estimate_distance(self, cityA: str, cityB: str) -> Dict[str, float]:
        """
        Estimates the distance between two cities.
        Args:
            cityA (str): The zipcode of the first city.
            cityB (str): The zipcode of the second city.
        Returns:
            distance (float): The distance between the two cities in km.
            intermediaryCities (List[str]): [Optional] The list of intermediary cities between the two cities.
        """
        if (cityA == "83214" and cityB == "74532") or (
            cityA == "74532" and cityB == "83214"
        ):
            distance = {"distance": 750.0}
        elif (cityA == "56108" and cityB == "62947") or (
            cityA == "62947" and cityB == "56108"
        ):
            distance = {"distance": 320.0}
        elif (cityA == "71354" and cityB == "83462") or (
            cityA == "83462" and cityB == "71354"
        ):
            distance = {"distance": 450.0}
        elif (cityA == "47329" and cityB == "52013") or (
            cityA == "52013" and cityB == "47329"
        ):
            distance = {"distance": 290.0}
        elif (cityA == "69238" and cityB == "51479") or (
            cityA == "51479" and cityB == "69238"
        ):
            distance = {"distance": 630.0}
        elif (cityA == "94016" and cityB == "83214") or (
            cityA == "83214" and cityB == "94016"
        ):
            distance = {"distance": 980.0}
        elif (cityA == "94016" and cityB == "94704") or (
            cityA == "94704" and cityB == "94016"
        ):
            distance = {"distance": 600.0}
        elif (cityA == "94704" and cityB == "08540") or (
            cityA == "08540" and cityB == "94704"
        ):
            distance = {"distance": 2550.0}
        elif (cityA == "94016" and cityB == "08540") or (
            cityA == "08540" and cityB == "94016"
        ):
            distance = {"distance": 1950.0}
        elif (cityA == "62947" and cityB == "47329") or (
            cityA == "47329" and cityB == "62947"
        ):
            distance = {"distance": 1053.0}
        elif (cityA == "94016" and cityB == "62947") or (
            cityA == "62947" and cityB == "94016"
        ):
            distance = {"distance": 780.0}
        elif (cityA == "74532" and cityB == "94016") or (
            cityA == "94016" and cityB == "74532"
        ):
            distance = {"distance": 880.0}
        else:
            distance = {"error": "distance not found in database."}

        if self.long_context:
            distance["intermediaryCities"] = INTERMEDIARY_CITIES
        return distance

    def get_zipcode_based_on_city(self, city: str) -> Dict[str, str]:
        """
        Gets the zipcode based on the city.
        Args:
            city (str): The name of the city.
        Returns:
            zipcode (str): The zipcode of the city.
        """
        if city == "Rivermist":
            return {"zipcode": "83214"}
        elif city == "Stonebrook":
            return {"zipcode": "74532"}
        elif city == "Maplecrest":
            return {"zipcode": "56108"}
        elif city == "Silverpine":
            return {"zipcode": "62947"}
        elif city == "Shadowridge":
            return {"zipcode": "71354"}
        elif city == "Sunset Valley":
            return {"zipcode": "83462"}
        elif city == "Oakendale":
            return {"zipcode": "47329"}
        elif city == "Willowbend":
            return {"zipcode": "52013"}
        elif city == "Crescent Hollow":
            return {"zipcode": "69238"}
        elif city == "Autumnville":
            return {"zipcode": "51479"}
        elif city == "San Francisco":
            return {"zipcode": "94016"}
        else:
            return {"zipcode": "00000"}

    def set_navigation(self, destination: str) -> Dict[str, str]:
        """
        Navigates to the destination.
        Args:
            destination (str): The destination to navigate in the format of street, city, state.
        Returns:
            status (str): The status of the navigation.
        """
        self.destination = destination
        return {"status": "Navigating to " + destination}

    def check_tire_pressure(self):
        """
        Checks the tire pressure of the vehicle.
        Returns:
            tirePressure (Dict): The tire pressure of the vehicle.
                - frontLeftTirePressure (float): The pressure of the front left tire in psi.
                - frontRightTirePressure (float): The pressure of the front right tire in psi.
                - rearLeftTirePressure (float): The pressure of the rear left tire in psi.
                - rearRightTirePressure (float): The pressure of the rear right tire in psi.
                - healthy_tire_pressure (bool): True if the tire pressure is healthy, False otherwise.
                - car_info (Dict): The metadata of the car.
        """
        # This is the healthy standard the vehicle use, though the user might have different preferences
        healthy_tire_pressure = (
            30
            <= (
                self.frontLeftTirePressure
                + self.frontRightTirePressure
                + self.rearLeftTirePressure
                + self.rearRightTirePressure
            )
            / 4
            <= 35
        )

        tire_status = {
            "frontLeftTirePressure": self.frontLeftTirePressure,
            "frontRightTirePressure": self.frontRightTirePressure,
            "rearLeftTirePressure": self.rearLeftTirePressure,
            "rearRightTirePressure": self.rearRightTirePressure,
            "healthy_tire_pressure": healthy_tire_pressure,
            "car_info": {},
        }

        if self.long_context:
            tire_status["car_info"] = CAR_STATUS_METADATA_EXTENSION
        return tire_status

    def find_nearest_tire_shop(self) -> Dict[str, str]:
        """
        Finds the nearest tire shop.
        Returns:
            shopLocation (str): The location of the nearest tire shop.
        """
        return {"shopLocation": "456 Oakwood Avenue, Rivermist, 83214"}
