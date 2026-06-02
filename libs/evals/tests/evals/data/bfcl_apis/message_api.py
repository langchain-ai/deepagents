import random
from copy import deepcopy
from typing import Dict, List, Optional, Union

DEFAULT_STATE = {
    "generated_ids": set(),
    "user_count": 4,
    "user_map": {
        "Alice": "USR001",
        "Bob": "USR002",
        "Catherine": "USR003",
        "Daniel": "USR004",
    },
    "inbox": [
        {
            "USR002": "My name is Alice. I want to connect.",
        },
        {
            "USR003": "Could you upload the file?",
        },
        {
            "USR004": "Could you upload the file?",
        },
    ],
    "message_count": 3,
    "current_user": None,
}


class MessageAPI:
    """
    A class representing a Message API for managing user interactions in a workspace.

    This class provides methods for user management, messaging, and message retrieval
    within a specific workspace. It maintains user information, sent messages, and
    received messages for each user.

    Attributes:
        user_map (Dict[str, str]): A mapping of user names to user IDs.
        inbox (Dict[int, Dict[str, Union[str, int]]]): A dictionary storing all messages.
        message_count (int): The total count of messages in the workspace.
        current_user (Optional[str]): The ID of the currently logged-in user.

    Methods:
        generate_id(): Generate a unique ID for a message.
        list_users(): List all users in the workspace.
        get_user_id(user: str): Get the user ID for a given username.
        login(user_id: str): Log in a user.
        send_message(receiver_id: str, message: str): Send a message to another user.
        view_messages_sent(): View messages sent by the current user.
        delete_message(receiver_id: str): Delete a sent message.
        add_contact(name: str, user_id: str): Add a new contact to the workspace.
        search_messages(keyword: str): Search for messages containing a keyword.
        get_message_stats(): Get messaging statistics for the current user.
    """

    def __init__(self):
        """
        Initialize the MessageAPI with a workspace ID.
        """
        self.generated_ids: set
        self.user_count: int
        self.user_map: Dict[str, str]
        self.inbox: List[Dict[str, str]]
        self.message_count: int
        self.current_user: Optional[str]
        self._api_description = "This tool belongs to the Message API, which is used to manage user interactions in a workspace."

    def _load_scenario(self, scenario: dict, long_context=False) -> None:
        """
        Load a scenario into the MessageAPI.

        Args:
            scenario (Dict): A dictionary containing message data.
        """
        DEFAULT_STATE_COPY = deepcopy(DEFAULT_STATE)
        self._random = random.Random((scenario.get("random_seed", 200191)))
        self.generated_ids = scenario.get(
            "generated_ids", DEFAULT_STATE_COPY["generated_ids"]
        )
        self.user_count = scenario.get("user_count", DEFAULT_STATE_COPY["user_count"])
        self.user_map = scenario.get("user_map", DEFAULT_STATE_COPY["user_map"])
        self.inbox = scenario.get("inbox", DEFAULT_STATE_COPY["inbox"])
        self.message_count = scenario.get(
            "message_count", DEFAULT_STATE_COPY["message_count"]
        )
        self.current_user = scenario.get("current_user", DEFAULT_STATE_COPY["current_user"])

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, MessageAPI):
            return False

        for attr_name in vars(self):
            if attr_name.startswith("_"):
                continue
            model_attr = getattr(self, attr_name)
            ground_truth_attr = getattr(value, attr_name)

            if model_attr != ground_truth_attr:
                return False

        return True

    def _generate_id(self):
        """
        Generate a unique ID for a message.

        Returns:
            new_id (int): A unique ID for a message.
        """
        new_id = self._random.randint(
            10000, 99999
        )  # first 5 mapped by initial configuration.
        while new_id in self.generated_ids:
            new_id = self._random.randint(10000, 99999)
        self.generated_ids.add(new_id)
        return {"new_id": new_id}

    def list_users(self) -> Dict[str, List[str]]:
        """
        List all users in the workspace.

        Returns:
          user_list (List[str]): List of all users in the workspace.
        """
        return {"user_list": list(self.user_map.keys())}

    def get_user_id(self, user: str) -> Dict[str, Optional[str]]:
        """
        Get user ID from user name.

        Args:
            user (str): User name of the user.

        Returns:
            user_id (str): User ID of the user
        """
        if user not in self.user_map:
            return {"error": f"User '{user}' not found in the workspace."}
        return {"user_id": self.user_map.get(user)}

    def message_login(self, user_id: str) -> Dict[str, Union[str, bool]]:
        """
        Log in a user with the given user ID to messeage application.

        Args:
            user_id (str): User ID of the user to log in.

        Returns:
            login_status (bool): True if login was successful, False otherwise.
            message (str): A message describing the result of the login attempt.
        """
        if user_id not in [id for id in self.user_map.values()]:
            return {"login_status": False, "message": f"User ID '{user_id}' not found."}
        self.current_user = user_id
        return {
            "login_status": True,
            "message": f"User '{user_id}' logged in successfully.",
        }

    def message_get_login_status(self) -> Dict[str, bool]:
        """
        Get the login status of the current user.

        Returns:
            login_status (bool): True if the current user is logged in, False otherwise.
        """
        return {"login_status": bool(self.current_user)}

    def send_message(self, receiver_id: str, message: str) -> Dict[str, Union[str, bool]]:
        """
        Send a message to a user.
        Args:
            receiver_id (str): User ID of the user to send the message to.
            message (str): Message to be sent.
        Returns:
            sent_status (bool): True if the message was sent successfully, False otherwise.
            message_id (int): ID of the sent message.
            message (str): A message describing the result of the send attempt.
        """
        # Check if there is a current user logged in
        if not self.current_user:
            return {"error": "No user is currently logged in."}
        # Validate receiver existence
        if receiver_id not in self.user_map.values():
            return {"error": f"Receiver ID '{receiver_id}' not found."}
        # Generate a unique message ID
        message_id = self._generate_id()
        # Store the message in the inbox
        self.inbox.append({receiver_id: message})
        self.message_count += 1
        return {
            "sent_status": True,
            "message_id": message_id,
            "message": f"Message sent to '{receiver_id}' successfully.",
        }

    def delete_message(self, receiver_id: str) -> Dict[str, Union[bool, str]]:
        """
        Delete the latest message sent to a receiver.
        Args:
            receiver_id (str): User ID of the user to send the message to.
        Returns:
            deleted_status (bool): True if the message was deleted successfully, False otherwise.
            receiver_id (str): ID of the receiver of the deleted message.
            message (str): A message describing the result of the deletion attempt.
        """
        if not self.current_user:
            return {"error": "No user is currently logged in."}

        # Loop through the inbox in reverse order to find the first message sent to the receiver
        for message in self.inbox[::-1]:
            receiver, _ = list(message.items())[0]
            if receiver == receiver_id:
                self.inbox.remove(message)
                return {
                    "deleted_status": True,
                    "receiver_id": receiver,
                    "message": f"Receiver {receiver_id}'s latest message deleted successfully.",
                }
        return {"error": f"Receiver ID {receiver_id} not found."}

    def view_messages_sent(self) -> Dict[str, Union[Dict[str, List[str]], str]]:
        """
        View all historical messages sent by the current user.

        Returns:
            messages (Dict): Dictionary of messages grouped by receiver An example of the messages dictionary is {"USR001":["Hello"],"USR002":["World"]}.

        """
        if not self.current_user:
            return {"error": "No user is currently logged in."}
        # Dictionary to collect messages grouped by receiver
        sent_messages = {}
        # Loop through the inbox and collect messages sent by the current user
        for message in self.inbox:
            receiver, message_content = list(message.items())[0]
            if receiver not in sent_messages:
                sent_messages[receiver] = [message_content]
            else:
                sent_messages[receiver].append(message_content)
        return {"messages": sent_messages}

    def add_contact(self, user_name: str) -> Dict[str, Union[bool, str]]:
        """
        Add a contact to the workspace.
        Args:
            user_name (str): User name of contact to be added.
        Returns:
            added_status (bool): True if the contact was added successfully, False otherwise.
            user_id (str): User ID of the added contact.
            message (str): A message describing the result of the addition attempt.
        """
        if user_name in self.user_map:
            return {"error": f"User name '{user_name}' already exists."}
        self.user_count += 1
        user_id = f"USR{str(self.user_count).zfill(3)}"
        if user_id in self.user_map.values():
            return {"error": f"User ID '{user_id}' already exists."}
        self.user_map[user_name] = user_id
        return {
            "added_status": True,
            "user_id": user_id,
            "message": f"Contact '{user_name}' added successfully.",
        }

    def search_messages(
        self, keyword: str
    ) -> Dict[str, Union[List[Dict[str, Union[str, List[str]]]], str]]:
        """
        Search for messages containing a specific keyword.
        Args:
            keyword (str): The keyword to search for in messages.
        Returns:
            results (List[Dict]): List of dictionaries containing matching messages.
                - receiver_id (str): User ID of the receiver of the message.
                - message (str): The message containing the keyword.
        """
        if not self.current_user:
            return {"error": "No user is currently logged in."}
        keyword_lower = keyword.lower()
        results = []
        # Iterate through the inbox to search for the keyword in messages
        # for message_id, message_data in self.inbox.items():
        for message_data in self.inbox:
            receiver_id, message_content = list(message_data.items())[0]
            if keyword_lower in message_content.lower():
                results.append(
                    {
                        "receiver_id": receiver_id,
                        "message": message_content,
                    }
                )
        return {"results": results}

    def get_message_stats(self) -> Dict[str, Union[Dict[str, int], str]]:
        """
        Get statistics about messages for the current user.
        Returns:
            stats (Dict): Dictionary containing message statistics.
                - received_count (int): Number of messages received by the current user.
                - total_contacts (int): Total number of contacts the user has interacted with.
        """
        if not self.current_user:
            return {"error": "No user is currently logged in."}
        sent_count = 0
        received_count = 0
        contacts = set()
        # Loop through the inbox to calculate stats
        for message_data in self.inbox:
            receiver_id, message_content = list(message_data.items())[0]
            received_count += 1
            contacts.add(receiver_id)
        total_contacts = len(contacts)
        return {
            "stats": {
                "received_count": received_count,
                "total_contacts": total_contacts,
            }
        }
