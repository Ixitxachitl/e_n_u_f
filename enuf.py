from twitchAPI.twitch import Twitch
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage
import asyncio
import random
import os
import collections
import configparser
from typing import Dict
import re
from twitchAPI.oauth import UserAuthenticator

# read config.ini file
config_object = configparser.ConfigParser()
config_object.read("config.ini")

credentials = config_object["TWITCH_CREDENTIALS"]

APP_ID = credentials['APP_ID']
APP_SECRET = credentials['APP_SECRET']
OAUTH_TOKEN = credentials['OAUTH_TOKEN']
REFRESH_TOKEN = credentials['REFRESH_TOKEN']
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
TARGET_CHANNEL = ['','']


class MarkovChatbot:
    """
    :class:`MarkovChatbot` is a class that represents a chatbot trained using a Markov chain model. It uses a given order value to determine the number of previous words to consider when
    * generating responses.

    :param room_name: The name of the chat room.
    :type room_name: str
    :param order: The order of the Markov chain model (default: 2).
    :type order: int

    :ivar order: The order of the Markov chain model.
    :ivar transitions: A dictionary that maps a sequence of words to a list of words that tend to follow that sequence.
    :ivar data_file: The file name to load and save the training data.
    :ivar eos_tokens: A set of end-of-sentence tokens used to determine when to stop generating a response.

    :Example:

    .. code-block:: python

        chatbot = MarkovChatbot("chatroom")
        chatbot.append_data("Hello, how are you?")
        chatbot.append_data("I'm fine, thank you.")
        chatbot.generate("Hello")

    .. attribute:: def __init__(self, room_name, order=2)

        Initializes a new instance of the :class:`MarkovChatbot` class.

        :param room_name: The name of the chat room.
        :type room_name: str
        :param order: The order of the Markov chain model (default: 2).
        :type order: int

        .. code-block:: python

            chatbot = MarkovChatbot("chatroom")
            chatbot = MarkovChatbot("chatroom", order=3)

    .. attribute:: def load_and_train(self)

        Loads existing training data from the data file and trains the chatbot.

        .. code-block:: python

            chatbot.load_and_train()

    .. attribute:: def train(self, text)

        Trains the chatbot with the provided text.

        :param text: The text to use for training the chatbot.
        :type text: str

        .. code-block:: python

            chatbot.train("Hello, how are you?")

    .. attribute:: def append_data(self, text)

        Appends new training data to the data file.

        :param text: The new training data to be appended to the data file.
        :type text: str

        .. code-block:: python

            chatbot.append_data("I'm fine, thank you.")
            chatbot.append_data("Nice to meet you!")

    .. attribute:: def generate(self, input_text)

        Generates a response based on the given input text.

        :param input_text: The input text to generate a response for.
        :type input_text: str
        :return: The generated response.
        :rtype: str

        .. code-block:: python

            response = chatbot.generate("Hello, how are you?")
            print(response)

    .. note::
        - The generated response may contain spaces at the beginning due to the Markov chain model.
        - The order of the Markov chain model should be chosen based on the expected length of the input text and the complexity of the language used in the training data.
    """
    def __init__(self, room_name, order=2):
        self.order = order
        self.transitions = collections.defaultdict(list)
        self.data_file = f"{room_name}.txt"
        self.load_and_train()

    def load_and_train(self):
        # Load existing training data from the data file if it exists and train chatbot
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, "r", encoding="utf-8") as file:
                    data = file.read()
                self.train(data)
        except Exception as e:
            print(f"Error loading and training data: {str(e)}")

    def train(self, text):
        # Train the chatbot with the provided text
        # Use regex to split inputs on whitespaces and punctuation.
        # This treats punctuation as separate words
        words = re.findall(r"[\w']+|[.!?]", text)
        for i in range(len(words) - self.order):
            # Split the text into sequences of words, learning what word tends to follow a given sequence
            current_state = tuple(words[i: i + self.order])
            next_word = words[i + self.order]
            self.transitions[current_state].append(next_word)

    def append_data(self, text):
        # Append new training data to the data file
        with open(self.data_file, "a", encoding="utf-8") as append_file:
            append_file.write(text + '\n')

    def generate(self, input_text):
        split_input_text = input_text.split()
        current_order = max(self.order, len(split_input_text))
        current_state = tuple(split_input_text[-current_order:])
        generated_words = []
        eos_tokens = {'.', '!', '?'}
        while not generated_words:
            new_words = []
            next_word = ""
            max_length = random.randint(8, 20)  # This will generate a random number between 8 and 20
            while next_word not in eos_tokens and len(new_words) < max_length:
                if current_state not in self.transitions:
                    current_state = random.choice(list(self.transitions.keys()))
                next_word = random.choice(self.transitions[current_state])
                # If it's the first word and is an eos token, then continue to the next iteration
                if not new_words and next_word in eos_tokens:
                    continue
                # Adds space only if next_word is not an eos_token
                space = "" if next_word in eos_tokens else " "
                new_words.append(space + next_word)
                current_state = tuple((*current_state[1:], next_word))
            generated_words = new_words
        return ''.join(generated_words).lstrip()  # lstrip to remove potential initial space


class ChatBotHandler:
    """
    The `ChatBotHandler` class is responsible for handling incoming messages in a chat room and generating responses using a `MarkovChatbot` instance.

    Attributes:
    - `chatbots` (dict): A dictionary that stores instances of `MarkovChatbot` for each chat room.
    - `message_counter` (dict): A dictionary that keeps track of the number of messages received for each chat room.
    - `target_counter` (dict): A dictionary that stores the randomly set target number of messages for each chat room.

    Methods:
    - `handle_bot_startup(ready_event: EventData)`: Static method that handles the bot startup. It joins the target channel for chat rooms.
    - `handle_incoming_message(msg: ChatMessage)`: Method that handles an incoming message in a chat room. It updates the chatbot's data and training with the message, and generates a response
    * if a certain target is reached.

    Example usage:
    ```python
    chat_handler = ChatBotHandler()

    # handle bot startup
    ready_event = EventData()
    ChatBotHandler.handle_bot_startup(ready_event)

    # handle incoming message
    message = ChatMessage(room=chat_room, user=user, text="Hello")
    ChatBotHandler.handle_incoming_message(message)
    ```
    """
    def __init__(self):
        self.chatbots = {}
        # Initialize the message counter as empty dictionary
        self.message_counter: Dict[str, int] = {}
        # Initialize target counter as empty dictionary
        self.target_counter: Dict[str, int] = {}

    @staticmethod
    async def handle_bot_startup(ready_event: EventData):
        print('Bot is ready for work, joining channels')
        await ready_event.chat.join_room(TARGET_CHANNEL)

    async def handle_incoming_message(self, msg: ChatMessage):
        """
        :param msg: The incoming ChatMessage object representing the received message.
        :return: None

        This method is used to handle an incoming message in a chat room. It prints the message, updates the chatbot's data and training with the message, and generates a response based on the
        * message if a certain target is reached.

        The method takes in two parameters:
        - `self`: The instance of the class containing this method.
        - `msg`: The ChatMessage object representing the received message.

        The method does not return any value.

        Example usage:
            message = ChatMessage(room=chat_room, user=user, text="Hello")
            await handle_incoming_message(self, message)
        """
        print(f'In {msg.room.name}, {msg.user.name}: {msg.text}')

        # create a new instance of MarkovChatbot for this room if it doesn't already exist
        if msg.room.name not in self.chatbots:
            self.chatbots[msg.room.name] = MarkovChatbot(msg.room.name)

        self.chatbots[msg.room.name].append_data(msg.text)
        self.chatbots[msg.room.name].train(msg.text)
        # Increment message counter for the specific room
        self.message_counter[msg.room.name] = self.message_counter.get(msg.room.name, 0) + 1

        # If this is the first message in the room initialize the target counter
        if msg.room.name not in self.target_counter:
            self.target_counter[msg.room.name] = random.randint(15, 25)

        # If the message counter reaches the randomly set target for specific room, generate a response
        if self.message_counter[msg.room.name] == self.target_counter[msg.room.name]:
            response = self.chatbots[msg.room.name].generate(msg.text)
            print(f'Generated in {msg.room.name}: {response}')

            if random.random() <= .05:
                await msg.reply(response)
            else:
                await msg.chat.send_message(msg.room.name, response)

            # Reset the message counter for the specific room
            self.message_counter[msg.room.name] = 0
            # Generate a new random target between 10 and 20 for the next response in this room
            self.target_counter[msg.room.name] = random.randint(15, 25)


async def run(oauth_token='', refresh_token=''):
    """
    :param oauth_token: The OAuth token used for Twitch authentication. Defaults to an empty string.
    :param refresh_token: The refresh token used for Twitch authentication. Defaults to an empty string.
    :return: None

    This method runs the Twitch chat bot. It sets up the necessary authentication, registers event handlers, and starts the chat. It also stops the chat and closes the Twitch connection
    * when the user presses ENTER to stop.

    If the OAuth token and refresh token parameters are not provided, the method will prompt the user to authenticate with Twitch and update the credentials in the config file.

    Example usage:
        await run(oauth_token='your_oauth_token', refresh_token='your_refresh_token')
    """
    handler = ChatBotHandler()

    twitch = await Twitch(APP_ID, APP_SECRET)

    if oauth_token == '' or refresh_token == '':
        auth = UserAuthenticator(twitch, USER_SCOPE, force_verify=True)
        # this will open your default browser and prompt you with the twitch verification website
        oauth_token, refresh_token = await auth.authenticate()
        # add User authentication
        # Update the OAUTH_TOKEN and REFRESH_TOKEN in the config file
        config_object["TWITCH_CREDENTIALS"]["OAUTH_TOKEN"] = oauth_token
        config_object["TWITCH_CREDENTIALS"]["REFRESH_TOKEN"] = refresh_token
        # Write changes back to file
        with open('config.ini', 'w') as conf:
            config_object.write(conf)
    await twitch.set_user_authentication(oauth_token, USER_SCOPE, refresh_token)

    chat = await Chat(twitch)
    chat.register_event(ChatEvent.READY, handler.handle_bot_startup)
    chat.register_event(ChatEvent.MESSAGE, handler.handle_incoming_message)

    chat.start()

    try:
        input('press ENTER to stop\n')
    finally:
        chat.stop()
        await twitch.close()


asyncio.run(run(OAUTH_TOKEN, REFRESH_TOKEN))
