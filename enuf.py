import asyncio
import collections
import configparser
import os
import random
import re
from typing import Dict

from nltk import bigrams
from nltk.stem import WordNetLemmatizer
from twitchAPI.chat import Chat, EventData, ChatMessage
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
from twitchAPI.type import AuthScope, ChatEvent


# read config.ini file
config_object = configparser.ConfigParser()
config_object.read("config.ini")

credentials = config_object["TWITCH_CREDENTIALS"]

APP_ID = credentials['APP_ID']
APP_SECRET = credentials['APP_SECRET']
OAUTH_TOKEN = credentials['OAUTH_TOKEN']
REFRESH_TOKEN = credentials['REFRESH_TOKEN']
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
TARGET_CHANNEL = ['']


class MarkovChatbot:
    def __init__(self, room_name, order=3):
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
        lemmatizer = WordNetLemmatizer()

        # Train the chatbot with the provided text
        # Use regex to split inputs on whitespaces and punctuation.
        # This treats punctuation as separate words
        words = re.findall(r"[\w']+|[.!?]", text.lower())

        # Lemmatize words
        words = [lemmatizer.lemmatize(word) for word in words]

        words_bigrams = list(bigrams(words))

        for i in range(len(words_bigrams)):
            # Check for newline in the sequence and the next word
            if '\n' in words_bigrams[i][0] or '\n' in words_bigrams[i][1]:
                continue  # If found, skip the current index
            # Split the text into sequences of words, learning what word tends to follow a given sequence
            current_state = tuple(words_bigrams[i])
            next_word = words[i + 2] if i + 2 < len(words) else ''
            self.transitions[current_state].append(next_word)

    def append_data(self, text):
        # Append new training data to the data file
        with open(self.data_file, "a", encoding="utf-8") as append_file:
            append_file.write(text + '\n')

    def generate(self, input_text):
        lemmatizer = WordNetLemmatizer()

        # Preprocess input text
        split_input_text = [lemmatizer.lemmatize(word.lower()) for word in input_text.split()]

        current_order = min(self.order, len(split_input_text))
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
                if not new_words and next_word in {'.', '?'}:
                    continue
                # Adds space only if next_word is not an eos_token
                space = "" if next_word in eos_tokens else " "
                new_words.append(space + next_word)
                current_state = tuple((*current_state[1:], next_word))

            generated_words = new_words

        generated_message = ''.join(generated_words).lstrip()  # remove potential initial space

        # remove the '.' token from the end of the generated message if it's there
        if generated_message.endswith('.'):
            generated_message = generated_message[:-1]

        return generated_message


class ChatBotHandler:
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
