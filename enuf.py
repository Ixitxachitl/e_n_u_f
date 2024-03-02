import asyncio
import collections
import configparser
import math
import os
import random
import re
from typing import Dict

import nltk
from nltk import bigrams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
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


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()  # Get the first letter of the POS tag
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)  # Return the corresponding WordNet POS tag


class MarkovChatbot:
    def __init__(self, room_name, order=2):
        self.order = order
        self.transitions = collections.defaultdict(list)
        self.data_file = f"{room_name}.txt"
        self.lemmatizer = WordNetLemmatizer()  # Instantiate the lemmatizer here
        self.load_and_train()

    def load_and_train(self):
        # Load existing training data from the data file if it exists and train chatbot
        print("Loading and Training...")
        if os.path.exists(self.data_file):
            with open(self.data_file, "r", encoding="utf-8") as file:
                data = file.readlines()
            for line in data:
                self.train(line.strip())
        else:
            print(f"No data file found at location: {self.data_file}")
        print("Training Completed!")

    def train(self, text):
        # Train the chatbot with the provided text
        print("Training...")
        words = re.findall(r"[\w']+|[.!?]", text)  # Updated line
        words = [self.lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
        words_bigrams = list(bigrams(words))
        for i in range(len(words_bigrams)):
            current_state = tuple(words_bigrams[i])
            if i + 2 < len(words):
                next_word = words[i + 2]
                self.transitions[current_state].append(next_word)

    def append_data(self, text):
        # Append new training data to the data file
        print("Appending data...")
        with open(self.data_file, "a", encoding="utf-8") as append_file:
            append_file.write(text + '\n')

    def generate(self, input_text, max_length=20):
        print("Generating response...")
        split_input_text = [self.lemmatizer.lemmatize(word.lower()) for word in input_text.split()]
        current_order = min(self.order, len(split_input_text))
        current_state = tuple(split_input_text[-current_order:])
        generated_words = []
        eos_tokens = {'.', '!', '?'}

        while not generated_words:
            new_words = []
            while True:
                if current_state not in self.transitions or not self.transitions[current_state]:
                    print(
                        f"Current state '{current_state}' not in transitions or has no valid transitions. Selecting a random state.")
                    current_state = random.choice(list(self.transitions.keys()))

                possible_transitions = self.transitions[current_state]
                scale_factor = len(new_words) / max_length
                if len(new_words) > 0:
                    continuation_probability = math.exp(-0.1 * scale_factor)
                else:
                    continuation_probability = 1.0
                print(f"Continuation probability: {continuation_probability}")
                continue_generation = \
                    random.choices([True, False], weights=[continuation_probability, 1 - continuation_probability])[0]
                if not continue_generation:
                    stop_reason = "Decided not to continue generation"
                    break

                # Keep generating a word until it's not an eos_token if it's the first word
                while True:
                    next_word = random.choice(possible_transitions)
                    # if all possible transitions are eos tokens
                    if all(word in eos_tokens for word in possible_transitions):
                        break
                    # if it's not the first word, or it's not an eos token
                    if new_words or next_word not in eos_tokens:
                        break

                print(f"Adding next word '{next_word}' to the new words.")

                space = "" if next_word in eos_tokens else " "
                next_word = re.sub(' +', ' ', next_word)
                new_words.append(space + next_word.strip())
                current_state = tuple((*current_state[1:], next_word))
                if new_words and next_word in eos_tokens:
                    stop_reason = "Hit end-of-sentence token"
                    break
            generated_words = new_words
            print(f"Generated words '{generated_words}'.")
        generated_message = ''.join(generated_words).lstrip()
        print(f"Final message: '{generated_message}'")
        print(f"Reason for stopping: {stop_reason}")
        return generated_message


class ChatBotHandler:
    def __init__(self):
        self.chatbots = {}
        # Initialize the message counter as empty dictionary
        self.message_counter: Dict[str, int] = {}

    @staticmethod
    async def handle_bot_startup(ready_event: EventData):
        print('Bot is ready for work, joining channels')
        await ready_event.chat.join_room(TARGET_CHANNEL)

    async def handle_incoming_message(self, msg: ChatMessage, max_messages=55):
        print(f'In {msg.room.name}, {msg.user.name}: {msg.text}')
        # Create a new instance of MarkovChatbot for this room if it doesn't already exist
        if msg.room.name not in self.chatbots:
            self.chatbots[msg.room.name] = MarkovChatbot(msg.room.name)
        self.chatbots[msg.room.name].append_data(msg.text)
        self.chatbots[msg.room.name].train(msg.text)
        # Increment message counter for the specific room
        self.message_counter[msg.room.name] = self.message_counter.get(msg.room.name, 0) + 1

        # Calculate respond probability
        x = (self.message_counter[msg.room.name] - 1) / max_messages
        respond_probability = max(0, math.pow(1.2, x) - 1)  # using max to avoid negative probability at x=0
        print(f'Respond probability in {msg.room.name}: {respond_probability}')

        # Generate a response if random value is less than respond probability
        random_val = random.random()
        print(f'Random value: {random_val}')

        if random_val < respond_probability:
            response = self.chatbots[msg.room.name].generate(msg.text)
            print(f'Generated in {msg.room.name}: {response}')
            if random.random() < 0.05:
                await msg.reply(response)
            else:
                await msg.chat.send_message(msg.room.name, response)
            # Reset message counter after a response is sent
            self.message_counter[msg.room.name] = 0


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


if __name__ == "__main__":
    asyncio.run(run(OAUTH_TOKEN, REFRESH_TOKEN))
