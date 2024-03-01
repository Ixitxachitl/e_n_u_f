import asyncio
import collections
import configparser
import math
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
    def __init__(self, room_name, order=2):
        self.order = order
        self.transitions = collections.defaultdict(list)
        self.data_file = f"{room_name}.txt"
        self.load_and_train()

    def load_and_train(self):
        # Load existing training data from the data file if it exists and train chatbot
        print("Loading and Training...")
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
        print("Training...")
        words = re.findall(r"[\w']+|[.!?]", text)
        words = [lemmatizer.lemmatize(word) for word in words]
        words_bigrams = list(bigrams(words))
        for i in range(len(words_bigrams)):
            if '\n' in words_bigrams[i][0] or '\n' in words_bigrams[i][1]:
                continue
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
        lemmatizer = WordNetLemmatizer()
        print("Generating response...")
        split_input_text = [lemmatizer.lemmatize(word.lower()) for word in input_text.split()]
        current_order = min(self.order, len(split_input_text))
        current_state = tuple(split_input_text[-current_order:])
        generated_words = []
        eos_tokens = {'.', '?'}
        while not generated_words:
            new_words = []
            while True:
                if current_state not in self.transitions:
                    print(f"Current state '{current_state}' not in transitions. Selecting a random state.")
                    current_state = random.choice(list(self.transitions.keys()))
                # Get transitions for the current state
                possible_transitions = self.transitions[current_state]

                # Define the decay scale factor relative to max_length
                scale_factor = len(new_words) / max_length
                # Calculate exponential decay for whether any next word is chosen at all, scaled to max_length
                if len(new_words) > 0:  # apply decay only after the first word
                    continuation_probability = math.exp(-0.5 * scale_factor)
                else:
                    continuation_probability = 1.0  # don't apply decay for the first word
                print(f"Continuation probability: {continuation_probability}")

                # Make the decision whether to continue generating words or not
                continue_generation = \
                    random.choices([True, False], weights=[continuation_probability, 1 - continuation_probability])[0]
                if not continue_generation:
                    break  # stop generating words
                # The probability of picking any individual word is proportional to its frequency (i.e., the standard behavior of random.choices when the 'weights' parameter is not specified)
                next_word = random.choice(possible_transitions)
                print(f"Adding next word '{next_word}' to the new words.")

                if not new_words and next_word in eos_tokens:
                    continue
                space = "" if next_word in eos_tokens else " "
                next_word = re.sub(' +', ' ', next_word)
                new_words.append(space + next_word.strip())
                current_state = tuple((*current_state[1:], next_word))
                # Break the loop if we hit an end-of-sentence token
                if next_word in eos_tokens:
                    break

            generated_words = new_words
            print(f"Generated words '{generated_words}'.")
        generated_message = ''.join(generated_words).lstrip()
        print(f"Final message: '{generated_message}'")
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

    async def handle_incoming_message(self, msg: ChatMessage, max_messages=25):
        print(f'In {msg.room.name}, {msg.user.name}: {msg.text}')
        # create a new instance of MarkovChatbot for this room if it doesn't already exist
        if msg.room.name not in self.chatbots:
            self.chatbots[msg.room.name] = MarkovChatbot(msg.room.name)
        self.chatbots[msg.room.name].append_text(msg.text)
        self.chatbots[msg.room.name].train(msg.text)
        # Increment message counter for the specific room
        self.message_counter[msg.room.name] = self.message_counter.get(msg.room.name, 0) + 1
        # Calculate respond probability
        exponent = -self.message_counter[msg.room.name] / max_messages
        respond_probability = 1 - (1 - 1 / math.e) ** exponent  # added exponential growth
        print(f'Respond probability in {msg.room.name}: {respond_probability}')  # added print statement
        # Generate a response if random value is less than respond probability
        random_val = random.random()
        print(f'Random value: {random_val}')  # added print statement
        if random_val < respond_probability:
            response = self.chatbots[msg.room.name].generate(msg.text)
            print(f'Generated in {msg.room.name}: {response}')
            if random.random() < 0.05:
                await msg.reply(response)
            else:
                await msg.chat.send_message(msg.room.name, response)
            # Reset the message counter for the specific room
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
