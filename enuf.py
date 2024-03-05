import asyncio
import collections
import configparser
import math
import numpy as np
import os
import pickle
import random
import re
import spacy
# python -m spacy download en_core_web_sm
from typing import Dict

import nltk
from nltk.corpus import wordnet
# python -m nltk.downloader averaged_perceptron_tagger wordnet
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
TARGET_CHANNEL = ['drkrdnk']

nlp = spacy.load('en_core_web_sm')  # spacy's English model
nlp_dict = set(w.lower_ for w in nlp.vocab)

def print_line(text, line_num):
    print('\033[{};0H'.format(line_num) + ' ' * 200)  # clear the line by writing 50 spaces
    print('\033[{};0H'.format(line_num) + text)  # write your text at the start of the cleared line


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()  # Get the first letter of the POS tag
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)  # Return the corresponding WordNet POS tag


def custom_lemmatizer(nlp_doc):
    lemmas = []
    forms_of_be = {"be", "is", "am", "are", "was", "were", "been", "being"}  # Add all forms of 'be' here
    for token in nlp_doc:
        if token.pos_ != "NOUN" and token.lemma_ not in forms_of_be:  # Only lemmatize non-nouns that are not 'be'
            # Only lemmatize if word exists in the dictionary
            lemmas.append(token.lemma_ if token.text.lower() in nlp_dict else token.text)
        else:
            lemmas.append(token.text)
    return lemmas


class MarkovChatbot:
    def __init__(self, room_name, order=2):
        self.order = order
        self.transitions = collections.defaultdict(collections.Counter)
        self.data_file = f"{room_name}.txt"
        self.pickle_file = f"{room_name}.pickle"
        self.load_and_train()

    def load_and_train(self):
        print_line("Loading and Training...", 5)
        if os.path.exists(self.pickle_file):
            with open(self.pickle_file, "rb") as file:
                self.transitions = pickle.load(file)
        else:
            self.train_from_data_file()
        print_line("Loading Completed!", 5)

    def train_from_data_file(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, "r", encoding="utf-8") as file:
                data = file.readlines()
            for line in data:
                self.train(line.strip())
        else:
            print_line(f"No data file found at location: {self.data_file}", 5)

    def train(self, text):
        print_line("Training...", 5)
        # Regular expression to match words that may include an apostrophe or a punctuation character.
        # \b: word boundary.
        # \w: a word character (equivalent to [a-zA-Z0-9_]).
        # [\w']*: any sequence of word characters and/or apostrophes.
        # \b\w[\w']*\b: a whole word that may include an apostrophe.
        # | : OR operator.
        # [.!?]: a punctuation character.
        # PUTTING IT ALL TOGETHER: a word (which may contain an apostrophe) OR a punctuation character.
        words = re.findall(r"\b\w[\w']*\b|[.!?]", text)
        words = custom_lemmatizer(nlp(' '.join(words)))  # Call to custom lemmatizer

        for i in range(len(words) - self.order):
            current_state = tuple(words[i: i + self.order])
            next_word = words[i + self.order]
            self.transitions[current_state][next_word] = self.transitions[current_state].get(next_word, 0) + 1
        print_line("Training Completed!", 5)

    def append_data(self, text):
        print_line("Appending data...", 5)
        with open(self.data_file, "a", encoding="utf-8") as append_file:
            append_file.write(text + '\n')
        self.train(text)
        with open(self.pickle_file, "wb") as pickleFile:
            pickle.dump(self.transitions, pickleFile)

    def generate(self, input_text, min_length=5, max_length=20):
        coord_conjunctions = {'for', 'and', 'nor', 'but', 'or', 'yet', 'so'}
        prepositions = {'in', 'at', 'on', 'of', 'to', 'up', 'with', 'over', 'under',
                        'before', 'after', 'between', 'into', 'through', 'during',
                        'without', 'about', 'against', 'among', 'around', 'above',
                        'below', 'along', 'since', 'toward', 'upon'}
        number_words = set(map(str, range(10)))
        invalid_start_words = coord_conjunctions.union(prepositions).union(number_words)

        split_input_text = [token.lemma_ for token in nlp(input_text)]
        current_order = min(self.order, len(split_input_text))
        current_state = tuple(split_input_text[-current_order:])
        generated_words = []
        eos_tokens = {'.', '!', '?'}
        stop_reason = ''

        while not generated_words:
            new_words = []

            while True:
                if current_state not in self.transitions or not self.transitions[current_state]:
                    print_line(f"No transitions for {current_state}", 6)
                    current_state = random.choice(list(self.transitions.keys()))
                    print_line(f"Chose a new current state: {current_state}", 7)

                possible_transitions = self.transitions[current_state]

                x = len(new_words)
                continuation_probability = 1 - math.exp((x - max_length) / 5)
                print_line(f"Continuation Probability: {round(continuation_probability*100)}%", 9)
                continue_generation = random.choices(
                    [True, False], weights=[continuation_probability, 1 - continuation_probability]
                )[0]

                if not continue_generation:
                    stop_reason = "Decided not to continue generation"
                    break

                if all(word in invalid_start_words or word in eos_tokens for word in possible_transitions.keys()):
                    current_state = random.choice(list(self.transitions.keys()))
                    print_line(f"All possible transitions were invalid, chose a new current state: {current_state}", 10)
                    continue

                next_word = np.random.choice(list(possible_transitions.keys()),
                                             p=[freq / sum(possible_transitions.values()) for freq in
                                                possible_transitions.values()])

                print_line(f"Chose transition from '{current_state}' to '{next_word}'", 8)

                if len(new_words) == 0 and next_word in invalid_start_words:
                    print_line(f"The chosen word '{next_word}' is an invalid start word, restarting selection.", 10)
                    continue

                space = "" if (next_word in eos_tokens or next_word.startswith("'")) else " "

                # Only add the next word if it is not an eos token or if min length has been reached
                if not (next_word in eos_tokens and len(new_words) < min_length):
                    new_words.append(f"{space}{re.sub(' +', ' ', next_word.strip())}")

                current_state = tuple((*current_state[1:], next_word))

                if next_word in eos_tokens:
                    stop_reason = "Hit end-of-sentence token"
                    break

                if len(new_words) >= max_length:
                    stop_reason = "Reached maximum length"
                    break

            generated_words = new_words

        generated_message = ''.join(generated_words).lstrip()
        print_line(f"Final message: '{generated_message}'", 11)
        print_line(f"Reason for stopping: {stop_reason}", 12)
        return generated_message


class ChatBotHandler:
    def __init__(self):
        self.chatbots = {}
        # Initialize the message counter as empty dictionary
        self.message_counter: Dict[str, int] = {}
        # Initialize ignore_users list
        self.ignore_users = ['streamelements', 'nightbot', 'soundalerts','buttsbot','sery_bot',
                             'pokemoncommunitygame','elbierro']

    @staticmethod
    async def handle_bot_startup(ready_event: EventData):
        print_line(f'Bot is ready for work, joining channel(s) {TARGET_CHANNEL} ', 0)
        await ready_event.chat.join_room(TARGET_CHANNEL)

    async def handle_incoming_message(self, msg: ChatMessage, max_messages=35):
        if msg.user.name in self.ignore_users:
            return
        print_line(f'In {msg.room.name}, {msg.user.name}: {msg.text}', 1)
        # Create a new instance of MarkovChatbot for this room if it doesn't already exist
        if msg.room.name not in self.chatbots:
            self.chatbots[msg.room.name] = MarkovChatbot(msg.room.name)
        self.chatbots[msg.room.name].append_data(msg.text)
        # Increment message counter for the specific room
        self.message_counter[msg.room.name] = self.message_counter.get(msg.room.name, 0) + 1

        # Calculate respond probability
        x = self.message_counter[msg.room.name]
        respond_probability = np.exp((x - max_messages) / 4)
        print_line(f'Respond probability in {msg.room.name}: {round(respond_probability*100)}%', 2)

        # Generate a response if random value is less than respond probability
        random_val = random.random()
        print_line(f'Rolled: {round(random_val*100)}', 3)

        if random_val < respond_probability:
            response = self.chatbots[msg.room.name].generate(msg.text)
            print_line(f'Generated in {msg.room.name}: {response}', 4)
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
        input()
    finally:
        chat.stop()
        await twitch.close()


if __name__ == "__main__":
    asyncio.run(run(OAUTH_TOKEN, REFRESH_TOKEN))
