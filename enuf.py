import asyncio
import collections
import configparser
import json
import math
import numpy as np
import random
import re
import spacy
# python -m spacy download en_core_web_sm
from typing import Dict

import nltk
from nltk.corpus import wordnet
# python -m nltk.downloader averaged_perceptron_tagger wordnet
from twitchAPI.chat import Chat, EventData, ChatMessage, ChatCommand
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
from twitchAPI.type import AuthScope, ChatEvent

import sqlite3

# read config.ini file
config_object = configparser.ConfigParser()
config_object.read("config.ini")

credentials = config_object["TWITCH_CREDENTIALS"]
channels_str = config_object.get('CHANNELS', 'target_channels')
target_channels = [channel.strip() for channel in channels_str.split(',') if channel.strip()]

APP_ID = credentials['APP_ID']
APP_SECRET = credentials['APP_SECRET']
OAUTH_TOKEN = credentials['OAUTH_TOKEN']
REFRESH_TOKEN = credentials['REFRESH_TOKEN']
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
TARGET_CHANNEL = target_channels

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
    db_connection = None
    db_cursor = None

    def __init__(self, room_name, order=2):
        self.order = order
        self.transitions = collections.defaultdict(collections.Counter)
        self.room_name = room_name
        # If connection does not exist, create it.
        if MarkovChatbot.db_connection is None or MarkovChatbot.db_cursor is None:
            self.connection = self.create_connection()
            self.cursor = self.connection.cursor()
            MarkovChatbot.db_connection = self.connection  # Set class level connection
            MarkovChatbot.db_cursor = self.cursor  # Set class level cursor
            self.create_table()
        else:
            # If connection already exists, reuse it.
            self.connection = MarkovChatbot.db_connection
            self.cursor = MarkovChatbot.db_cursor
        self.load_and_train()

    def load_and_train(self):
        print_line("Loading and Training...", 5)

        # Get transitions from transitions table for room
        self.cursor.execute("""
            SELECT current_state, next_word, count FROM transitions_table WHERE room_name = ?
        """, (self.room_name,))
        rows = self.cursor.fetchall()

        # Build transitions dictionary from result rows
        for row in rows:
            current_state = tuple(json.loads(row[0]))
            next_word = row[1]
            count = row[2]
            self.transitions[current_state][next_word] += count

        # If we didn't retrieve any transitions, train from data file
        if len(self.transitions) == 0:
            self.train_from_data_file()

        print_line("Loading Completed!", 5)

    def train_from_data_file(self):
        # Get texts from data_file table for room
        self.cursor.execute("""
            SELECT data FROM data_file_table WHERE room_name = ?
        """, (self.room_name,))
        rows = self.cursor.fetchall()

        if rows:
            for row in rows:
                self.train(row[0].strip())
        else:
            print_line(f"No data found for room: {self.room_name}", 5)

    def update_transition_counts(self, current_state, next_word):
        # Prepare values including current_state
        values = (self.room_name, json.dumps(current_state), next_word)
        # Check if record with given state and next word exists
        self.cursor.execute(
            """
            SELECT count, current_state FROM transitions_table 
            WHERE room_name = ? AND current_state = ? AND next_word = ?
            """,
            values
        )
        result = self.cursor.fetchone()
        if result is None:
            self.cursor.execute(
                """
                INSERT INTO transitions_table 
                   (room_name, current_state, next_word, count) 
                VALUES (?, ?, ?, 1)
                """,
                (self.room_name, json.dumps(current_state), next_word)
            )
            self.transitions[current_state][next_word] += 1
        else:
            db_current_state = tuple(json.loads(result[1]))

            if db_current_state == current_state:
                self.cursor.execute(
                    """
                    UPDATE transitions_table 
                    SET count = count + 1
                    WHERE room_name = ? AND current_state = ? AND next_word = ?
                    """,
                    values
                )
                self.transitions[current_state][next_word] += 1

    def train(self, text):
        print_line("Training...", 5)
        text = re.sub(' +', ' ', text)  # add this line to remove consecutive spaces
        words = custom_lemmatizer(nlp(text))

        current_state = ('',) * self.order
        for word in words:
            self.update_transition_counts(current_state, word)
            current_state = current_state[1:] + (word,)

        self.connection.commit()
        print_line("Training Completed!", 5)

    def append_data(self, text):
        print_line("Appending data...", 5)
        text = re.sub(' +', ' ', text)  # add this line to remove consecutive spaces
        # Insert text into data_file_table
        self.cursor.execute("""
            INSERT INTO data_file_table (room_name, data) VALUES (?, ?)
        """, (self.room_name, text))
        self.train(text)
        print_line("Appended data and updated transitions!", 5)

    def generate(self, input_text, min_length=5, max_length=20):
        """
        This function generates a message based on a higher order Markov model.

        Inputs should be tokenized and lemmatized and the output is a sentence generated with such a treatment in mind.

        Steps:
        1. Define invalid start and end words for a sentence.
        2. Tokenize and lemmatize the input text and set the initial state of the Markov model.
            'current_order' is used here, which can accommodate input texts shorter than the maximum order of the model.
        3. Initialize an empty list to hold the generated words.
        4. Loop through:
            - If the current state (consisting of 'current_order' recent words) has no transitions in the model, select
              a new current state randomly from available states.
            - Determine whether to continue generating words based on the length of the generated sentence and a
              probabilistic condition.
            - Choose the next word based on transition probabilities. If the next word could potentially be the last,
              and it is an invalid end word, continue choosing a new next word until a valid end word is chosen.
            - Update the current state, adding the chosen word and discarding the oldest word from it. The length of
              'current_state' always equals to 'current_order'.
            - Break the loop either when an end-of-sentence token is reached, the maximum sentence length is reached, or
              the probabilistic condition to stop generating words is met.
        5. Concatenate the generated words to create the output message, ensuring that the last word is not an invalid
           end word.
        6. Return the generated message.
        """

        # Coordinating conjunctions, cannot start or end a sentence.
        coord_conjunctions = {'for', 'and', 'nor', 'but', 'or', 'yet', 'so'}
        # Prepositions, cannot start a sentence.
        prepositions = {'in', 'at', 'on', 'of', 'to', 'up', 'with', 'over', 'under',
                        'before', 'after', 'between', 'into', 'through', 'during',
                        'without', 'about', 'against', 'among', 'around', 'above',
                        'below', 'along', 'since', 'toward', 'upon'}
        # Other words that are not suitable to end a sentence.
        common_articles = {'the', 'an', 'a'}
        pronouns = {'this', 'these', 'it', 'he', 'she', 'they'}
        demonstratives = {'this', 'that', 'these', 'those'}
        # Words that represent numbers, not suitable to start a sentence.
        number_words = set(map(str, range(10)))
        # Symbols that are invalid as start words.
        invalid_start_symbols = {'/', '\\', '|', '?', '&', '%', '#', '-', '+', '^', '.', ',', ')'}
        # Symbols that are invalid as end words.
        invalid_end_symbols = {'/', '\\', '|', '&', '%', '#', '@', '-', '+', '^', ',', '('}
        # Gather all invalid start words.
        invalid_start_words = coord_conjunctions.union(prepositions).union(number_words).union(invalid_start_symbols)
        # Gather all invalid end words.
        invalid_end_words = coord_conjunctions.union(common_articles).union(pronouns).union(demonstratives).union(
            invalid_end_symbols).union(number_words)

        split_input_text = [token.lemma_ for token in nlp(input_text)]

        # For inputs with less than 'self.order' words, we append empty strings
        # to the start of the current_state tuple.
        if len(split_input_text) < self.order:
            current_state = ("",) * (self.order - len(split_input_text)) + tuple(split_input_text)
        else:
            current_state = tuple(split_input_text[-self.order:])

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

                x = len(new_words)
                continuation_probability = 1 - math.exp((x - max_length) / 5)
                print_line(f"Continuation Probability: {round(continuation_probability * 100)}%", 9)

                continue_generation = random.choices(
                    [True, False], weights=[continuation_probability, 1 - continuation_probability]
                )[0]

                possible_transitions = self.transitions[current_state]
                # Get the keys (transitions) and their corresponding counts
                transitions = list(possible_transitions.keys())
                counts = list(possible_transitions.values())

                # Compute the total count
                total_count = sum(counts)

                # The probabilities are computed as the count of each transition divided by the total count
                probabilities = [count / total_count for count in counts]

                # Choose the next_word based on the computed probabilities
                next_word = np.random.choice(transitions, p=probabilities)

                # Check if next word is potentially the last, and if it's invalid pick another word
                is_last_word = len(new_words) == max_length - 1 or not continue_generation
                while is_last_word and (next_word in invalid_end_words or len(new_words) == 0):
                    while all(word in invalid_end_words for word in possible_transitions.keys()):
                        print_line(f"All possible transitions were invalid, chose a new current state: {current_state}",
                                   10)
                        current_state = random.choice(list(self.transitions.keys()))
                        possible_transitions = self.transitions[current_state]
                    transitions = list(possible_transitions.keys())
                    counts = list(possible_transitions.values())

                    # Compute the total count
                    total_count = sum(counts)

                    # The probabilities are computed as the count of each transition divided by the total count
                    probabilities = [count / total_count for count in counts]

                    # Choose the next_word based on the computed probabilities
                    next_word = np.random.choice(transitions, p=probabilities)

                if not continue_generation:
                    if len(new_words) == 0 or new_words[-1] in invalid_end_words:
                        stop_reason = "The chosen end word is invalid, re-choosing."
                        continue
                    stop_reason = "Decided not to continue generation"
                    break

                print_line(f"Chose transition from '{current_state}' to '{next_word}'", 8)

                if len(new_words) == 0 and (next_word in invalid_start_words or next_word.startswith("'")):
                    print_line(f"The chosen word '{next_word}' is an invalid start word, restarting selection.", 10)
                    current_state = random.choice(list(self.transitions.keys()))
                    print_line(f"Chose a new current state: {current_state}", 7)
                    continue

                # Default behavior is to add a space
                space = " "

                next_word_is_punctuation = next_word in eos_tokens or next_word in {",", ".", ":", ";", "(", "[", "\"",
                                                                                    "{", "'", "_",
                                                                                    ")", "]", "}", "\""}

                previous_word_is_opening_punctuation = new_words and new_words[-1] in {"(", "[", "\"", "{", "'", "_"}

                # Don't add a space if next word is punctuation
                if next_word_is_punctuation:
                    space = ""
                # Don't add a space if the previous word is an opening punctuation.
                elif previous_word_is_opening_punctuation:
                    space = ""
                # Don't add a space if the next word starts with an apostrophe
                elif next_word.startswith("'") or next_word.startswith("’"):
                    space = ""
                # Only add the next word if it is not an eos token or if min length has been reached
                # And ensuring that the last word is not an invalid end word
                if (not (next_word in eos_tokens and (
                        len(new_words) < min_length or (len(new_words) > 1 and new_words[-2] in invalid_end_words)))
                        and next_word not in invalid_end_words):
                    next_word = f"{space}{re.sub(' +', ' ', next_word.strip())}"
                    new_words.append(next_word)  # add next word

                current_state = tuple((*current_state[1:], next_word))

                if next_word in eos_tokens:
                    stop_reason = "Hit end-of-sentence token"
                    break

                if len(new_words) >= max_length:
                    stop_reason = "Reached maximum length"
                    break

            print_line(f"Reason for stopping: {stop_reason}", 12)

            generated_words = new_words

        generated_message = ''.join(generated_words).lstrip()
        print_line(f"Final message: '{generated_message}'", 11)
        return generated_message

    @staticmethod
    def create_connection():
        conn = None
        try:
            # This will create a new database if it doesn't exist
            conn = sqlite3.connect('chatbot_db.sqlite')
            print(f'Successful connection with sqlite version {sqlite3.version}')

            return conn
        except Exception as e:
            print(f'The error {e} occurred')

        return conn

    def create_table(self):
        # Create tables in the database if they do not exist
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_file_table (
                id INTEGER PRIMARY KEY,
                room_name TEXT NOT NULL,
                data TEXT NOT NULL
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS transitions_table (
                id INTEGER PRIMARY KEY,
                room_name TEXT NOT NULL,
                current_state TEXT,
                next_word TEXT,
                count INT
            )
        """)
        # Commit the transaction
        self.connection.commit()


class ChatBotHandler:
    def __init__(self):
        self.name = ''
        self.chatbots = {}
        # Initialize the message counter as empty dictionary
        self.message_counter: Dict[str, int] = {}
        # Initialize ignore_users list
        self.ignore_users = ['streamelements', 'streamlabs', 'nightbot', 'soundalerts', 'buttsbot', 'sery_bot',
                             'pokemoncommunitygame', 'elbierro', 'streamlootsbot', 'kofistreambot']

    async def join_room(self, cmd: ChatCommand):
        if (cmd.room.name == cmd.chat.username and cmd.user.name not in target_channels
                and cmd.user.name != cmd.room.name):
            await cmd.chat.join_room(cmd.user.name)
            print_line(f'{self.name} joined {cmd.user.name}', 0)
            await cmd.reply(f'Joined {cmd.user.name}')

            # add the new channel to the target_channels list
            target_channels.append(cmd.user.name)

            # update it in the config_object
            config_object.set('CHANNELS', 'target_channels', ', '.join(target_channels))

            # write the changes back to the config file
            with open('config.ini', 'w') as f:
                config_object.write(f)

    async def leave_room(self, cmd: ChatCommand):
        if cmd.room.name == cmd.chat.username and cmd.user.name in target_channels and cmd.user.name != cmd.room.name:
            await cmd.chat.leave_room(cmd.user.name)
            print_line(f'{self.name} left {cmd.user.name}', 0)
            await cmd.reply(f'Left {cmd.user.name}')

            # remove the channel from the target_channels list
            target_channels.remove(cmd.user.name)

            # update it in the config_object
            config_object.set('CHANNELS', 'target_channels', ', '.join(target_channels))

            # write the changes back to the config file
            with open('config.ini', 'w') as f:
                config_object.write(f)

    async def handle_bot_startup(self, ready_event: EventData):
        self.name = ready_event.chat.username
        await ready_event.chat.join_room(self.name)
        print_line(f'{self.name} joined it\'s own channel', 0)
        if TARGET_CHANNEL:
            print_line(f'{self.name} is ready for work, joining channel(s) {TARGET_CHANNEL} ', 0)
            await ready_event.chat.join_room(TARGET_CHANNEL)

    async def handle_incoming_message(self, msg: ChatMessage, max_messages=35):
        if msg.room.name == msg.chat.username:
            return
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
        print_line(f'Respond probability in {msg.room.name}: {round(respond_probability * 100)}%', 2)

        # Generate a response if random value is less than respond probability
        random_val = random.random()
        print_line(f'Rolled: {round(random_val * 100)}', 3)

        if random_val < respond_probability:
            response = self.chatbots[msg.room.name].generate(msg.text)
            # Reset message counter after a response is generated
            self.message_counter[msg.room.name] = 0
            print_line(f'Generated in {msg.room.name}: {response}', 4)
            await asyncio.sleep(random.randint(1, 5))
            if random.random() < 0.05:
                await msg.reply(response)
            else:
                await msg.chat.send_message(msg.room.name, response)


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
    chat.register_command('join', handler.join_room)
    chat.register_command('leave', handler.leave_room)

    chat.start()

    try:
        input()
    finally:
        chat.stop()
        await twitch.close()


if __name__ == "__main__":
    asyncio.run(run(OAUTH_TOKEN, REFRESH_TOKEN))
