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
    print('\033[{};0H\033[K'.format(line_num) + text)


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

        # Coordinating conjunctions, cannot start or end a sentence.
        coord_conjunctions = {'for', 'and', 'nor', 'but', 'or', 'yet', 'so'}
        # Prepositions, cannot start or end a sentence.
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
        self.eos_tokens = {'.', '!', '?'}
        # Gather all invalid start words.
        self.invalid_start_words = coord_conjunctions.union(prepositions).union(number_words).union(invalid_start_symbols)
        self.invalid_end_words = coord_conjunctions.union(common_articles).union(pronouns).union(demonstratives).union(
            invalid_end_symbols).union(number_words).union(prepositions)

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

    def format_new_word(self, new_word, generated_words):
        space = " "
        next_word_is_punctuation = new_word in self.eos_tokens or new_word in {",", ".", ":", ";", "(", "[", "\"",
                                                                               "{", "'", "_", "...",
                                                                               ")", "]", "}", "\""}
        previous_word_is_opening_punctuation = generated_words and generated_words[-1] in {"(", "[", "\"", "{", "'",
                                                                                           "_"}

        if next_word_is_punctuation:
            space = ""
        elif previous_word_is_opening_punctuation:
            space = ""
        elif new_word.startswith("'") or new_word.startswith("’"):
            space = ""

        return f"{space}{re.sub(' +', ' ', new_word.strip())}"

    def validate_and_update_end_word(self, generated_words):
        print(f"Initial words: {generated_words}")
        last_word = generated_words[-1].strip()
        print(f"Last word: {last_word}")
        if last_word in self.eos_tokens:  # if last word is End Of Sentence (eos) token
            print(f"EOS detected")
            if len(generated_words) > 4:  # if there are more than 4 words
                state = [generated_words[-4].strip(), generated_words[-3].strip()]  # save the two words before the word immediately preceding the EOS token
            elif len(generated_words) > 3:
                state = ["", generated_words[
                    -4].strip()] # if there are only four words, the state includes an empty string and the word two positions before the EOS token
            else:
                state = ["", ""]  # for all other cases, the state contains two empty strings
            print(f"State before EOS: {state}")
            if generated_words[-2].lower() not in self.invalid_end_words:
                return generated_words  # if the word immediately before the EOS token is valid, no replacement is needed; return the original words
            # if the word immediately before the EOS token is invalid, proceed to next phase to replace it
        else:  # if last word is not End Of Sentence (eos)
            if last_word.lower() not in self.invalid_end_words:  # to check if last word is invalid
                return generated_words
            print(f"Invalid last word: {last_word}")
            if len(generated_words) > 2:
                state = [generated_words[-3].strip(), generated_words[-2].strip()]  # save the state of last two words
            elif len(generated_words) == 2:
                state = ["", generated_words[-2].strip()]  # save the state of last two words
            else:
                state = ["", ""]  # if generated_words have only one word

        print(f"State: {state}")
        # proceed with the rest of the function for replacing the invalid end word

        if len(state) < self.order:
            state = ("",) * (self.order - len(state)) + tuple(state)
        else:
            state = tuple(state[-self.order:])

        print(f"Processed state: {state}")
        valid_end_words = [word for word in self.transitions[state] if word not in self.invalid_end_words]
        valid_end_words = [word for word in valid_end_words if word not in self.eos_tokens]

        print(f"Valid end words for state: {valid_end_words}")

        if valid_end_words:
            new_word = random.choice(valid_end_words)
        else:  # if no valid end word found for present state
            print(f"No valid end words found for present state.")
            for possible_state in self.transitions:
                valid_end_words = [word for word in self.transitions[possible_state] if
                                   word not in self.invalid_end_words]
                valid_end_words = [word for word in valid_end_words if word not in self.eos_tokens]
                if valid_end_words:
                    new_word = random.choice(valid_end_words)
                    print(f"New valid end word found in the transitions for a different state {possible_state}: {new_word}")
                    break
            else:  # if no valid end word found for any state
                print(f"No valid end words found for any state.")
                state = random.choice(list(self.transitions.keys()))  # pick a random state with a valid transition
                new_word = random.choice(
                    list(self.transitions[
                             state]))  # picking another random new word, without checking for eos_token here
                print(f"Randomly selected new word: {new_word}")

        print(f"New word: {new_word}")

        self.format_new_word(new_word, generated_words)

        # replace either the last word with it or if the last word was an eos token replace the one before it.
        if last_word in self.eos_tokens:
            generated_words[-2] = new_word
        else:
            generated_words[-1] = new_word
        print(f"Final words: {generated_words}")
        return generated_words

    def generate(self, input_text, min_length=5, max_length=20):
        """
        This function generates text based on a higher-order Markov transition matrix model. The model leverages the
        transition state to generate the next word and takes certain rules around invalid start and end words into
        consideration.

        The process is defined as follows:

        1. The function initializes with given input text and determines the current state based on input size and the
        order of the Markov model.

        2. It starts the main generation loop. Within this loop, it first calculates the continuation probability based on
        the length of the generated text and the maximum word limit.

        3. The function then decides whether to continue generating words based on the calculated probability.

        4. If the current state does not exist in the transition matrix or all potential next words are invalid, it chooses a new
        random current state.

        5. The next word is chosen based on transition probabilities from the current state. If the next word is potentially
        the last word (due to reaching maximum length or low continuation probability) and it's invalid as an end word or
        it is an end-of-sentence (EOS) token with an invalid prior word, it replaces it with a valid one.

        6. If the generation does not continue, it checks if the last generated word is valid. If not, it reruns the
        generation loop.

        7. If the next word starts a new sentence and it's declared invalid, it chooses a new random state and reruns the
        generation loop.

        8. If the next word is not an EOS token and it's not an invalid end word, it is added into the list of generated words.

        9. The current state is updated for the next word based on the Markov model's order.

        10. Generation is stopped if an EOS token is generated or the maximum length is reached.

        Finally, it returns a joined string of the generated words. The function ensures that it doesn't produce sentences
        starting with coordinating conjunctions, prepositions, and certain symbols. It also confirms that the sentences
        don't end with coordinating conjunctions, common articles, pronouns, demonstratives and other invalid symbols.
        """

        split_input_text = [token.text for token in nlp(input_text)]
        if len(split_input_text) < self.order:
            current_state = ("",) * (self.order - len(split_input_text)) + tuple(split_input_text)
        else:
            current_state = tuple(split_input_text[-self.order:])

        generated_words = []
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
                if len(new_words) == 0:
                    possible_transitions = {word: freq for word, freq in possible_transitions.items() if
                                            word.lower() not in self.invalid_start_words}

                transitions = list(possible_transitions.keys())
                counts = list(possible_transitions.values())
                total_count = sum(counts)
                probabilities = [count / total_count for count in counts]
                next_word = np.random.choice(transitions, p=probabilities)

                print_line(f"Chose transition from '{current_state}' to '{next_word}'", 8)

                self.format_new_word(next_word, new_words)
                new_words.append(next_word)

                current_state = tuple((*current_state[1:], next_word.strip()))

                # Check if the next word is an EOS token,
                # in this case the generation of the sentence should also end,
                # but we still need to ensure that the last word is valid.
                if next_word in self.eos_tokens:
                    stop_reason = 'Hit end-of-sentence token'
                    if len(new_words) <= min_length:
                        stop_reason += ', but sentence was too short so removed EOS token'

                        # Remove the EOS token at the end
                        new_words.pop()
                    new_words = self.validate_and_update_end_word(new_words)
                    break

                if not continue_generation or len(new_words) >= max_length:
                    # Ensure that the last word is valid before ending the sentence generation
                    new_words = self.validate_and_update_end_word(new_words)

                    # Set stop reason
                    if len(new_words) >= max_length:
                        stop_reason = 'Reached maximum length'
                    else:
                        stop_reason = 'Decided not to continue generation after ensuring last word is valid'

                    # Break the loop since the sentence is completed
                    break

            generated_words = new_words
            print_line(f"Reason for stopping: {stop_reason}", 12)
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
