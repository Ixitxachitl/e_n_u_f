# Twitch Chatbot using Markov Chain Model

This README provides information on how to set up and operate the Twitch chatbot which generates text using the Markov Chain model.

## Prerequisites

In order to successfully run the chatbot, the following prerequisites must be met:

- You require a Twitch account.
- You require a Twitch developer application with a Client ID and Client Secret.

## How to run the Twitch Chatbot

To operate this project, follow these steps:

### 1. Clone the repository:

Clone this repository into your local machine.

### 2. Setting up your Twitch developer application credentials:

- Navigate to your Twitch developer dashboard and select your application. Copy the Client ID and Client Secret.

- The next step is to open the `config.ini` file and replace the `APP_ID` and `APP_SECRET` with your Client ID and Client Secret that were just copied.

The `config.ini` file should look as follows:

```ini
[TWITCH_CREDENTIALS]
APP_ID = your_twitch_client_id
APP_SECRET = your_twitch_client_secret
OAUTH_TOKEN = 
REFRESH_TOKEN =
```

### 3. Run the `enuf.py`:

Having set your Twitch developer application credentials, the final step is to run the `enuf.py` program which relies on the `config.ini` file for authentication with the Twitch servers. You can run the following in your Python environment:
```
python enuf.py
```

If no OAuth token and refresh token are found within the config file, the program will initiate the Twitch OAuth 2.0 authentication process which prompts you to log in through your default web browser. Once logged in correctly, the OAuth token and refresh token are stored within the config file.

### 4. Using the chatbot:

By running the file, the program will first connect to the Twitch IRC server before joining the chat rooms which are specified in the `TARGET_CHANNEL` array.

Once a message is sent within a chat room, an instance of the Markov Chatbot class is created for each room in the array. The message text content is then added to the end of the training data which is handled by the bot.

The bot will generate a chat message once a quota of 15 to 25 user messages (random value) have been sent in the chat room. Following this event, this value will reset to another random number between 15 to 25.

To stop the bot from sending messages, simply press ENTER.

Enjoy your newly created Twitch chatbot!

