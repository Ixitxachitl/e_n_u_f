Twitch Chatbot with Markov Chain Model
This chatbot is designed for Twitch's chat platform and uses the Markov Chain model to generate text.
Prerequisites
Before you begin, ensure you have met the following requirements:
You have a Twitch account.
You have a Twitch developer application with a Client ID and Client Secret.
Running Twitch Chatbot with Markov Chain Model
To run this project, you need to follow the steps below:
Clone this repository: Clone this repo to your local machine.
Set up your Twitch developer application credentials
Go to your Twitch developer dashboard and click on your application. Copy the Client ID and Client Secret.
Open config.ini, replace APP_ID, and APP_SECRET with your Twitch application Client ID and Client Secret, respectively.
Your config.ini should look like this:
[TWITCH_CREDENTIALS] APP_ID = your_twitch_client_id APP_SECRET = your_twitch_client_secret OAUTH_TOKEN = REFRESH_TOKEN =
Run enuf.py
You can then run the enuf.py also provided in this repo which uses the config file to authenticate with Twitch's servers.
The app triggers the Twitch OAuth2 authentication process if it does not find the OAuth token and refresh token in the config file. It prompts you to login on your default web browser and after successful login, it stores the OAuth token and refresh token in the config file.
Chat in the Target Channels
The app connects to the Twitch IRC server and joins the target chat rooms defined in the TARGET_CHANNEL list.
Use the Bot
Every time a message is sent in a chat room, an instance of a Markov Chatbot for the chat room is created, and the text of the sent message is appended as new training data for the chatbot.
Deciding When the Bot Talks
The bot by default generates a chat message after every 15 to 25 user messages in a chat room. After sending a message, this target is reset with another random number between 15 and 25.
You can stop the bot at any point by pressing ENTER.
Enjoy using the Twitch Chatbot with Markov Chain Model!