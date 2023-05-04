# bot.py
import os
import json
from datetime import datetime, timedelta
import re

import discord
import socketio


class CharlieClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super(CharlieClient, self).__init__(*args, **kwargs)

        self.message_cooldown = timedelta(seconds=60)
        self.last_message_time = datetime.now() - self.message_cooldown

        self.sio: socketio.AsyncClient = socketio.AsyncClient()
        self.callbacks()

    def callbacks(self):
        @self.sio.event
        async def connect():
            print("Socketio connected!")
            await self.sio.emit(
                "initcharlie",
                (
                    "000000000000000000000",
                    "sessionKeyLocal",
                    True,
                    {
                        "userUID": "000000000000000000000",
                        "name": "John",
                        "gender": "female",
                        "gender-user": "male",
                        "language": "EN-US",
                        "memory_size": 3,
                        "style_en": "sassy girl that matches the tone, vocabulary, details and length of the person she responds to",
                        "situation_en": "chatting in a discord server with a few nerds",
                        "tts-method": "notts",
                    },
                ),
            )

        @self.sio.event
        async def disconnect():
            print("Socketio disconnected!")

        @self.sio.on("charliesessioninit")
        async def session_init(data):
            print("session init with data", data)
            self.session_key_local = data["key"]
            self.session_token = data["session_token"]

        @self.sio.on("logging")
        async def logging(data):
            print("Logging:", data["message"])

        @self.sio.on("convmsg")
        async def post_msg(data):
            print(
                "Post msg with data",
                data,
                " in channel ",
                self.target_channel,
                " and user ",
                self.target_user,
            )
            if self.target_channel is None or self.target_user is None:
                return
            conv_message = data["message"]
            if self.target_user in conv_message or "SYSTEM" in conv_message:
                return

            conv_message_content_match = re.search("\[.*?\]\[.*?\](.*)", conv_message)
            if conv_message_content_match is not None:
                await self.target_channel.send(conv_message_content_match.group(1))
                self.is_responsive = True

    async def async_connect_sio(self):
        print("Connect client")

        self.target_channel = None
        self.target_user = None
        await self.sio.connect("http://127.0.0.1:5000")

    async def on_ready(self):
        print(f"{client.user} has connected to Discord!")
        await self.async_connect_sio()

    async def on_message(self, message):
        if message.author == client.user:
            return

        if (
            datetime.now() - self.last_message_time < self.message_cooldown
            or not self.is_responsive
        ):
            return
        self.last_message_time = datetime.now()

        channel_name = message.channel.name

        if channel_name != "charlie-test" and channel_name != "chat-with-charlie":
            return

        self.target_user = message.author.name
        self.target_channel = message.channel
        print(f'Send content "{message.content}" to server')
        await self.sio.emit(
            "sendmessage", (self.session_token, message.content, self.target_user)
        )
        self.is_responsive = False


token = json.load(
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.json"))
)["token"]

intents = discord.Intents(messages=True, guilds=True, message_content=True)

client = CharlieClient(intents=intents)
client.run(token)
