# bot.py
import os
import json
from datetime import datetime, timedelta
import re
import hashlib

import discord
import socketio


class CharlieBackendSession:
    def __init__(self, guild_id: int):
        self.guild_id = guild_id
        self.uid = self._generate_uid(self.guild_id)

        self.target_channel = None
        self.target_user = None

        self.message_cooldown = timedelta(seconds=10)
        self.last_message_time = datetime.now() - self.message_cooldown

        self.sio: socketio.AsyncClient = socketio.AsyncClient()
        self.callbacks()
        self.initialized = False

        self.is_responsive = True

    def _generate_uid(self, guild_id: int) -> str:
        uid = "dcb"
        h = hashlib.new("sha256")
        h.update(str(guild_id).encode())
        uid += h.hexdigest()
        return uid[:21]

    def callbacks(self):
        @self.sio.event
        async def connect():
            print("Socketio connected!")
            if not self.initialized:
                await self.sio.emit(
                    "initcharlie",
                    (
                        self.uid,
                        self.uid + "sessionKeyLocal",
                        True,
                        {
                            "userUID": self.uid,
                            "name": "John",
                            "gender": "female",
                            "gender-user": "male",
                            "language": "EN-US",
                            "memory_size": 3,
                            "style_en": "nice girl that is pleasant to talk to",
                            "situation_en": "chatting in a discord server with a few friends",
                            "tts-method": "notts",
                        },
                    ),
                )
                self.initialized = True

        @self.sio.event
        async def disconnect():
            print("Socketio disconnected!")

        @self.sio.on("charliesessioninit")
        async def session_init(data):
            print("session init with data", data)
            self.session_key_local = data["key"]
            self.session_token = data["session_token"]

        @self.sio.on("logging")
        async def logging(data: dict):
            print("Logging:", data["message"])

        @self.sio.on("convmsg")
        async def post_msg(data: dict):
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

    async def dispatch_message(
        self,
        msg_guild_id: int,
        msg_author_name: str,
        msg_content: str,
        msg_channel: discord.TextChannel,
    ):
        if (
            datetime.now() - self.last_message_time < self.message_cooldown
            or not self.is_responsive
        ):
            print("not responsive or cd not rdy")
            return
        self.last_message_time = datetime.now()

        self.target_user = msg_author_name
        self.target_channel = msg_channel

        # Emote format is <:kek:648738053074190366>
        msg_content_clean: str = re.sub("<:.*?:\d*>", "", msg_content)

        print(f'Send content "{msg_content_clean}" to server')
        await self.sio.emit(
            "sendmessage", (self.session_token, msg_content_clean, self.target_user)
        )
        self.is_responsive = False


class CharlieDiscordClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super(CharlieDiscordClient, self).__init__(*args, **kwargs)

        self.valid_guild_ids = json.load(
            open(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "guilds.json")
            )
        )["ids"]

        self.sessions: dict[int, CharlieBackendSession] = {}

    async def on_ready(self):
        print(f"{client.user} has connected to Discord!")
        print(f"{client.guilds}", type(client.guilds))
        for g in client.guilds:
            if g.id in self.valid_guild_ids:
                print(g.id, "is valid")
                self.sessions[g.id] = CharlieBackendSession(g.id)
                await self.sessions[g.id].async_connect_sio()

    async def on_message(self, message: discord.Message):
        if message.author == client.user or (type(message.author) != discord.Member):
            return

        msg_guild_id: int = message.guild.id
        msg_author_name: str = message.author.name
        msg_content: str = message.content
        msg_channel: discord.TextChannel = message.channel

        if (
            msg_channel.name != "charlie-test"
            and msg_channel.name != "chat-with-charlie"
        ):
            print("wrong channel name")
            return
        elif msg_guild_id not in self.valid_guild_ids:
            print("not a valid guild")
            return

        await self.sessions[msg_guild_id].dispatch_message(
            msg_guild_id, msg_author_name, msg_content, msg_channel
        )


token = json.load(
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.json"))
)["token"]

intents = discord.Intents(messages=True, guilds=True, message_content=True)

client = CharlieDiscordClient(intents=intents)
client.run(token)
