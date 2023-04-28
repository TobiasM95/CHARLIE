import sys
import os
import shutil
import re
import json
import hashlib
from threading import Lock
import datetime

import numpy as np
import soundfile as sf

from flask import Flask, Response, request
from flask_cors import CORS
from flask_socketio import SocketIO

import charlie
import utils.data_structs as uds

mutex = Lock()

app = Flask(__name__, static_folder="../live2d/", static_url_path="/live2d/")
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")
# executor = Executor(app)

charlie_sessions: dict[str, charlie.CharlieSession] = {}

# TODO: Rewrite the whole socketio session token system to use rooms instead


@app.route("/")
def root() -> dict:
    return {
        "a-information": "The endpoints listed here are not suited for public use as they are very circumstantial and specialized. This is just an overview.",
        "http-endpoints": [
            "/users/access/<string:emailclean>",
            "conversations/<string:useruid>",
            "/conversation/<string:useruid>/<int:conversation_id>",
            "/newconversation/init/<string:useruid>",
            "/newconversation/end/<string:sessiontoken>",
            "/newconversation/changesettings/<string:sessiontoken>",
        ],
        "socketio-endpoints": [
            "initcharlie",
            "endcharlie",
            "updateconfig",
            "sendmessage",
            "streamaudio",
            "resendgenderinfo",
        ],
    }


@app.route("/users/access/<string:emailhash>")
def checkUserAccess(emailhash="") -> dict:
    # TODO: Make this much more production ready, but for low user counts this should be fine
    accessList = json.load(
        open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "userAccess.json")
        )
    )
    hasAccessList = accessList["hasAccess"]
    return {"access": emailhash in hasAccessList}


@app.route("/users/requestaccess", methods=["POST"])
def requestUserAccess() -> None:
    request_body = request.get_json()
    email_hash = request_body["email-hash"]

    user_access_json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "userAccess.json"
    )
    accessList = json.load(open(user_access_json_path))

    if email_hash not in accessList["requestedAccess"]:
        accessList["requestedAccess"].append(email_hash)

        json.dump(accessList, open(user_access_json_path, "w"), indent=4)

    return {"status": "todo: put some useful info in here"}


@app.route("/conversations/<string:useruid>")
def get_conversations_as_dict(useruid="0" * 21) -> dict:
    conversation_list_unsorted = []
    conversation_dict = {}
    logfile_dir = os.path.join(os.path.dirname(__file__), "logfiles", useruid)
    if not os.path.isdir(logfile_dir):
        os.makedirs(logfile_dir, exist_ok=True)

    for i, conv_file in enumerate(next(os.walk(logfile_dir))[2]):
        conversation_list_unsorted.append(conv_file)
    for i, filename in enumerate(reversed(sorted(conversation_list_unsorted))):
        conversation_dict[i] = filename
    return conversation_dict


@app.route("/conversation/<string:useruid>/<int:conversation_id>")
def get_conversation(useruid="0" * 21, conversation_id=0) -> dict | None:
    conversation_dict = get_conversations_as_dict(useruid)
    if conversation_id not in conversation_dict:
        pass
    else:
        return _format_conversation_data(useruid, conversation_dict[conversation_id])


@app.route("/newconversation/init/<string:useruid>", methods=["POST"])
def init_new_conversation(useruid, key) -> dict:
    request_body = request.get_json()
    charlie_session = charlie.CharlieSession(useruid)
    charlie_session.charlie_is_responsive = False
    socketio.emit(
        "charliesessioninit",
        {"key": key, "session_token": charlie_session.session_token},
    )
    socketio.emit(
        "responsiveness",
        {"session_token": charlie_session.session_token, "isResponsive": False},
    )
    socketio.start_background_task(
        _handle_async_init_new_conversation, charlie_session, request_body
    )
    return {"status": "todo: put some useful info in here"}


def _handle_async_init_new_conversation(
    charlie_session: charlie.CharlieSession, request_body: dict
) -> None:
    global charlie_sessions
    _add_session(charlie_session)
    charlie_sessions[
        charlie_session.session_token
    ].charlie_instance.initialize_conversation(
        charlie_session.session_token, request_body, socketio=socketio
    )
    _set_responsiveness(charlie_session.session_token, True)
    socketio.emit(
        "responsiveness",
        {
            "session_token": charlie_sessions[
                charlie_session.session_token
            ].session_token,
            "isResponsive": True,
        },
    )


@app.route("/newconversation/end/<string:sessiontoken>", methods=["POST"])
def end_new_conversation(sessiontoken: str) -> dict:
    if not _get_responsiveness(sessiontoken):
        return {"status": f"Charlie session {sessiontoken} is not responsive yet!"}
    _set_responsiveness(sessiontoken, False)
    socketio.emit(
        "responsiveness",
        {"session_token": sessiontoken, "isResponsive": False},
    )
    socketio.start_background_task(_handle_async_end_new_conversation, sessiontoken)
    return {"status": "todo: put some useful info in here"}


def _handle_async_end_new_conversation(session_token: str) -> None:
    global charlie_sessions
    charlie_sessions[session_token].charlie_instance.end_conversation()
    _del_session(session_token)
    _set_responsiveness(session_token, True)
    socketio.emit(
        "responsiveness",
        {"session_token": session_token, "isResponsive": True},
    )


def _handle_async_post_text(session_token: str) -> None:
    global charlie_sessions
    charlie_sessions[session_token].charlie_instance.process_external_text_input()
    _set_responsiveness(session_token, True)
    socketio.emit(
        "responsiveness",
        {"session_token": session_token, "isResponsive": True},
    )


@app.route("/newconversation/changesettings/<string:sessiontoken>", methods=["POST"])
def change_settings_new_conversation(sessiontoken: str) -> dict:
    if not _get_responsiveness(sessiontoken):
        return {"status": f"Charlie session {sessiontoken} is not responsive yet!"}
    settings_list = request.get_json()["settings"]
    _set_responsiveness(sessiontoken, False)
    socketio.emit(
        "responsiveness",
        {"session_token": sessiontoken, "isResponsive": False},
    )
    socketio.start_background_task(
        _handle_async_change_settings_new_conversation, sessiontoken, settings_list
    )
    return {"newsettings": settings_list}


def _handle_async_change_settings_new_conversation(
    session_token: str, settings_list: list
) -> None:
    global charlie_sessions
    charlie_sessions[session_token].charlie_instance.set_settings_manually(
        settings_list=settings_list
    )
    _set_responsiveness(session_token, True)
    socketio.emit(
        "responsiveness",
        {"session_token": session_token, "isResponsive": True},
    )


@socketio.on("initcharlie")
def handle_connect(user_uid: str, key: str, config: dict) -> None:
    print("initcharlie:", user_uid, key, config)
    charlie_session = charlie.CharlieSession(user_uid)
    charlie_session.charlie_is_responsive = False
    socketio.emit(
        "charliesessioninit",
        {"key": key, "session_token": charlie_session.session_token},
    )
    socketio.emit(
        "responsiveness",
        {"session_token": charlie_session.session_token, "isResponsive": False},
    )
    socketio.start_background_task(
        _handle_async_init_new_conversation, charlie_session, config
    )


@socketio.on("endcharlie")
def handle_disconnect(session_token: str) -> None:
    print("endcharlie")
    if not _get_responsiveness(session_token):
        print("End Not responsive yet")
        socketio.emit(
            "logging",
            {
                "session_token": session_token,
                "message": f"Charlie session {session_token} is not responsive yet!",
            },
        )
        return
    _set_responsiveness(session_token, False)
    socketio.emit(
        "responsiveness",
        {"session_token": session_token, "isResponsive": False},
    )
    socketio.start_background_task(_handle_async_end_new_conversation, session_token)


@socketio.on("updateconfig")
def handle_updateconfig(session_token: str, new_config: dict) -> None:
    print("updateconfig")
    if not _get_responsiveness(session_token):
        print("End Not responsive yet")
        socketio.emit(
            "logging",
            {
                "session_token": session_token,
                "message": f"Charlie session {session_token} is not responsive yet!",
            },
        )
        return
    _set_responsiveness(session_token, False)
    socketio.emit(
        "responsiveness",
        {"session_token": session_token, "isResponsive": False},
    )
    socketio.start_background_task(
        _handle_async_update_config, session_token, new_config
    )


def _handle_async_update_config(session_token: str, new_config: dict) -> None:
    global charlie_sessions
    charlie_sessions[session_token].charlie_instance.update_config(new_config)
    _set_responsiveness(session_token, True)
    socketio.emit(
        "responsiveness",
        {"session_token": session_token, "isResponsive": True},
    )


@socketio.on("sendmessage")
def handle_message(session_token: str, message: str) -> None:
    global charlie_sessions
    print("sendmessage", message)
    if not _get_responsiveness(session_token):
        print("Send message not responsive yet")
        socketio.emit(
            "logging",
            {
                "session_token": session_token,
                "message": f"Charlie session {session_token} is not responsive yet!",
            },
        )
        return
    _set_responsiveness(session_token, False)
    socketio.emit(
        "responsiveness",
        {"session_token": session_token, "isResponsive": False},
    )
    charlie_sessions[session_token].last_update = datetime.datetime.today()
    text_accept_result = charlie_sessions[
        session_token
    ].charlie_instance.accept_external_text_input(message)
    print("DEBUG text_accept_result", text_accept_result)
    if text_accept_result is None:
        socketio.start_background_task(_handle_async_post_text, session_token)
    else:
        _set_responsiveness(True)
        socketio.emit(
            "responsiveness",
            {"session_token": session_token, "isResponsive": True},
        )


@socketio.on("streamaudio")
def process_audio_stream(session_token: str, data: object) -> None:
    global charlie_sessions
    if not _get_responsiveness(session_token):
        print("Stream audio not responsive yet")
        socketio.emit(
            "logging",
            {
                "session_token": session_token,
                "message": f"Charlie session {session_token} is not responsive yet!",
            },
        )
        return

    blob = data["blob"]
    sr = data["sr"]
    data_s16 = np.frombuffer(blob, dtype=np.int16, count=len(blob) // 2, offset=0)

    charlie_sessions[session_token].last_update = datetime.datetime.today()
    recording_event = charlie_sessions[
        session_token
    ].charlie_instance.append_and_process_audio_buffer_arr(data_s16[100:-100])

    if recording_event == "START":
        socketio.emit(
            "startrecording",
            {"session_token": session_token, "message": "Start actual voice recording"},
        )
    elif recording_event == "STOP" or recording_event == "STARTSTOP":
        socketio.emit(
            "stoprecording",
            {
                "session_token": session_token,
                "message": f"{recording_event} actual voice recording",
            },
        )
        _set_responsiveness(session_token, False)
        socketio.emit(
            "responsiveness",
            {"session_token": session_token, "isResponsive": False},
        )
        socketio.start_background_task(_handle_async_post_audio, session_token)


def _handle_async_post_audio(session_token: str) -> None:
    global charlie_sessions
    charlie_sessions[session_token].charlie_instance.process_external_audio_input()
    _set_responsiveness(session_token, True)
    socketio.emit(
        "responsiveness",
        {"session_token": session_token, "isResponsive": True},
    )


@socketio.on("resendgenderinfo")
def resend_gender_info(session_token: str):
    global charlie_sessions
    print("Resend gender info to", session_token)
    if (
        session_token in charlie_sessions
        and charlie_sessions[session_token].charlie_instance.initialized
    ):
        if charlie_sessions[session_token].charlie_instance.gender == uds.Gender.MALE:
            socketio.emit("live2dchangemodelmale", session_token)
        else:
            socketio.emit("live2dchangemodelfemale", session_token)
    else:
        socketio.emit(
            "logging",
            {
                "session_token": session_token,
                "message": f"Charlie session {session_token} is not responsive yet or the session token is invalid!",
            },
        )


def _format_conversation_data(userUID, filepath):
    conversation_data = {}
    conversation_data["lineInformation"] = []
    with open(
        os.path.join(os.path.dirname(__file__), "logfiles", userUID, filepath), "r"
    ) as conv_file:
        message_count = 0
        for i, line in enumerate(conv_file):
            message_count += 1
            line_info = _extract_line_info(line)
            conversation_data["lineInformation"].append(
                {
                    "index": i,
                    "timestamp": line_info["timestamp"],
                    "mode": line_info["mode"],
                    "speaker": line_info["speaker"],
                    "message": line_info["message"],
                }
            )
    conversation_data["metaData"] = {
        "timestamp": filepath[:-4].split("_")[1].replace("-", ":").replace(":", "-", 2),
        "messageCount": message_count,
    }
    return conversation_data


def _extract_line_info(line):
    # example line: [2023-03-17 14:16:32][SYSTEM, system] Initialize whisper {model_type} model..
    match_obj = re.match("^\[(.*?)\]\[(.*?),\s*(.*?)\]\s*(.*)$", line)
    return {
        "timestamp": match_obj.group(1),
        "mode": match_obj.group(2),
        "speaker": match_obj.group(3),
        "message": match_obj.group(4),
    }


def _set_responsiveness(session_token: str, new_responsiveness_value: bool) -> None:
    global charlie_sessions
    mutex.acquire()
    try:
        charlie_sessions[session_token].charlie_is_responsive = new_responsiveness_value
    finally:
        mutex.release()


def _get_responsiveness(session_token: str) -> bool:
    global charlie_sessions
    mutex.acquire()
    resp: bool = False
    try:
        resp = charlie_sessions[session_token].charlie_is_responsive
    finally:
        mutex.release()

    return resp


def _add_session(session: charlie.CharlieSession) -> None:
    global charlie_sessions
    # cleanup old sessions
    current_timestamp: datetime.datetime = datetime.datetime.today()
    for session_token, charlie_session in charlie_sessions.items():
        if (current_timestamp - charlie_session.last_update) > datetime.timedelta(
            seconds=60 * 60
        ):
            print(f"End session {session_token} due to old age")
            charlie_session.charlie_instance.end_conversation()
            _del_session(session_token)

    # cleanup output sound files
    output_sounds_root_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "..",
        "live2d",
        "Samples",
        "Resources",
        "output_sounds",
    )
    if os.path.isdir(output_sounds_root_dir):
        session_directories = next(os.walk(output_sounds_root_dir))[1]
        for directory in session_directories:
            if directory not in charlie_sessions:
                try:
                    shutil.rmtree(os.path.join(output_sounds_root_dir, directory))
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))

    # add new session
    mutex.acquire()
    try:
        charlie_sessions[session.session_token] = session
    finally:
        mutex.release()


def _del_session(session_token: str) -> None:
    global charlie_sessions
    mutex.acquire()
    try:
        if session_token in charlie_sessions:
            del charlie_sessions[session_token]
    finally:
        mutex.release()


def _read_protocol_from_frontend_settings() -> bool:
    with open("../frontend/src/Settings/Constants.ts", "r") as constants_file:
        all_constants = constants_file.read().replace("\n", "")
    protocol_match = re.search(r'protocol.*?"([\w]*)"', all_constants)
    if protocol_match is not None:
        protocol = protocol_match.group(1)
        return protocol == "https"
    else:
        return False


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "debug":
        print("Start with debug..")
        debug = True
    else:
        debug = False
    uses_https: bool = _read_protocol_from_frontend_settings()
    if uses_https:
        socketio.run(
            app,
            debug=debug,
            port=5000,
            host=sys.argv[1],
            certfile="../ssl/apache-selfsigned.crt",
            keyfile="../ssl/apache-selfsigned.key",
        )
    else:
        socketio.run(app, debug=debug, port=5000, host=sys.argv[1])
