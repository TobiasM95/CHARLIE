####################################################################
#####     DATA STRUCTS      ########################################
####################################################################


class Mode:
    DORMANT = "DORMANT"
    SYSTEM = "SYSTEM"
    CONVERSATION = "CONVERSATION"
    TRANSLATION = "TRANSLATION"
    INFORMATION = "INFORMATION"


class Language:
    GERMAN = "DE"
    ENGLISH = "EN-US"

    @staticmethod
    def get(name):
        if name == "DE":
            return "DE"
        if name == "EN-US":
            return "EN-US"
        raise Exception(f"Language {name} not yet supported!")


class Gender:
    FEMALE = "female"
    MALE = "male"

    @staticmethod
    def get(gender):
        if gender == "female":
            return "female"
        if gender == "male":
            return "male"
        raise Exception(f"Gender {gender} not yet supported!")


class MessagePair:
    def __init__(
        self,
        name_user,
        msg_user=None,
        msg_charlie_dict=None,
        reply_style=None,
        reply_length=None,
    ):
        self.name_user = name_user
        self.msg_user = msg_user
        self.msg_charlie = None
        self.msg_charlie_raw = None
        self.msg_charlie_clean = None
        self.msg_charlie_style = None
        if msg_charlie_dict is not None:
            if "none" in msg_charlie_dict:
                self.msg_charlie = msg_charlie_dict["none"]
            if "raw" in msg_charlie_dict:
                self.msg_charlie_raw = msg_charlie_dict["raw"]
            if "clean" in msg_charlie_dict:
                self.msg_charlie_clean = msg_charlie_dict["clean"]
            if "style" in msg_charlie_dict:
                self.msg_charlie_style = msg_charlie_dict["style"]

        # reply style that was used to generate the response of chat gpt
        self.reply_style = reply_style
        self.reply_length = reply_length


class RecordingState:
    is_recording = False
    recorded_frames = 0


class RecordingParams:
    def __init__(self, config):
        self.buffer_length_seconds = config["recording_settings"][
            "buffer_length_seconds"
        ]
        self.sample_rate = config["recording_settings"]["sample_rate"]
        self.chunk_size = config["recording_settings"]["chunk_size"]
        self.activation_threshold = config["recording_settings"]["activation_threshold"]
        self.deactivation_threshold = config["recording_settings"][
            "deactivation_threshold"
        ]
        self.activation_window = config["recording_settings"]["activation_window"]
        self.deactivation_window = config["recording_settings"]["deactivation_window"]


gpt_models = {
    # Mode.CONVERSATION: "text-davinci-003",
    Mode.CONVERSATION: "gpt-3.5-turbo",
    Mode.INFORMATION: "gpt-3.5-turbo",
}
