import "../index.css";
import { useEffect } from "react";
import useState from 'react-usestateref'
import { ConversationInfo } from "./ConversationInfo";
import { ConversationContent, ConversationMessage } from "./ConversationContent";
import { conversationsOverviewAPI, conversationContentAPI } from "./ConversationsOverviewAPI";
import { wsURL } from "../Settings/Constants";

import MuiAppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import Drawer from "@mui/material/Drawer";
import Divider from "@mui/material/Divider";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";

import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import ChatIcon from "@mui/icons-material/Chat";
import AddCommentIcon from "@mui/icons-material/AddComment";
import SettingsIcon from "@mui/icons-material/Settings";
import LogoutIcon from "@mui/icons-material/Logout";
import HomeIcon from '@mui/icons-material/Home';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import VolumeDownIcon from '@mui/icons-material/VolumeDown';
import { THEMENAME } from "../App";
import Switch from "@mui/material/Switch";
import moment from "moment";
import { dateTimeFullFormat } from "../Settings/Constants";
import ConversationReviewPage from "./ConversationReviewPage";
import ConversationPage from "./ConversationPage";
import { Socket, io } from "socket.io-client";
import { DefaultEventsMap } from "socket.io/dist/typed-events";

import RecordRTC, { StereoAudioRecorder } from "recordrtc";
import TextField from "@mui/material/TextField";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Button from "@mui/material/Button";
import Slider from "@mui/material/Slider";
import Icon from "@mui/material/Icon";

const toolbarHeight = "64px";
const drawerWidth = 300;

function StringToConversationMessage(index: number, message: string): ConversationMessage {
  console.log("index, message: ", index, message)
  // [2023-03-26T00:37:44][SYSTEM, system] Initialize Google TTS API..
  const regexp = /^\[(.*?)\]\[(.*?),\s*(.*?)\]\s*(.*)$/g;
  const matches = Array.from(message.matchAll(regexp))[0];

  let convMsg: ConversationMessage = new ConversationMessage({
    "index": index,
    "message": matches[4],
    "mode": matches[2],
    "speaker": matches[3],
    "timestamp": matches[1]
  });
  return convMsg;
}

export interface IConversationsPageProps {
  changeAppTheme: (themeName: THEMENAME) => void;
  logOutFunc: () => void;
  userFirstName: string;
  userSUB: string; // is a google account unique ID
}

function ConversationsPage({ changeAppTheme, logOutFunc, userFirstName, userSUB }: IConversationsPageProps) {
  type Mode = "DEFAULT" | "CONVERSATION" | "REVIEW"; // | "SETTINGS"

  //This is temporary for quicker development and only allows a toggle between light and dark theme
  const [useDarkTheme, setUseDarkTheme] = useState<boolean>(true);
  const [showSystemMessages, setShowSystemMessages] = useState<boolean>(false);
  const [playRecordingSound, setPlayRecordingSound] = useState<boolean>(false);
  const [volumeSetting, setVolumeSetting] = useState<number>(30);

  const toggleTheme = () => {
    if (useDarkTheme) {
      setUseDarkTheme(false);
      changeAppTheme("LIGHT");
    } else {
      setUseDarkTheme(true);
      changeAppTheme("DARK");
    }
  };

  const toggleShowSystemMessages = () => {
    setShowSystemMessages(showSystemMessages ? false : true)
  }

  const togglePlayRecordingSound = () => {
    setPlayRecordingSound(playRecordingSound ? false : true)
  }

  const [conversationInfos, setConversationInfos] = useState<ConversationInfo[]>([]);
  const [conversationContent, setConversationContent] = useState<ConversationContent | undefined>();
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | undefined>(undefined);

  const [mode, setMode] = useState<Mode>("DEFAULT");

  const [socketInstance, setSocketInstance] = useState<Socket<DefaultEventsMap, DefaultEventsMap> | undefined>(undefined);
  const [canInteract, setCanInteract] = useState<boolean>(false);
  const [currentConversationMessages, setCurrentConversationMessages] = useState<ConversationMessage[]>([]);

  const [isRecording, setRecording] = useState<boolean>(false);
  const [recordingCacheLength, setRecordingCacheLength] = useState<number>(0);
  const [recorder, setRecorder] = useState<RecordRTC | undefined>(undefined);
  const [isVoiceRecording, setIsVoiceRecording] = useState<boolean>(false);

  const [userUID] = useState<string>(userSUB);
  const [sessionKey, setSessionKey, sessionKeyRef] = useState<string>("");
  const [sessionToken, setSessionToken, sessionTokenRef] = useState<string>("");
  const [settingPersistentSession, setSettingPersistentSession, settingPersistentSessionRef] = useState<boolean>(false);
  const [settingUserName, setSettingUserName, settingUserNameRef] = useState<string>(userFirstName === "" ? "John" : userFirstName);
  const [settingCharlieGender, setSettingCharlieGender, settingCharlieGenderRef] = useState<string>("female");
  const [settingUserGender, setSettingUserGender, settingUserGenderRef] = useState<string>("male");
  const [settingLanguage, setSettingLanguage, settingLanguageRef] = useState<string>("EN-US");
  const [settingMemorySize, setSettingMemorySize, settingMemorySizeRef] = useState<number>(3);
  const [settingStyle, setSettingStyle, settingStyleRef] = useState<string>("stereotypical good mate");
  const [settingTTSMethod, setSettingTTSMethod, settingTTSMethodRef] = useState<string>("google_tts");

  function handleLogOutSelection() {
    setRecording(false);
    setIsVoiceRecording(false);
    disconnectSocketIO();
    setMode("DEFAULT");
    logOutFunc();
  }

  const handleHomeScreenSelection = () => {
    if (mode === "CONVERSATION") {
      setRecording(false);
      setIsVoiceRecording(false);
      disconnectSocketIO();
      //endCharlieREST()
    }
    setMode("DEFAULT");
  }

  const handleStartConversationSelection = (persistentSession: boolean) => {
    setSettingPersistentSession(persistentSession)
    if (mode !== "CONVERSATION") {
      connectSocketIO();
      //initCharlieREST()
    }
    setMode("CONVERSATION")
  }

  const [conversationID, setConversationID] = useState<number>(-1);
  const handleMessageReviewSelection = (id: number) => {
    if (mode === "CONVERSATION") {
      disconnectSocketIO();
      //endCharlieREST()
    }
    setConversationID(id);
    setMode("REVIEW");
  };

  const [conversationsDrawerOpen, setconversationsDrawerOpen] = useState(false);
  const handleconversationsDrawerOpen = () => {
    setconversationsDrawerOpen(true);
  };

  const handleconversationsDrawerClose = () => {
    setconversationsDrawerOpen(false);
  };

  const [settingsDrawerOpen, setSettingsDrawerOpen] = useState(false);
  const handleSettingsDrawerOpen = () => {
    setSettingsDrawerOpen(true);
  };

  const handleSettingsDrawerClose = () => {
    setSettingsDrawerOpen(false);
  };

  function connectSocketIO() {
    if (!socketInstance) {
      const socket = io(wsURL, {
        transports: ["websocket"],
      });

      socket.on("connect", () => {
        console.log("Connected at ", socket.id);
        const sessionKeyLocal = new Date().toISOString();
        setSessionKey(prevSessionKey => (sessionKeyLocal));
        console.log("Send info to backend:", userUID, sessionKeyLocal)
        socket.emit(
          "initcharlie",
          userUID,
          sessionKeyLocal,
          settingPersistentSessionRef.current,
          {
            "userUID": userUID,
            "name": settingUserNameRef.current === "" ? "John" : settingUserNameRef.current,
            "gender": settingCharlieGenderRef.current,
            "gender-user": settingUserGenderRef.current,
            "language": settingLanguageRef.current,
            "memory_size": settingMemorySizeRef.current,
            "style_en": settingStyleRef.current === "" ? "stereotypical good mate" : settingStyleRef.current,
            "tts-method": settingTTSMethodRef.current
          }
        )
      });

      socket.on("disconnect", (reason) => {
        console.log("Disconnected for ", reason);
      });

      socket.on("charliesessioninit", (data) => {
        console.log("charliesessioninit with", data, sessionKeyRef);
        if (sessionKeyRef.current !== data["key"]) {
          return;
        }
        console.log("update session token", data["session_token"]);
        setSessionToken(prevSessionToken => (data["session_token"]))
      });

      socket.on("responsiveness", (data) => {
        if (sessionTokenRef.current !== data["session_token"]) {
          return;
        }
        setCanInteract(data["isResponsive"])
        console.log(data, data["isResponsive"])
      });

      socket.on("convmsg", (data) => {
        console.log("convmsg", sessionTokenRef.current, data["session_token"]);
        if (sessionTokenRef.current !== data["session_token"]) {
          return;
        }
        const newConversationMessage: ConversationMessage = StringToConversationMessage(0, data["message"]);
        if (newConversationMessage.message === "Shut down Charlie..") {
          return;
        } else if (newConversationMessage.message === "Start session.") {
          console.log("Frontend: Restart conversation")
          setCurrentConversationMessages(
            prevState => {
              return [newConversationMessage]
            }
          )
          return;
        }
        setCurrentConversationMessages(
          prevState => {
            if (prevState) {
              return [
                ...prevState,
                newConversationMessage
              ]
            } else {
              return [newConversationMessage]
            }
          }
        )
      });

      socket.on("startrecording", (data) => {
        if (sessionTokenRef.current !== data["session_token"]) {
          return;
        }
        setIsVoiceRecording(true);
      })

      socket.on("stoprecording", (data) => {
        if (sessionTokenRef.current !== data["session_token"]) {
          return;
        }
        setIsVoiceRecording(false);
      })

      socket.on("responseaudio", (data) => {
        if (sessionTokenRef.current !== data["session_token"]) {
          return;
        }
        console.log("play audio")
        const audio = new Audio();
        const blob = new Blob([data["audio_content"]], { type: "audio/wav" });
        audio.src = URL.createObjectURL(blob);
        audio.volume = volumeSetting / 100.0;
        audio.play();
      });

      socket.on("logging", (data) => {
        if (sessionTokenRef.current !== data["session_token"]) {
          return;
        }
        console.log(data["message"]);
      })

      setSocketInstance(socket);
      console.log("Create new socketInstance")
    }
    else if (socketInstance.connected === false) {
      socketInstance.connect();
      console.log("Reconnected at: ", socketInstance.id)
    }
    else {
      console.log("Already connected at: ", socketInstance.id)
    }
    setCanInteract(true)
    setCurrentConversationMessages(prevState => { return [] })
  }

  function disconnectSocketIO() {
    if (socketInstance && socketInstance.connected === true) {
      socketInstance.emit("endcharlie", sessionTokenRef.current)
      socketInstance.disconnect();
      console.log("Disconnect")
    }
    else if (socketInstance && socketInstance.connected === false) {
      console.log("Already disconnected")
    }
    else {
      console.log("socketInstance undefined")
    }
    if (recorder) {
      if (isRecording) {
        recorder.stopRecording();
      }
      recorder.destroy();
    }
  }

  function updateCharlieSettings() {
    if (socketInstance && socketInstance.connected === true) {
      socketInstance.emit(
        "updateconfig",
        sessionTokenRef.current,
        {
          "name": settingUserNameRef.current === "" ? "John" : settingUserNameRef.current,
          "gender": settingCharlieGenderRef.current,
          "gender-user": settingUserGenderRef.current,
          "language": settingLanguageRef.current,
          "memory_size": settingMemorySizeRef.current,
          "style_en": settingStyleRef.current === "" ? "a good mate" : settingStyleRef.current,
          "tts-method": settingTTSMethodRef.current
        }
      );
    }
  }

  function sendMessage(message: string) {
    if (message !== "" && socketInstance) {
      console.log("Send message to server", message);
      setCanInteract(false)
      socketInstance.emit("sendmessage", sessionTokenRef.current, message)
    }
  }

  useEffect(() => {
    async function updateConversationData(id: number) {
      setLoading(true);
      setConversationContent(undefined);
      try {
        const data = await conversationContentAPI.get(userUID, id);
        setError("");
        setConversationContent(data);
      } catch (e) {
        if (e instanceof Error) {
          setError(e.message);
        }
      } finally {
        setLoading(false);
      }
    }
    if (mode === "REVIEW" && conversationID >= 0) {
      updateConversationData(conversationID);
    }
  }, [mode, conversationID, userUID]);

  useEffect(() => {
    async function loadConversationInfos() {
      setLoading(true);
      setConversationInfos([]);
      try {
        const data = await conversationsOverviewAPI.get(userUID);
        setError("");
        setConversationInfos(data);
      } catch (e) {
        if (e instanceof Error) {
          setError(e.message);
        }
      } finally {
        setLoading(false);
      }
    }
    if (conversationsDrawerOpen) {
      loadConversationInfos();
    }
  }, [conversationsDrawerOpen, userUID]);

  async function setRecorderRef() {
    const audioStream = await navigator.mediaDevices.getUserMedia({
      audio: true
    })
    const newRecorder = new RecordRTC(audioStream, {
      type: 'audio',
      mimeType: 'audio/wav',
      // sampleRate: 16000,
      desiredSampRate: 16000,
      recorderType: StereoAudioRecorder,
      numberOfAudioChannels: 1,

      // get intervals based blobs
      // value in milliseconds
      timeSlice: 1000,

      // requires timeSlice above
      // returns blob via callback function
      ondataavailable: function (blob: Blob) {
        if (socketInstance && socketInstance.connected) {
          socketInstance.emit("streamaudio", sessionTokenRef.current, { "sr": newRecorder.sampleRate, "blob": blob });
        }
        setRecordingCacheLength((prevState) => (prevState + blob.size / 32000))
      },
    })
    setRecorder(newRecorder)
    newRecorder.startRecording();
  }
  useEffect(() => {
    if (isRecording) {
      console.log("Start recording");
      setRecorderRef();
    }
    else {
      console.log("Stop recording");
      if (recorder) {
        // recorder.stopRecording(() => {
        //   invokeSaveAsDialog(recorder.getBlob());
        // });
        recorder.stopRecording();
      }
      else {
        console.log("Error stopping the recording");
      }

    }
  }, [isRecording])

  useEffect(() => {
    if (!playRecordingSound) {
      return;
    }
    if (isVoiceRecording) {
      const audio = new Audio("./sounds/start.mp3")
      audio.volume = volumeSetting / 100.0;
      audio.play()
    } else {
      const audio = new Audio("./sounds/stop.mp3")
      audio.volume = volumeSetting / 100.0;
      audio.play()
    }
  }, [isVoiceRecording])

  useEffect(() => {
    if (recordingCacheLength > 300) {
      recorder?.reset()
      recorder?.startRecording()
      setRecordingCacheLength((prevState) => (0))
      console.log("clear recording cache")
    }
  }, [recordingCacheLength])

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "center",
      }}
    >
      <MuiAppBar
        position="fixed"
        sx={{
          height: toolbarHeight,
          zIndex: (theme) => theme.zIndex.drawer + 1,
          boxShadow: "none",
        }}
      >
        <Toolbar>
          <Typography
            variant="h6"
            noWrap
            component="div"
          >
            CHARLIE - Conversational Human-like ARtificial intelligence for Language Interactions through voicE
          </Typography>
        </Toolbar>
      </MuiAppBar>
      <Drawer
        variant="permanent"
        anchor="left"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: {
            width: drawerWidth,
            boxSizing: "border-box",
          },
        }}
      >
        <Toolbar />
        <List>
          <ListItem
            key="homeScreen"
            disablePadding
          >
            <ListItemButton onClick={handleHomeScreenSelection}>
              <ListItemIcon>
                <HomeIcon />
              </ListItemIcon>
              <ListItemText primary="Home" />
            </ListItemButton>
          </ListItem>
          <ListItem
            key="startNewConvo"
            disablePadding
          >
            <ListItemButton onClick={() => handleStartConversationSelection(true)}>
              <ListItemIcon>
                <Icon><img src={"./images/charlieAvatarFemaleIcon.png"} height={24} width={24} /></Icon>
              </ListItemIcon>
              <ListItemText primary="Persistent chat (BETA)" />
            </ListItemButton>
          </ListItem>
          <ListItem
            key="startNewConvo"
            disablePadding
          >
            <ListItemButton onClick={() => handleStartConversationSelection(false)}>
              <ListItemIcon>
                <AddCommentIcon />
              </ListItemIcon>
              <ListItemText primary="Start a new conversation" />
            </ListItemButton>
          </ListItem>
          <ListItem
            key="reviewConvos"
            disablePadding
          >
            <ListItemButton onClick={handleconversationsDrawerOpen}>
              <ListItemIcon>
                <ChatIcon />
              </ListItemIcon>
              <ListItemText primary="Review conversations" />
              <ChevronRightIcon />
            </ListItemButton>
          </ListItem>
        </List>
        <Divider sx={{ marginTop: "auto" }} />
        <List>
          <ListItem
            key="accountSettings"
            disablePadding
            onClick={handleSettingsDrawerOpen}
          >
            <ListItemButton>
              <ListItemIcon>
                <SettingsIcon />
              </ListItemIcon>
              <ListItemText primary="Account settings" />
            </ListItemButton>
          </ListItem>
          <ListItem
            key="logOut"
            disablePadding
            onClick={handleLogOutSelection}
          >
            <ListItemButton>
              <ListItemIcon>
                <LogoutIcon />
              </ListItemIcon>
              <ListItemText primary="Log out" />
            </ListItemButton>
          </ListItem>
        </List>

        <Drawer
          className="{classes.drawer}"
          variant="persistent"
          anchor="left"
          open={conversationsDrawerOpen}
        >
          <Toolbar />
          <ListItem
            key="back"
            sx={{ px: 0, pb: 0 }}
            onClick={handleconversationsDrawerClose}
          >
            <ListItemButton>
              <ListItemIcon>
                <ChevronLeftIcon />
              </ListItemIcon>
              <ListItemText primary="Back" />
            </ListItemButton>
          </ListItem>
          <Divider />
          <List disablePadding sx={{ overflowY: "auto" }}>
            {conversationInfos.map((info, id) => (
              <ListItem
                key={id}
                disablePadding
              >
                <ListItemButton onClick={() => handleMessageReviewSelection(Number(info.id))}>
                  <ListItemIcon>
                    <ChatIcon />
                  </ListItemIcon>
                  <ListItemText primary={moment(info.date).format(dateTimeFullFormat)} />
                </ListItemButton>
              </ListItem>
            ))}
            {error && (
              <div className="row">
                <div className="card large error">
                  <section>
                    <p>
                      <span className="icon-alert inverse "></span>
                      {error}
                    </p>
                  </section>
                </div>
              </div>
            )}
            {loading && (
              <ListItem key={"loading"}>
                <ListItemText primary="Loading..." />
              </ListItem>
            )}
          </List>
        </Drawer>

        <Drawer
          className="{classes.drawer}"
          variant="persistent"
          anchor="left"
          open={settingsDrawerOpen}
        >
          <Toolbar />
          <List disablePadding>
            <ListItem
              key="back"
              sx={{ px: 0, pb: 0 }}
              onClick={handleSettingsDrawerClose}
            >
              <ListItemButton>
                <ListItemIcon>
                  <ChevronLeftIcon />
                </ListItemIcon>
                <ListItemText primary="Back" />
              </ListItemButton>
            </ListItem>
            <Divider />
            <ListItem
              key="systemSettingsLabel"
            >
              <ListItemText primary="System settings:" />
            </ListItem>
            <ListItem
              key="themeToggle"
              disablePadding
            >
              <ListItemButton onClick={toggleTheme}>
                <ListItemText primary="Dark theme" />
              </ListItemButton>
              <Switch
                checked={useDarkTheme}
                onChange={toggleTheme}
                inputProps={{ "aria-label": "controlled" }}
              />
            </ListItem>
            <ListItem
              key="sysMsgToggle"
              disablePadding
            >
              <ListItemButton onClick={toggleShowSystemMessages}>
                <ListItemText primary="Show system messages" />
              </ListItemButton>
              <Switch
                checked={showSystemMessages}
                onChange={toggleShowSystemMessages}
                inputProps={{ "aria-label": "controlled" }}
              />
            </ListItem>
            <ListItem
              key="recordSoundToggle"
              disablePadding
            >
              <ListItemButton onClick={togglePlayRecordingSound}>
                <ListItemText primary="Play sound on start/stop recording" />
              </ListItemButton>
              <Switch
                checked={playRecordingSound}
                onChange={togglePlayRecordingSound}
                inputProps={{ "aria-label": "controlled" }}
              />
            </ListItem>
            <ListItem
              key="volumeSettingSlider"
              disablePadding
            >
              <VolumeDownIcon />
              <Slider aria-label="Volume" min={0} max={100} value={volumeSetting} onChange={(event, value) => { setVolumeSetting(Number(value)) }} />
              <VolumeUpIcon />
            </ListItem>
            <Divider />
            <ListItem
              key="charlieSettingsLabel"
            >
              <ListItemText primary="Charlie settings:" />
            </ListItem>
            <ListItem
              key="charlieUserNameTextField"
            >
              <ListItemText primary="User name:" />
              <TextField
                id="userNameTextField"
                margin="dense"
                maxRows={1}
                size="small"
                variant="standard"
                inputProps={{
                  maxLength: 19,
                  placeholder: "John",
                  title: "Allowed characters are: a-z A-Z äöüÄÖÜßáéíóúàèìòùâêîôûÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ"
                }}
                value={settingUserName}
                onChange={(e) => {
                  setSettingUserName(e.target.value.replaceAll(/[^a-zA-ZäöüÄÖÜßáéíóúàèìòùâêîôûÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ]/g, ""))
                }}
              />
            </ListItem>
            <ListItem
              key="charlieGenderToggle"
              disablePadding
            >
              <ListItemButton onClick={() => { settingCharlieGender === "male" ? setSettingCharlieGender("female") : setSettingCharlieGender("male") }}>
                <ListItemText primary="Charlie's gender:" />
              </ListItemButton>
              <Typography>Male</Typography>
              <Switch
                checked={settingCharlieGender === "female"}
                onChange={() => { settingCharlieGender === "male" ? setSettingCharlieGender("female") : setSettingCharlieGender("male") }}
                inputProps={{ "aria-label": "controlled" }}
                sx={{ m: -1 }}
              />
              <Typography sx={{ pr: 2 }}>Female</Typography>
            </ListItem>
            <ListItem
              key="charlieUserToggle"
              disablePadding
            >
              <ListItemButton onClick={() => { settingUserGender === "male" ? setSettingUserGender("female") : setSettingUserGender("male") }}>
                <ListItemText primary="User's gender:" />
              </ListItemButton>
              <Typography>Male</Typography>
              <Switch
                checked={settingUserGender === "female"}
                onChange={() => { settingUserGender === "male" ? setSettingUserGender("female") : setSettingUserGender("male") }}
                inputProps={{ "aria-label": "controlled" }}
                sx={{ m: -1 }}
              />
              <Typography sx={{ pr: 2 }}>Female</Typography>
            </ListItem>
            <ListItem
              key="charlieLanguageDropdown"
            >
              <ListItemText primary="Language:" />
              <Select
                labelId="charlieLDLID"
                id="charlieLDID"
                value={settingLanguage}
                onChange={(e) => { setSettingLanguage(e.target.value) }}
                variant="standard"
              >
                <MenuItem value={"EN-US"}>English</MenuItem>
                <MenuItem value={"DE"}>Deutsch</MenuItem>
              </Select>
            </ListItem>
            <ListItem
              key="charlieMemSizeDropdown"
            >
              <ListItemText primary="Memory capacity:" />
              <Select
                labelId="charlieMSDLID"
                id="charlieMSDID"
                value={settingMemorySize}
                onChange={(e) => { setSettingMemorySize(+e.target.value) }}
                variant="standard"
              >
                <MenuItem value={1}>1</MenuItem>
                <MenuItem value={2}>2</MenuItem>
                <MenuItem value={3}>3</MenuItem>
              </Select>
            </ListItem>
            <ListItem
              key="charlieStyleDescLabel"
              sx={{ pb: 0 }}
            >
              <ListItemText primary="Style description:" />
            </ListItem>
            <ListItem
              key="charlieStyleDescInput"
              sx={{ pt: 0 }}
            >
              <TextField
                fullWidth id="styleDescInput"
                margin="dense"
                maxRows={1}
                size="small"
                variant="standard"
                inputProps={{
                  maxLength: 150,
                  placeholder: "a good mate",
                  title: "Allowed characters are: a-z A-Z äöüÄÖÜßáéíóúàèìòùâêîôûÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ space,"
                }}
                value={settingStyle}
                onChange={(e) => {
                  setSettingStyle(e.target.value.replaceAll(/[^a-zA-ZäöüÄÖÜßáéíóúàèìòùâêîôûÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ ,]/g, ""))
                }}
              />
            </ListItem>
            <ListItem
              key="charlieTTSMethodDropdown"
            >
              <ListItemText primary="TTS Method:" />
              <Select
                labelId="charlieTTSMD"
                id="charlieTTSID"
                value={settingTTSMethod}
                onChange={(e) => { setSettingTTSMethod(e.target.value) }}
                variant="standard"
              >
                <MenuItem value={"google_tts"}>Google TTS</MenuItem>
                <MenuItem value={"elevenlabs_tts"}>elevenlabs.ai</MenuItem>
                <MenuItem value={"notts"}>No TTS</MenuItem>
              </Select>
            </ListItem>
            <ListItem
              key="charlieSettingsUpdateButtonItem"
            >
              <Button onClick={updateCharlieSettings} variant="outlined">
                Update settings
              </Button>
            </ListItem>
          </List>
        </Drawer>
      </Drawer>

      {mode === "DEFAULT" && (
        <Box
          component="div"
          sx={{
            height: `calc(100vh - ${toolbarHeight})`,
            display: "flex",
            flexGrow: 1,
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            marginTop: toolbarHeight,
          }}
        >
          <Box className="charlieBackgroundTextBox" sx={{ opacity: 0.7, unselectable: "on" }}>
            <Typography
              variant="h6"
              align="center"
            >
              Welcome to CHARLIE
            </Typography>
            <Typography
              variant="overline"
              align="center"
              display="block"
            >
              Start a new conversation on the left
            </Typography>
            <Typography
              variant="overline"
              align="center"
              display="block"
            >
              or review past conversations.
            </Typography>
            <Typography
              variant="overline"
              display="block"
            >
              You can also change account settings and log out
            </Typography>
          </Box>
        </Box>
      )}

      {mode === "CONVERSATION" && (
        <ConversationPage
          toolbarHeight={toolbarHeight}
          canInteract={canInteract}
          sendMessageFunction={sendMessage}
          conversationMessages={currentConversationMessages}
          showSystemMessages={showSystemMessages}
          isRecording={isRecording}
          setRecording={setRecording}
          isVoiceRecording={isVoiceRecording}
          gender={settingCharlieGenderRef.current}
          session_token={sessionToken}
        />
      )}

      {mode === "REVIEW" && (
        <ConversationReviewPage
          conversationContent={conversationContent}
          showSystemMessages={showSystemMessages}
          toolbarHeight={toolbarHeight}
          gender={settingCharlieGenderRef.current}
        />
      )}
    </Box>
  );
}

export default ConversationsPage;
