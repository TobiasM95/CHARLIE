import Grid from "@mui/material/Grid";
import TextField from "@mui/material/TextField";
import ToggleButton from "@mui/material/ToggleButton";
import SendIcon from '@mui/icons-material/Send';
import MicIcon from '@mui/icons-material/Mic';
import Button from "@mui/material/Button";
import { useEffect, useRef, useState } from "react";
import { ConversationMessage } from "../datastructs/ConversationContent";
import ChatHistoryDisplay from "./ChatHistoryDisplay";
import Live2dIframe from "../common/Live2dIframe";
import '../index.css';


interface IConversationPageProps {
    toolbarHeight: string;
    canInteract: boolean;
    sendMessageFunction(message: string): void;
    conversationMessages: ConversationMessage[];
    showSystemMessages: boolean;
    isRecording: boolean;
    setRecording(newIsRecording: boolean): void;
    isVoiceRecording: boolean;
    gender: string;
    session_token: string;
}

function ConversationPage({ toolbarHeight, canInteract, sendMessageFunction, conversationMessages, showSystemMessages, isRecording, setRecording, isVoiceRecording, gender, session_token }: IConversationPageProps) {

    const [messageState, setMessageState] = useState<string>("")
    const messagesEndRef = useRef<HTMLDivElement>()
    const inputTextFieldRef = useRef<HTMLDivElement>()

    function handleTextFieldKeyDown(event: any) {
        if (event.keyCode === 13) {
            event.preventDefault();
            sendMessageFunction(messageState);
            setMessageState("");
        }
    }

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'auto' })
    });

    useEffect(() => {
        inputTextFieldRef.current?.focus();
    }, [canInteract])

    return (
        <Grid container
            sx={{
                height: `calc(100vh - ${toolbarHeight})`,
                marginTop: toolbarHeight,
                overflowX: "hidden",
                overflowY: "auto",
                direction: "column"
            }}>
            <Grid item xs={12} style={{ height: `calc((100vh - ${toolbarHeight}) * 0.22)` }}>
                <Live2dIframe session_token={session_token} />
            </Grid>
            <Grid xs={12} item container style={{ height: `calc((100vh - ${toolbarHeight}) * 0.7)`, overflowY: 'auto', overflowX: "hidden", flexGrow: 1 }}>
                <ChatHistoryDisplay
                    conversationMessages={conversationMessages}
                    showSystemMessages={showSystemMessages}
                    gender={gender}
                />
                <div ref={messagesEndRef as React.RefObject<HTMLDivElement>} />
            </Grid>
            <Grid item container xs={12} style={{ height: `calc((100vh - ${toolbarHeight}) * 0.08)` }}>
                <Grid item xs={2} display="flex" flexDirection="column" flexGrow="2">
                    <ToggleButton
                        value="check"
                        color={canInteract ? (isVoiceRecording ? "success" : "primary") : "error"}
                        selected={isRecording}
                        onChange={() => { setRecording(!isRecording); }}
                        disabled={!canInteract}
                    >
                        <MicIcon />
                    </ToggleButton>
                </Grid>
                <Grid item xs={8}>
                    <TextField
                        id="outlined-multiline-flexible"
                        multiline
                        maxRows={4}
                        fullWidth
                        value={messageState}
                        onChange={(event) => { setMessageState(event.target.value) }}
                        onKeyDown={handleTextFieldKeyDown}
                        disabled={!canInteract || isRecording}
                        inputRef={inputTextFieldRef}
                    />
                </Grid>
                <Grid item xs={2} display="flex" flexDirection="row" flexGrow="1">
                    <Button variant={"contained"} onClick={() => { sendMessageFunction(messageState); setMessageState(""); }} disabled={!canInteract || isRecording}>
                        <SendIcon />
                    </Button>
                </Grid>
            </Grid>
        </Grid>
    )
}

export default ConversationPage