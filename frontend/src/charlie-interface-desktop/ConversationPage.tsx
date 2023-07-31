import Grid from "@mui/material/Grid";
import TextField from "@mui/material/TextField";
import ToggleButton from "@mui/material/ToggleButton";
import SendIcon from '@mui/icons-material/Send';
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
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'flex-end'
            }}>
            <Grid
                item
                xs
                sx={{
                    overflowY: "auto"
                }}
                display="flex"
                flexDirection="row"
                justifyContent="center"
            >
                <Grid item xs={8}
                    sx={{
                        overflowY: "auto",
                        overflowX: "hidden"
                    }}>
                    <ChatHistoryDisplay
                        conversationMessages={conversationMessages}
                        showSystemMessages={showSystemMessages}
                        gender={gender}
                    />
                    <div ref={messagesEndRef as React.RefObject<HTMLDivElement>} />
                </Grid>
                <Grid item xs={4}
                    sx={{
                        overflowY: "hidden",
                        overflowX: "hidden"
                    }}>
                    <Live2dIframe session_token={session_token} />
                </Grid>
            </Grid>
            <Grid item xs="auto" container>
                <Grid item xs="auto" display="flex" flexDirection="row" justifyContent="center">
                    <ToggleButton
                        value="check"
                        color={canInteract ? (isVoiceRecording ? "success" : "primary") : "error"}
                        selected={isRecording}
                        onChange={() => { setRecording(!isRecording); }}
                        disabled={!canInteract}
                    >
                        Activate microphone input
                    </ToggleButton>
                </Grid>
                <Grid item xs>
                    <TextField
                        id="outlined-multiline-flexible"
                        label="Message to Charlie"
                        multiline
                        rows={2}
                        fullWidth
                        value={messageState}
                        onChange={(event) => { setMessageState(event.target.value) }}
                        onKeyDown={handleTextFieldKeyDown}
                        disabled={!canInteract || isRecording}
                        inputRef={inputTextFieldRef}
                    />
                </Grid>
                <Grid item xs="auto" display="flex" flexDirection="row" flexGrow="1">
                    <Button variant={"contained"} onClick={() => { sendMessageFunction(messageState); setMessageState(""); }} disabled={!canInteract || isRecording}>
                        <SendIcon />
                    </Button>
                </Grid>
            </Grid>
        </Grid>
    )
}

export default ConversationPage