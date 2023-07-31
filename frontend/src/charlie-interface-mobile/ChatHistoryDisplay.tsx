import Grid from "@mui/material/Grid";
import MessageDisplay from "./MessageDisplay";
import { ConversationMessage } from "../datastructs/ConversationContent";

interface IChatHistoryDisplayProps {
    conversationMessages: ConversationMessage[];
    showSystemMessages: boolean;
    gender: string;
}

function ChatHistoryDisplay({ conversationMessages, showSystemMessages, gender }: IChatHistoryDisplayProps) {
    return (
        <Grid
            container
            item
            xs={12}
            rowSpacing={1}
            p={2}
        >
            {conversationMessages.filter((message) => (message.mode !== 'SYSTEM' || showSystemMessages)).map((message, id) => (
                <Grid
                    item
                    xs={12}
                    key={id}
                >
                    <MessageDisplay
                        key={id}
                        id={id}
                        message={message}
                        gender={gender}
                    />
                </Grid>
            ))}
        </Grid>
    )
}

export default ChatHistoryDisplay