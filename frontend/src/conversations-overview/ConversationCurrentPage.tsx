import Grid from "@mui/material/Grid";
import ConversationMessageDisplay from "./ConversationMessageDisplay";
import { ConversationMessage } from "./ConversationContent";

interface IConversationCurrentPageProps {
    conversationMessages: ConversationMessage[];
    showSystemMessages: boolean;
    gender: string;
}

function ConversationCurrentPage({ conversationMessages, showSystemMessages, gender }: IConversationCurrentPageProps) {
    return (
        <Grid
            container
            item
            xs={12}
            rowSpacing={1}
        >
            {conversationMessages.filter((message) => (message.mode !== 'SYSTEM' || showSystemMessages)).map((message, id) => (
                <Grid
                    item
                    xs={12}
                    key={id}
                >
                    <ConversationMessageDisplay
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

export default ConversationCurrentPage