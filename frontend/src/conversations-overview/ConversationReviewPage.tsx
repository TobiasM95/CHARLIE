import Grid from "@mui/material/Grid";
import ConversationMessageDisplay from "./ConversationMessageDisplay";
import { ConversationContent } from "./ConversationContent";

interface IConversationReviewPageProps {
    conversationContent: ConversationContent | undefined;
    showSystemMessages: boolean;
    toolbarHeight: string;
    gender: string;
}

function ConversationReviewPage({ conversationContent, showSystemMessages, toolbarHeight, gender }: IConversationReviewPageProps) {
    return (
        <Grid
            container
            item
            xs={8}
            rowSpacing={1}
            sx={{
                maxHeight: `calc(100vh - ${toolbarHeight})`,
                marginTop: toolbarHeight,
                p: 2,
            }}
        >
            {conversationContent?.messages.filter((message) => (message.mode !== 'SYSTEM' || showSystemMessages)).map((message, id) => (
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

export default ConversationReviewPage