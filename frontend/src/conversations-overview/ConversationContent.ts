export type Mode = "SYSTEM" | "CONVERSATION"


interface IConversationMessageJson {
    index: number;
    message: string;
    mode: string;
    speaker: string;
    timestamp: string;
}

interface IConversationMetadataJson {
    messageCount: number;
    timestamp: string;
}

export interface IConversationContentJson {
    lineInformation: IConversationMessageJson[];
    metaData: IConversationMetadataJson;
}

export class ConversationMessage {
    index: number = 0;
    message: string = "";
    mode: Mode = "SYSTEM";
    speaker: string = "system";
    timestamp: Date = new Date('1999-12-31 00:00:00');

    constructor(iConversationMessageJson: IConversationMessageJson) {
        this.index = iConversationMessageJson.index;
        this.message = iConversationMessageJson.message;
        this.mode = iConversationMessageJson.mode === "SYSTEM" ? "SYSTEM" : "CONVERSATION";
        this.speaker = iConversationMessageJson.speaker;
        this.timestamp = new Date(iConversationMessageJson.timestamp);
    }
}

export class ConversationContent {
    messageCount: number = 0;
    timestamp: Date = new Date('1999-12-31 00:00:00');
    messages: ConversationMessage[] = []

    constructor(messageCount?: number, timestamp?: Date, messages?: ConversationMessage[]) {
        if (messageCount) this.messageCount = messageCount;
        if (timestamp) this.timestamp = timestamp;
        if (messages) this.messages = messages
    }
}