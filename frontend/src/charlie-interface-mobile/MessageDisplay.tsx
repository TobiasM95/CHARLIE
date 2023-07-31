import Typography from "@mui/material/Typography";
import { ConversationMessage } from "../datastructs/ConversationContent";
import Box from "@mui/material/Box";
import Grid from "@mui/material/Grid";
import Avatar from "@mui/material/Avatar";
import React from "react";
import moment from "moment";
import { dateTimeFullFormat } from "../Settings/Constants";

export interface IConversationMessageProps {
  id: number;
  message: ConversationMessage;
  gender: string;
}

const Timestamp = (messageTimestamp: Date, speaker: string | undefined = undefined) => {
  let timestampMessage = speaker === undefined ? "" : speaker + ", ";
  timestampMessage += moment(messageTimestamp).format(dateTimeFullFormat);
  return (
    <Typography
      variant="overline"
      display="block"
      align="right"
      sx={{ marginRight: 2 }}
    >
      <>{timestampMessage}</>
    </Typography>
  )
};

const Message = (bgcolor: string, message: ConversationMessage, variant: any = "body1") => (
  <Box
    sx={{
      border: "1px solid",
      borderRadius: 0,
      borderColor: "black",
      backgroundColor: bgcolor,
      px: 1
    }}
  >
    <Typography
      variant={variant}
      display="block"
      align="left"
      sx={{ p: 1 }}
      color={"black"}
    >
      {message.message}
    </Typography>
  </Box>
);

const MessageDisplay: React.FunctionComponent<IConversationMessageProps> = ({ id, message, gender }: IConversationMessageProps) => {
  if (message.mode === "SYSTEM") {
    return (
      <Grid
        item
        container
        xs={12}
        justifyContent="center"
      >
        <Grid
          item
          xs={12}
        >
          {Message("lightgray", message, "overline")}
          {Timestamp(message.timestamp, message.speaker)}
        </Grid>
      </Grid>
    );
  } else {
    if (message.speaker === "Charlie") {
      return (
        <Grid
          item
          container
          xs={12}
          justifyContent="flex-end"
        >
          <Grid
            item
            xs={11}
          >
            <Grid container>
              <Grid
                item
                xs={11}
              >
                {Message("lightsalmon", message)}
                {Timestamp(message.timestamp)}
              </Grid>
              <Grid
                item
                xs={1}
              >
                <Avatar
                  src={gender === "female" ? "./images/charlieAvatarFemale.png" : "./images/charlieAvatarMale.png"}
                  variant="rounded"
                  sx={{
                    ml: 0.5,
                  }}
                />
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      );
    } else {
      return (
        <Grid
          item
          container
          xs={12}
          justifyContent="flex-start"
        >
          <Grid
            item
            container
            xs={11}
          >
            <Grid
              item
              xs={1}
              container
              justifyContent="flex-end"
            >
              <Avatar variant="rounded">{message.speaker.at(0)}</Avatar>
            </Grid>
            <Grid
              item
              xs={11}
            >
              {Message("lightblue", message)}
              {Timestamp(message.timestamp)}
            </Grid>
          </Grid>
        </Grid>
      );
    }
  }
};

export default MessageDisplay;
