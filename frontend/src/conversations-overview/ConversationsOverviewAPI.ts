import { ConversationContent, ConversationMessage, IConversationContentJson } from './ConversationContent';
import { ConversationInfo } from './ConversationInfo';
import { apiURL } from '../Settings/Constants';

const baseUrl = apiURL;
const url = `${baseUrl}`;
const convInfoUrl = url + "/conversations/"
const convContentUrl = url + "/conversation/"

function translateStatusToErrorMessage(status: number) {
  switch (status) {
    case 401:
      return 'Please login again.';
    case 403:
      return 'You do not have permission to view the conversations overview.';
    default:
      return 'There was an error retrieving the conversations overview. Please try again.';
  }
}

function checkStatus(response: any) {
  if (response.ok) {
    return response;
  } else {
    const httpErrorInfo = {
      status: response.status,
      statusText: response.statusText,
      url: response.url,
    };
    console.log(`log server http error: ${JSON.stringify(httpErrorInfo)}`);

    let errorMessage = translateStatusToErrorMessage(httpErrorInfo.status);
    throw new Error(errorMessage);
  }
}

function parseJSON(response: Response) {
  return response.json();
}

function cleanConversationNameString(raw: string): string {
  let rawTimestamp = raw.split("_")[1].split(".")[0]
  return rawTimestamp.split("T")[0] + "T" + rawTimestamp.split("T")[1].replace("-", ":").replace("-", ":")
}

function convertToConversationInfoModels(data: { [id: string]: string }): ConversationInfo[] {
  let conversationInfos: ConversationInfo[] = [];
  for (let key in data) {
    let value = data[key];
    let dateString: string = cleanConversationNameString(value)
    conversationInfos.push(convertToProjectModel(key, dateString))
  }
  return conversationInfos;
}

function convertToConversationContentModel(data: IConversationContentJson): ConversationContent {
  let messages: ConversationMessage[] = [];
  for (let line of data.lineInformation) {
    messages.push(new ConversationMessage(line))
  }
  console.log(data.metaData.timestamp)
  return new ConversationContent(data.metaData.messageCount, new Date(data.metaData.timestamp), messages);
}

function convertToProjectModel(id: string, date: string): ConversationInfo {
  return new ConversationInfo(id, date);
}

const conversationsOverviewAPI = {
  get(userUID: string) {
    return fetch(`${convInfoUrl}${userUID}`)
      .then(checkStatus)
      .then(parseJSON)
      .then(convertToConversationInfoModels)
      .catch((error: TypeError) => {
        console.log('log client error ' + error);
        throw new Error(
          'There was an error retrieving the conversations overview. Please try again.'
        );
      });
  },
};

const conversationContentAPI = {
  get(userUID: string, index: number) {
    return fetch(`${convContentUrl}${userUID}/${index}`)
      .then(checkStatus)
      .then(parseJSON)
      .then(convertToConversationContentModel)
      .catch((error: TypeError) => {
        console.log('log client error ' + error);
        throw new Error(
          'There was an error retrieving the conversations overview. Please try again.'
        );
      });
  },
}

export { conversationsOverviewAPI, conversationContentAPI };