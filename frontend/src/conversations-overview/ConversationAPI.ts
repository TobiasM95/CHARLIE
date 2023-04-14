import { apiURL } from "../Settings/Constants";

const baseUrl = apiURL + '/newconversation';
const convInitUrl = baseUrl + "/init"
const convEndUrl = baseUrl + "/end"
const convPostTextUrl = baseUrl + "/posttext"

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

interface IConversationInitConfig {
    name: string;
    gender: string;
    language: string;
    memory_size: number;
}

const conversationAPI = {

    init(config: IConversationInitConfig = { name: "John", gender: "female", language: "EN-US", memory_size: 3 }) {
        const options: RequestInit = {
            method: "POST",
            body: JSON.stringify(config),
            headers: {
                Accept: "application/json, text/plain",
                "Content-Type": "application/json;charset=UTF-8"
            }
        };
        return fetch(`${convInitUrl}`, options)
            .then(checkStatus)
            .catch((error: TypeError) => {
                console.log('log client error ' + error);
                throw new Error(
                    'There was an error retrieving the conversations overview. Please try again.'
                );
            });
    },

    end() {
        const options: RequestInit = {
            method: "POST",
            body: JSON.stringify(""),
            headers: {
                Accept: "application/json, text/plain",
                "Content-Type": "application/json;charset=UTF-8"
            }
        };
        return fetch(`${convEndUrl}`, options)
            .then(checkStatus)
            .catch((error: TypeError) => {
                console.log('log client error ' + error);
                throw new Error(
                    'There was an error retrieving the conversations overview. Please try again.'
                );
            });
    },

    posttext(message: string) {
        const options: RequestInit = {
            method: "POST",
            body: JSON.stringify({ "text": message }),
            headers: {
                Accept: "application/json, text/plain",
                "Content-Type": "application/json;charset=UTF-8"
            }
        };
        return fetch(`${convPostTextUrl}`, options)
            .then(checkStatus)
            .catch((error: TypeError) => {
                console.log('log client error ' + error);
                throw new Error(
                    'There was an error retrieving the conversations overview. Please try again.'
                );
            });
    },
}

export { conversationAPI };