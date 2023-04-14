import { apiURL } from "../Settings/Constants";
import { SHA256 } from "crypto-js"

const baseUrl = apiURL + '/users';
const convAccessUrl = baseUrl + "/access/"
const convRequestAccessUrl = baseUrl + "/requestaccess"

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

function extractAccessStatus(data: any): boolean {
    return data["access"];
}

const userAPI = {

    checkAccess(email_cleaned: string): Promise<boolean> {
        const emailHash = SHA256(email_cleaned).toString();
        return fetch(`${convAccessUrl}${emailHash}`)
            .then(checkStatus)
            .then(parseJSON)
            .then(extractAccessStatus)
            .catch((error: TypeError) => {
                console.log('log client error ' + error);
                throw new Error(
                    'There was an error retrieving the conversations overview. Please try again.'
                );
            });
    },

    requestAccess(emailCleaned: string): Promise<object> {
        const emailHash = SHA256(emailCleaned).toString();
        const options: RequestInit = {
            method: "POST",
            body: JSON.stringify({ "email-hash": emailHash }),
            headers: {
                Accept: "application/json, text/plain",
                "Content-Type": "application/json;charset=UTF-8"
            }
        };
        return fetch(`${convRequestAccessUrl}`, options)
            .then(checkStatus)
            .catch((error: TypeError) => {
                console.log('log client error ' + error);
                throw new Error(
                    'There was an error retrieving the conversations overview. Please try again.'
                );
            });
    }
}

export { userAPI };