export class ConversationInfo {
    id: string = "-1";
    date: string = '';

    constructor(id: string, date: string) {
        if (!id || !date) return;
        if (id) this.id = id;
        if (date) this.date = date;
    }
}