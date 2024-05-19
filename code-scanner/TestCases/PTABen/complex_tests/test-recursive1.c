struct list {
    unsigned field;
    struct list * next;
    struct list * prev;
};

extern void * malloc(unsigned);
struct list * construct (struct list * head, unsigned i) {
    if (i > 0) {
        struct list * tmp = malloc(sizeof(struct list));
        tmp->field = 0;
        tmp->next = head;
        head->prev = tmp;
        tmp->prev = 0;

        return construct (tmp, i--);


    } 
    return head;
}

struct list * walklist (struct list * input) {
    struct list * head = input;
    struct list * curr;
    while (head->prev != 0) 
        head = head->prev;

    while (head->next) {
        curr = head;
        curr->field ++;

        if (curr->field > 5) 
            head = head->next;
        else 
            head = head->prev;
        head->prev = walklist (head);
        head->next = curr;
    }
    return head;
}

int main () {
    struct list * head;


    head = malloc(sizeof(struct list));
    head->field = 0;
    head->next = 0;
    head->prev = 0;

    head = construct (head, 10);
    head = walklist(head);   
}


