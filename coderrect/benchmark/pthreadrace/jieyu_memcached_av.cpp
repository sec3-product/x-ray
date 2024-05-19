// @purpose two threads atomicity violation
// @orderViolations 1

#include <iostream>
#include <pthread.h>
#include <string.h>


struct item_t {
    int key;
    int len;
    char* data;

    item_t *next;
    item_t *prev;
};


pthread_mutex_t lock;
item_t *head;


item_t *item_get(int key) {
    pthread_mutex_lock(&lock);
    item_t *tmp = head;
    while (tmp != nullptr) {
        if (tmp->key == key)
            break;
        tmp = tmp->next;
    }
    pthread_mutex_unlock(&lock);

    return tmp;
}


void item_update(item_t *item, char* new_data, int len) {
    pthread_mutex_lock(&lock);

    if (item->len <= len) {
        memcpy(item->data, new_data, len);
        item->len = len;
    }
    else {
        item_t *new_item = new item_t();
        new_item->key = item->key;
        new_item->len = len;
        new_item->data = new char[len];
        memcpy(new_item->data, new_data, len);

        new_item->next = item->next;
        if (item->next != nullptr) {
            item->next->prev = new_item;
        }
        new_item->prev = item->prev;
        if (item->prev != nullptr) {
            item->prev->next = new_item;
        }

        if (head == item)
            head = new_item;

        delete item->data;
        delete item;
    }
    pthread_mutex_unlock(&lock);
}


void *worker(void *unused) {
    int key = rand();

    item_t *item = item_get(key);
    if (item != nullptr) {
        int len = rand();
        char *data = new char[len];

        item_update(item, data, len);
    }

    return nullptr;
}


int main() {
    pthread_t th1, th2;

    for (int i = 0; i < 10; i++) {
        item_t *item = new item_t();
        item->key = rand();
        item->len = rand();
        item->data = new char[item->len];
    }
    pthread_mutex_init(&lock, nullptr);
    pthread_create(&th1, nullptr, worker, nullptr);
    pthread_create(&th2, nullptr, worker, nullptr);

    pthread_join(th1, nullptr);
    pthread_join(th2, nullptr);
}
