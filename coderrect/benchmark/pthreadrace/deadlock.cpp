// @purpose use db 2-phase lock to test deadlock handling
// @dataRaces 0
// @deadlocks 1
// @orderViolations 0
// @misMatchedAPI 0

#include <iostream>
#include <map>
#include <pthread.h>

class Account
{
private:
    int id_;
    int balance_;
    pthread_mutex_t mutex_;

public:
    Account(int id, int balance) : id_(id), balance_(balance)
    {
        pthread_mutex_init(&mutex_, NULL);
    }

    void Lock()
    {
        pthread_mutex_lock(&mutex_);
    }

    void Unlock()
    {
        pthread_mutex_unlock(&mutex_);
    }

    int GetId()
    {
        return id_;
    }

    int GetBalance()
    {
        return balance_;
    }

    void SetBalance(int newBalance)
    {
        std::cout << "User " << id_ << "'s balance is now set to " << newBalance << "\n";
        balance_ = newBalance;
    }
};

static std::map<int, Account *> accounts_;
static pthread_barrier_t barrier_;

static pthread_cond_t txn_cond_;
static pthread_mutex_t txn_mutex_;
static int _finished_txn = 0;

struct TransferArgs
{
    int srcAcctId;
    int dstAcctId;
    int amount;

    TransferArgs(int srcAcctId, int dstAcctid, int amount) : srcAcctId(srcAcctId),
                                                             dstAcctId(dstAcctid),
                                                             amount(amount) {}
};

static void DownCountTxn()
{
    pthread_mutex_lock(&txn_mutex_);
    _finished_txn++;
    pthread_cond_broadcast(&txn_cond_);
    pthread_mutex_unlock(&txn_mutex_);
}

// I use a 2-phase lock alogrithm here
static void *Transfer(void *args)
{
    pthread_barrier_wait(&barrier_);
    TransferArgs *myargs = (TransferArgs *)args;

    Account *srcAcct = accounts_[myargs->srcAcctId];
    Account *dstAcct = accounts_[myargs->dstAcctId];

    srcAcct->Lock();
    int balance = srcAcct->GetBalance();
    if (balance < myargs->amount)
    {
        std::cout << "The balance isn't enough" << std::endl;
        srcAcct->Unlock();
        DownCountTxn();
        return nullptr;
    }
    srcAcct->SetBalance(balance - myargs->amount);

    dstAcct->Lock();
    dstAcct->SetBalance(dstAcct->GetBalance() + myargs->amount);
    dstAcct->Unlock();

    srcAcct->Unlock();

    DownCountTxn();
    return nullptr;
}

/**
 * A simple way to prevent the deadlock is to acquire lock of the smaller-id account
 */
static pthread_t TransferBalance(int srcAcctId, int dstAcctId, int amount)
{
    pthread_t thr;

    TransferArgs *args = new TransferArgs(srcAcctId, dstAcctId, amount);
    std::cout << "Transfer " << args->amount
              << " from user " << accounts_[args->srcAcctId]->GetId()
              << " to user " << accounts_[args->dstAcctId]->GetId()
              << "\n";
    pthread_create(&thr, NULL, Transfer, (void *)args);
    return thr;
}

int main(int argc, char **argv)
{
    pthread_barrier_init(&barrier_, NULL, 2);
    pthread_cond_init(&txn_cond_, NULL);
    pthread_mutex_init(&txn_mutex_, NULL);

    accounts_.emplace(1, new Account(1, 1000));
    accounts_.emplace(2, new Account(2, 2000));
    accounts_.emplace(3, new Account(3, 3000));

    // acquiring lock from account1 -> account2
    pthread_t t1 = TransferBalance(1, 2, 200);
    // DEADLOCK!
    // acquiring lock from account2 -> account1
    pthread_t t2 = TransferBalance(2, 1, 300);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    pthread_mutex_lock(&txn_mutex_);
    while (_finished_txn != 2)
        pthread_cond_wait(&txn_cond_, &txn_mutex_);
    pthread_mutex_unlock(&txn_mutex_);
}