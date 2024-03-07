#include <iostream>
#include <random>
#include <chrono>
#include <climits>
using namespace std;

// function uniform() returns a random real number from 0 to 1 from uniform probability distribution
double uniform() {
    // Using the current time as a seed for the random number generator
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}

/* Below struct node depicts structure of a node in skip list. */
struct node{
    long long int data;
    struct node* up;     
    struct node* down;
    struct node* left;          /* strictly speaking, this is a pointer to previous node */
    struct node* right;         /* strictly speaking, this is a pointer to next node */
};

// function create_node(long long int d) creates a node with data d and returns a pointer to it.
// Note: The node created is cyclic node. Since the skip list implemented here is cyclic.
struct node* create_node(long long int data){
    struct node* current = (struct node*)malloc(sizeof(struct node));
    current->data = data;
    current->up = NULL;
    current->down = NULL;
    current->left = current;
    current->right = current;
    return current;
}

// The below class skip_list provides API's for using skip list.
// The skip list implemented here is cicular.
// Here the skip list do not allow duplicates (which is okay for nodes of a graphs), similar to balanced BST.

class skip_list{
    // private:
    public:
    /*  
        In this implementation, head points to the node which has the highest height in 
        the skip list (for effecient operations).
    */
    struct node* head = NULL;

    /* rep is the pointer to poiting to minimum data element in skip list. */
    /* always rep->down = NULL that is rep lies in bottom layer */
    struct node* rep = NULL;

    /* Below helper function search_left(v)
        returns : 
                    NULL                (if no left parent of v)
                    left_parent_ptr     (if there is a left_parent) */
    struct node* search_left(struct node* );
    /* search_right is not needed, we can use the fact that skip list here is cyclic */

    /*  Below helper function connect_right()
            inserts current node in between nodeL and nodeL->right */
    void connect_left(struct node* nodeL, struct node* current);

    /*  Below helper function connect_left()
            inserts current node in between nodeR->left and nodeR */
    void connect_right(struct node* current, struct node* nodeR);

    /* Below helper function raise_up(current)
            Builds node up to the current based on probability distribution */
    void raise_up(struct node* current);

    /* Below helper function insert_helper(long long int data, struct node* trav) returns
            true    (inserts data and returns true if data is already not present in skip_list)
            false   (otherwise) */
    bool insert_helper(long long int , struct node* );

    /* Below helper function destroy_down(trav)
            deletes trav and all nodes down to trav by maintaing the pointers */
    void destroy_down(struct node* trav);

    public:
    /* Default probability is 1/2 for obvious reasons */
    double prob = 0.5;

    /* Class constructor for default probability */
    skip_list(){
    }

    /* Class constructor for custom probability */
    skip_list(double prob){
        this->prob = prob;
    }

    /*  Below (API) function  find_rep()
        returns :  
                    -1                  (if there are no elements in skip list)
                    rep                 (otherwise) */
    long long int find_rep();

    /*  Below (API) function  insert(d) :
            returns :
                    true                (if d is not present in skip list, inserts d into 
                                        skip list and returns true)
                    false               (if d is already present in skip list */
    bool insert(long long int );

    /* Below (API) search(d)
            returns :
                    true                (if d is present in skip list)
                    false               (otherwise)                             */
    bool search(long long int );

    /* Below (API) remove(long long int data)
            removes data from skip_list if data is present in skip_list
            ignores otherwise */
    void remove(long long int data);

    /* Below function (API) prints skip list contents */
    void print();
};

void skip_list::connect_left(struct node* nodeL, struct node* current){
    // seq: nodeL (current) nodeL->right
    // nodeL and current both are not NULL.

    current->left = nodeL;
    current->right = nodeL->right;
    nodeL->right->left = current;
    nodeL->right = current;

    return;
}

void skip_list::connect_right(struct node* current, struct node* nodeR){
    // seq: nodeR->left (current) nodeR->right
    /* current and nodeR are not NULL */

    current->right = nodeR;
    current->left = nodeR->left;
    nodeR->left->right = current;
    nodeR->left = current;

    return;
}


struct node* skip_list::search_left(struct node* v){
    struct node* current = v;
    while(current->up == NULL){
        current = current->left;
        if(current == v){
            /* Left parent not found */
            return NULL;
        }
    }
    /* Left parent found */
    return current->up;
}

long long int skip_list::find_rep(){
    return rep->data;
}

void skip_list::raise_up(struct node* current){
    while(uniform() < prob){
        /* up should be created and connected correctly */
        struct node* left_parent = search_left(current);
        current->up = create_node(current->data);
        current->up->down = current;
        if(left_parent != NULL) connect_left(left_parent, current->up);
        else head = current->up;
        current = current->up;
    }
    return;
}

bool skip_list::insert(long long int data){
    // The following are 3 edge cases
    // 1) head is NULL
    // 2) inserting data where data < min(all data in skip list)
    // 3) inserting data where data > max(all data in skip list)

    if(head == NULL){
        /* Empty skip list */
        struct node* current = create_node(data);
        rep = current;
        head = current;
        raise_up(current);
        return true;
    }

    if(rep->data > data){
        /* Edge case 2*/
        struct node* current = create_node(data);
        connect_right(current, rep);
        rep = current;
        raise_up(current);
        return true;
    }

    if(rep->left->data < data){
        /*Edge case 3*/
        struct node* current = create_node(data);
        connect_left(rep->left, current);
        raise_up(current);
        return true;
    }

    /* It is guranteed that : min(all elements) <= data <= max(all_elements) */
    /* So if data is not present in skip list, it is going to be inserted somewhere in between */

    struct node* trav = head;
    while(trav->left == trav && trav->down != NULL) trav = trav->down; // go down if only one element in level
    return insert_helper(data, trav);
}

bool skip_list::insert_helper(long long int data, struct node* trav){
    /* Here trav != trav->left */ 

    if(trav->data > data){
        if(trav->left->data >= data) return insert_helper(data, trav->left);
        /* trav->left->data < data. Hence data is somewhere in between trav->left, trav */
        if(trav->down == NULL){
            struct node* current = create_node(data);
            connect_right(current, trav);
            raise_up(current);
            return true;    // insertion done
        }
        return insert_helper(data, trav->down); // move down
    }
    
    if(trav->data < data) insert_helper(data, trav->right);

    /* data is already present i.e(trav->data = data) */
    return false;
}

void skip_list::print(){
    if(head == NULL){
        cout << "Empty skip list\n";
        return;
    }
    // skip_list is printed from down level to top level
    cout << "Skip list entries from down to top:\n";
    struct node* trav = rep;
    long long int level = 0;
    while(trav != NULL){
        cout << "Level " << level << " : ";
        struct node* start = trav;
        cout << trav->data << " ";
        trav = trav->right;
        while(trav != start){
            cout << trav->data << " ";
            trav = trav->right;
        }
        cout << "\n";
        if(trav->up != NULL){
            trav = trav->up;
        }else{
            trav = trav->right;
            while(trav != start && trav->up == NULL){
                trav = trav->right;
            }
            trav = trav->up;
        }
        level++;
    }
    cout << "\n";
    return;
}

bool skip_list::search(long long int data){
    if(head == NULL) return false;
    if(rep->data > data || rep->left->data < data) return false;

    struct node* trav = head;
    while(trav->left == trav && trav->down != NULL) trav = trav->down;
    while(1){
        if(trav->data == data) return true;
        if(trav->data < data){
            if(data <= trav->right->data){
                if(trav->down != NULL) trav = trav->down;
                else if(trav->right->data == data) return true;
                else return false;
            }else{
                if(trav->right->data > trav->data) trav = trav->right;
                else if(trav->down != NULL) trav = trav->down;
                else return false;
            }
        }else{
            if(trav->left->data < trav->data) trav = trav->left;
            else if(trav->down != NULL) trav = trav->down;
            else return false;
        }
    }
}


void skip_list::destroy_down(struct node* trav){
    while(trav != NULL){
        trav->left->right = trav->right;
        trav->right->left = trav->left;
        struct node* next = trav->down;
        free(trav);
        if(next != NULL) next->up = NULL;
        trav = next;
    }
    return ;
}

void skip_list::remove(long long int data){
    // Edge cases
    // 1) head = NULL
    if(head == NULL) return;

    struct node* trav = head;
    while(trav->left == trav && trav->down != NULL) trav = trav->down;
    while(1){
        if(trav->data == data){
            if(trav->left == trav){
                // skip_list has only one element, and that is being deleted
                destroy_down(head);
                head = NULL;
                rep = NULL;
                return;
            }
            // Number of elements in skip_list >= 2
            if(rep->data == data) rep = rep->right;
            if(head->data == data){
                struct node* new_head = trav->right;
                destroy_down(head);
                head = new_head;
                return;
            }
            destroy_down(trav);

            return;
        }
        if(trav->data < data){
            if(data <= trav->right->data){
                if(trav->down != NULL) trav = trav->down;
                else if(trav->right->data == data) trav = trav->right;
                else return;    // data is not present
            }else{
                if(trav->right->data > trav->data) trav = trav->right;
                else if(trav->down != NULL) trav = trav->down;
                else return;    // data is not present (here data > max(all elements in skip list))
            }
        }else{
            if(trav->left->data < trav->data) trav = trav->left;
            else if(trav->down != NULL) trav = trav->down;
            else return;        // data is not present (here data < min(all elements in skip list))
        }
    }
}