#include <iostream>

#include <random>
#include <chrono>
using namespace std;

// function uniform() returns some random real number from 0 to 1 from uniform probability distribution
double uniform() {
    // Using the current time as a seed for the random number generator
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}

// // probablity the a node should be created as up to a current node
// double prob = 0.5;

// below struct node is a structure of a node in skip list.
struct node{
    int data;
    struct node* up;
    struct node* down;
    struct node* left;
    struct node* right;
};

// function create_node(int d) creates a node with data d and returns a pointer to it.
// Note: The node created is cyclic node. Since the skip list implemented here is cyclic.
struct node* create_node(int data){
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
    private:
    /*  
        In this implementation, head points to the node which has the highest height in 
        the skip list (for effecient search).
    */
    struct node* head = NULL;

    /* transition_ptr points to the node containing maximum data in skip list. */
    struct node* transition_ptr = NULL;

    /* Below function search_left(v)
        returns : 
                    NULL                (if no left parent of v)
                    left_parent_ptr     (if there is a left_parent)     */
    struct node* search_left(struct node* );
    /* search_right is not needed because skip list here is cyclic :) */

    public:
    /* Default probability is 1/2 for obvious reasons */
    double prob = 0.5 

    /* Class constructor for default probability */
    skip_list(){
    }

    /* Class constructor for custom probability */
    skip_list(double prob){
        this->prob = prob;
    }

    /*  Below function find_rep()
        returns :  
                    -1                  (if there are no elements in skip list)
                    rep                 (representative to be choosen ?????????????????)    */
    int find_rep();

    /*  Below function insert(d) :
            returns :
                    true                (if d is not present in skip list, inserts d into 
                                        skip list and returns true)
                    false               (if d is already present in skip list */
    void insert(int data){
        if(head == NULL){
            struct node* current = create_node(data);

            if(uniform() < p){
                // up should be created
                current->up = create_node(data);
                current->up->down = current;
                current = current->up;
            }
            head = current;
            return;
        }
        if(head->data == data){
            cout << data << " already present\n";
            return;
        }
        struct node* trav = head;
        while(trav->down != NULL && trav->left == trav) trav = trav->down;
        if(trav->left == trav){
            // only one element is present in skip list
            // No need to care about in which direction we insert it to the head (since it is cyclic and only element is present)
            // 4 pointers need to be updated
            struct node* current = create_node(data);
            trav->left = current;
            trav->right = current
            current->right = trav
            current->left = trav

            if(data > head->data) transition_ptr = current;
            else transition_ptr = head;
            
            while(uniform() < p){
                struct node* left_parent = search_left(current);
                current->up = create_node(data);
                current->up->down = current;
                if(left_parent != NULL){
                    // seq can be : left->parent current->up left->parent->right
                    current->up->right = left_parent->right;
                    current->up->left = left_parent;
                    left_parent->right->left = current->up;
                    left_parent->right = current->up;
                }else head = current->up;
                current = crrent->up;
            }
        }
        insert_helper(data, head);
    }

    void connect(struct node* nodeL, struct node* current){
        // seq: nodeL (current) nodeL->right
        current->left = nodeL;
        current->right = nodeL->right;
        nodeL->right->left = current;
        nodeL->right = current;
    }

    void introduce_new_node(struct node* nodeL, int data){
        struct node* current = create_node(data);
        connect(nodeL, current);

        while(uniform() < p){
            struct node* left_parent = search_left(current);
            current->up = create_node(data);
            current->up->down = current;
            if(left_parent != NULL){
                connect(left->parent, current->up);
            }else head = current->up;
            current = current->up;
        }
        return;
    }

    void insert_helper(int data, struct node* trav){
        // Hadling left suffices, because of cyclicity.
        // Here trav is such that trav->left != trav implies trav->right != trav (again because of cyclicity) i.e thare are at least two elements in the skip list
        if(trav->data == data){
            cout << data << " already present\n";
            return;
        }

        if(trav->data > data){
            if(trav->left->data > data){
                if(trav->left->data == head->data){
                    // edge case: data < min(all data in current skip list)

                }else{
                    insert_helper(data, trav->left);
                }
            }else if(trav->left->data == data){
                cout << data << " already present\n";
                return;
            }else{
                if(trav->down == NULL){
                    // insertion done
                    introduce_new_node(trav->left, data);
                    return;
                }else insert_helper(data, trav->down); // move down
            }
        }else{
            if(trav->right == head->data){
                // edge case: data > max(all data in current skip list)

            }
            insert_helper(data, trav->right);
        }
    }

    bool search(int data){

    }

    void print(){

    }
};

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

int skip_list::find_rep(){
    return head->data;
}


