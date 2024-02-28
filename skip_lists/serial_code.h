#include <iostream>
#include <random>
#include <chrono>
using namespace std;

double uniform() {
    // Use the current time as a seed for the random number generator
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed);
    uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}

// probablity the a node should be created as up to a current node
double prob = 0.5;

struct node{
    int data;
    struct node* up;
    struct node* down;
    struct node* left;
    struct node* right;
};

struct node* create_node(int data){
    struct node* current = (struct node*)malloc(sizeof(struct node));
    current->data = data;
    current->up = NULL;
    current->down = NULL;
    current->left = current;
    current->right = current;
    return current;
}



class skip_list{
    // the skip lists here is circular
    // do not allow duplicates (which is okay for nodes of a graphs) similar to balanced BST

    private:
    // In this representation, head corresponds to the node which has the highest height in the skip list(for effecient search) 
    // Alternative choice: head can be also choosen to be the minimum data element.
    struct node* head = NULL;

    // storing additional pointer transition_ptr along with head saves logn time in search
    // transition_ptr infact points to node containing maximum data in skip list.
    struct node* transition_ptr = NULL; 

    public:
    double prob = 0.5
    skip_list(){
        // default prob 0.5
    }
    skip_list(double prob){
        this->prob = prob;
    }

    struct node* search_left(struct node* v){
        struct node* current = v;
        while(current->up == NULL){
            current = current->left;
            if(current == v) return NULL;
        }
        return current->up;
    }

    // search_right is not needed because it is cyclic :)
    int find_rep(){
        return head->data;
    }

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