#include "serial_code.h"
using namespace std;

int main(){
    long long int num_operations = 0;
    cin >> num_operations;
    skip_list testing;
    while(num_operations--){
        char op; cin >> op;
        long long int num; 
        cin >> num;
        if(op == '+'){
//             cout << "+ " << num << "\n";
            testing.insert(num);
        }else if(op == '-'){
   //         cout << "- " << num << "\n";
            testing.remove(num);
        }else{
     //       cout << "? " << num << "\n";
            cout << testing.search(num) << "\n";
        }
    }
    return 0;
}
