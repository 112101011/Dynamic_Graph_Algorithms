#include <bits/stdc++.h>
using namespace std;

int main(){
    long long int num_operations = 0;
    cin >> num_operations;
    set <long long int> s;
    while(num_operations--){
        char op; cin >> op;
        long long int num; 
        cin >> num;
        if(op == '+'){
            // cout << "+ " << num << "\n";
            s.insert(num);
        }else if(op == '-'){
            s.erase(num);
        }else{
            cout << s.count(num) << "\n";
        }
    }
    return 0;
}
