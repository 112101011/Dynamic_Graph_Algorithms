#include "serial_code.h"

int main(){
    skip_list testing;
    testing.insert(12);
    testing.insert(10);
    testing.insert(11);
    testing.insert(3);

    testing.print();
    cout << "Representative: " << testing.find_rep() << "\n";
    cout << "Head: " << testing.head->data << "\n";

    cout << "Search 2: " << testing.search(2) << "\n";
    cout << "Search 7: " << testing.search(7) << "\n";
    cout << "Search 3: " << testing.search(3) << "\n";
    cout << "Search 10: " << testing.search(10) << "\n";
    cout << "Search 11: " << testing.search(11) << "\n";
    cout << "Search 12: " << testing.search(12) << "\n";
    cout << "Search 28: " << testing.search(28) << "\n";

    return 0;    
    

    return 0;
}