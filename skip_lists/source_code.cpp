#include "serial_code.h"

int main(){
    skip_list testing;
    
    testing.insert(12);
    testing.print();
    testing.insert(10);
    testing.print();
    testing.insert(11);
    testing.print();
    testing.insert(3);
    testing.print();

    
    // cout << "Representative: " << testing.find_rep() << "\n";
    // cout << "Head: " << testing.head->data << "\n";

    cout << "Search 2: " << testing.search(2) << "\n";
    cout << "Search 7: " << testing.search(7) << "\n";
    cout << "Search 3: " << testing.search(3) << "\n";
    cout << "Search 10: " << testing.search(10) << "\n";
    cout << "Search 11: " << testing.search(11) << "\n";
    cout << "Search 12: " << testing.search(12) << "\n";
    cout << "Search 28: " << testing.search(28) << "\n";
    cout << "\n";

    testing.remove(11);
    cout << "After removing 11\n";
    testing.print();
    // return 0;
    
    testing.remove(3);
    cout << "After removing 3\n";
    testing.print();
    
    testing.remove(12);
    cout << "After removing 12\n";
    testing.print();
    
    testing.remove(10);
    cout << "After removing 10\n";
    testing.print();
    return 0;
}