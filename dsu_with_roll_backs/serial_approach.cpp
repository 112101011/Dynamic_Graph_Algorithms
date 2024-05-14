#include <iostream>
#include <stack>
#include <vector>
#include <map>
using namespace std;  

#define pii             pair <int, int>  
#define mk              make_pair  

static const int maxn = 3e5 + 5;
vector <pii> Tree[maxn * 4];  

// update(1, 1, q, l, r, mk(a, b));
// a and b is the 
// i and j is the region where edge {p.fi, p.se} is active
void update(int node, int a, int b, int i, int j, pii p){
      if (a > j or b < i) return;  
      if (a >= i and b <= j){
            Tree[node].push_back(p);
            return;
      }
      int lchild = node * 2;  
      int rchild = 2*node + 1;  
      int mid = (a + b) / 2;  
      update(lchild, a, mid, i, j, p);  
      update(rchild, mid + 1, b, i, j, p);  
}  

int par[maxn];  
int sz[maxn];  
stack <int> st;  
int component;  

// Making set of n nodes with itself as parent
void makeSet(){  
      for (int i = 1; i < maxn; i++){  
            par[i] = i;  
            sz[i] = 1;        // updating auxillary data structure to store size
      }  
}

// finding representative
// note: no path compression
int findRep(int r){  
      if (r == par[r]) return r;  
      return findRep(par[r]);  
}  

// merging two components
// union by size
void marge(int u, int v)  
{  
      int p = findRep(u);  
      int q = findRep(v);  
      if (p == q) return;  
      component--;  
      if (sz[p] > sz[q]) swap(p, q);  
      par[p] = q;  
      sz[q] += sz[p];  
      st.push(p);  
}  

// rolling back the top of the stack
void rollback(int moment){
      while (st.size() > moment){  
            int curr = st.top(); st.pop();  
            sz[ par[curr] ] -= sz[curr];  
            par[curr] = curr;  
            ++component;  
      }
}

int ans[maxn];  

// perforing dfs
void dfs(int node, int a, int b) {
      if (a > b) return;  
      int moment = st.size();  
      for (pii p : Tree[node]) marge(p.first, p.second);
      if (a == b){  
            ans[a] = component;  
      }else{  
            int lchild = node * 2;  
            int rchild = lchild + 1;  
            int mid = (a + b) / 2;  
            dfs(lchild, a, mid);  
            dfs(rchild, mid + 1, b);  
      }
      // rolling all vertices added to 
      // the top of the stack
      rollback(moment); 
}  

map <pii, int> in;      
vector <int> queries;  

int main() {
      // n is the number of vertices
      // q is number of quries
      int n, q;  
      cin >> n >> q;

      for (int i = 1; i <= q; i++){  
            string type;  
            cin >> type;  
            if (type == "?"){  
                  queries.push_back(i);  
            }else{  
                  int a, b;  
                  cin >> a >> b;  
                  if (a > b) swap(a, b);  
                  if (type == "+"){  
                        in[mk(a, b)] = i;  
                  }else{  
                        int l = in[mk(a, b)];  
                        int r = i - 1;
                        // edge a and b is active from l to r (inclusive)
                        update(1, 1, q, l, r, mk(a, b));  
                        in.erase(mk(a, b));  
                  }  
            }  
      }

      // some edges are left active even after q quries
      // those needs to processed (below loop)
      for (auto it : in){  
            int l = it.second;  
            int r = q;  
            int a = it.first.first;  
            int b = it.first.second;  
            update(1, 1, q, l, r, mk(a, b));  
      }  
      makeSet();

      // initially the number of componenets is n
      component = n;

      // performing dfs on segement tree
      dfs(1, 1, q);

      // printing the number of componenets for each query
      for (int p : queries) cout << ans[p] << '\n';  
}