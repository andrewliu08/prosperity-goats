#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <fstream>
using namespace std;
using namespace __gnu_pbds;
//typedef tree<int, null_type,less_equal<int>, rb_tree_tag,tree_order_statistics_node_update> os;

typedef long long ll;

typedef long double ld;
typedef pair<int,int> pii;
bool DEBUG = 1;
#define log2(x) ((x==0)? 0:63 - __builtin_clzll(x))
#define pb push_back
#define ms(x, y) memset(x, y, sizeof x)
#define popcount __builtin_popcount
#define all(v) v.begin(), v.end()

const int inf=0x3f3f3f3f; const ll LLINF=0x3f3f3f3f3f3f3f3f;
inline ll gcd(ll a, ll b) {return b == 0 ? a : gcd(b, a % b);}
inline ll lcm(ll a, ll b) { return a / gcd(a, b) * b;}

#define deb(...) logger(#__VA_ARGS__, __VA_ARGS__)
template <typename... Args>
void logger(string vals, Args &&...values){
    if (DEBUG){
        cout << vals << " = ";
        string delim = "";
        (..., (cout << delim << values, delim = ", "));
        cout << endl;
    }
}

const ll mod = 1e9+7;
struct tri{
    ll x,y,z;
    bool operator<(const tri &one)const{
        if(x==one.x) return y<one.y;
        return x<one.x;
    }//pqs are backwards
};
ll fpow(ll a, ll b){
    if (b == 0) return 1;
    ll res = fpow(a, b / 2)%mod;
    if (b % 2) return ((res * res) * a) %mod;
    else return (res * res) %mod;
}
const int maxn = 10;
ld adj[4][4] = {
    {(ld)1,(ld)0.48,(ld)1.52,(ld)0.71},
    {(ld)2.05,(ld)1,(ld)3.26,(ld)1.56},
    {(ld)0.64,(ld)0.3,(ld)1,(ld)0.46},
    {(ld)1.41,(ld)0.61,(ld)2.08,(ld)1}
    };
ld ans = 1;
int best_edge = 0;
vector<pii> edge_order;
vector<vector<pii>> all_combos;
vector<ld> all_combos_results;
void go(int cur, int edge_cnt, ld trade,vector<pii> &v){
    if(edge_cnt > 4) return;
    if(cur == 3) {
        if(ans < trade){
            ans = trade;
            edge_order = v;
        }
        all_combos.pb(v);
        all_combos_results.pb(trade);
    }
    for(int i =0; i<4; i++){
        vector<pii> t;
        for(pii e: v) t.pb(e);
        t.pb({cur,i});
        go(i,edge_cnt+1, trade*adj[cur][i],t);
    }
}
signed main(){
    ios_base::sync_with_stdio(0); cin.tie(0); cout.tie(0);
    //pizza, wasabi, snowball, shells

    vector<pii> v;
    go(3,0,1,v);
    cout<<fixed<<setprecision(10)<<ans<<"\n";
    for(pii e: edge_order){
        cout<<e.first<<" "<<e.second<<"\n";
    }
    for(int i =0; i<all_combos.size(); i++){
        vector<pii> edges = all_combos[i];
        ld result = all_combos_results[i];
        for(pii f: edges){
            cout<<"("<<f.first<<", "<<f.second<<") ";
        }
        cout<<fixed<<setprecision(10)<<result<<"\n";
    }
}


