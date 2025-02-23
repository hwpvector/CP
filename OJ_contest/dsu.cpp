#include <bits/stdc++.h>
using namespace std;

class dsu {
public:
    dsu(); // default constructor, capacity is 0
    explicit dsu(int n); // constructor, capacity is n (note, the index is 0 ~ n-1)
    void insert(); // insert a new element, belong to a new set
    int find(int x); // find the index of the set that x belongs to
    void merge(int x,int y); // merge the set that x belongs to and the set that y belongs to
    bool same(int x,int y); // check if x and y belong to the same set
    int size(int x); // get the size of the set that x belongs to
    int capacity(); // get the capacity of the dsu
    void clear(); // clear the dsu, capacity is no change
    void init(int _cap); // initialize or reset the dsu, capacity is _cap
};

class undo_dsu {
public:
    undo_dsu(); // default constructor, capacity is 0
    explicit undo_dsu(int n); // constructor, capacity is n (note, the index is 0 ~ n-1)
    void insert(); // insert a new element, belong to a new set, the time is increased by 1
    int find(int x); // find the index of the set that x belongs to
    void merge(int x,int y); // merge the set that x belongs to and the set that y belongs to, the time is increased by 1
    bool same(int x,int y); // check if x and y belong to the same set
    int size(int x); // get the size of the set that x belongs to
    int capacity(); // get the capacity of the dsu
    int time(); // get the time of the dsu
    void undo(); // undo the last operation, the time is decreased by 1
    void back(int t); // undo the operations until the time is t
    void clear(); // clear the dsu, capacity is no change
    void init(int _cap); // initialize or reset the dsu, capacity is _cap
};

class dsu {
private:
    int cap;
    vector<int> fa,siz;
public:
    dsu() : cap(0) {};
    explicit dsu(int n) : fa(n),siz(n,1),cap(n) {
        iota(fa.begin(),fa.end(),0);
    }
    void insert() {
        fa.push_back(cap);
        siz.push_back(1);
        ++cap;
    }
    int find(int x) {
        return fa[x]==x?x:fa[x]=find(fa[x]);
    }
    void merge(int x,int y) {
        x=find(x),y=find(y);
        if (x==y) return ;
        if (siz[x]<siz[y]) swap(x,y);
        fa[y]=x,siz[x]+=siz[y];
    }
    bool same(int x,int y) {
        return find(x)==find(y);
    }
    int size(int x) {
        return siz[find(x)];
    }
    int capacity() {
        return cap;
    }
    void clear() {
        iota(fa.begin(),fa.end(),0);
        fill(siz.begin(),siz.end(),1);
    }
    void init(int _cap) {
        cap=_cap;
        fa=vector<int>(cap),siz=vector<int>(cap,1);
        iota(fa.begin(),fa.end(),0);
    }
};

class undo_dsu {
private:
    int cap;
    vector<int> fa,siz;
    vector<pair<int,int>> stk;
public:
    undo_dsu() : cap(0) {};
    explicit undo_dsu(int n) : fa(n),siz(n,1),cap(n) {
        iota(fa.begin(),fa.end(),0);
    }
    void insert() {
        stk.emplace_back(-1,1);
        fa.push_back(cap);
        siz.push_back(1);
        ++cap;
    }
    int find(int x) {
        return fa[x]==x?x:find(fa[x]);
    }
    void merge(int x,int y) {
        x=find(x),y=find(y);
        if (x==y) {
            stk.emplace_back(-1,2);
            return ;
        }
        if (siz[x]<siz[y]) swap(x,y);
        stk.emplace_back(x,y);
        fa[y]=x,siz[x]+=siz[y];
    }
    bool same(int x,int y) {
        return find(x)==find(y);
    }
    int size(int x) {
        return siz[find(x)];
    }
    int capacity() {
        return cap;
    }
    int time() {
        return stk.size();
    }
    void undo() {
        if (stk.empty()) return ;
        auto [x,y]=stk.back(); stk.pop_back();
        if (x==-1) {
            if (y==1) {
                fa.pop_back();
                siz.pop_back();
                --cap;
            }
        } else {
            siz[x]-=siz[y];
            fa[y]=y;
        }
    }
    void back(int t) {
        while (stk.size()>t) undo();
    }
    void clear() {
        iota(fa.begin(),fa.end(),0);
        fill(siz.begin(),siz.end(),1);
        stk.clear();
    }
    void init(int _cap) {
        cap=_cap;
        fa=vector<int>(cap),siz=vector<int>(cap,1);
        iota(fa.begin(),fa.end(),0);
        stk.clear();
    }
};
