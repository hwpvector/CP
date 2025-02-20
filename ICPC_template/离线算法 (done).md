
# 离线算法

## 离散化

```cpp
const int N=1e5+10;
int n,t[N],a[N];

void solve() {
    memcpy(t,a,sizeof(a)); // 多测别 memcpy
    sort(t+1,t+n+1);
    int m=unique(t+1,t+n+1)-t-1;
    for(int i=1;i<=n;i++)
        a[i]=lower_bound(t+1,t+m+1,a[i])-t;
}
```

## CDQ 分治

```cpp
const int N=1e5+10,M=1e6+10;
struct node{int x,y,z,w,ans; vector<int> id;}a[N],A[N];
// x,y,z 三维; a[i].w 出现次数; id 原编号
// a[i].ans a[j].x/y/z<=a[i].x/y/z 的数量
// 注意没有加上重复次数，真实答案应为 a[i].ans+a[i].w-1
int n,k,f[M],ans[N]; // n 元素个数; k 值域大小
int lowbit(int x) {return x&-x;}
void add(int x,int v) {while (x<=k)f[x]+=v,x+=lowbit(x);}
int sum(int x) {int ans=0;while (x>0) ans+=f[x],x-=lowbit(x);return ans;}
bool cmp1(const node &u,const node &v) {return u.x==v.x?u.y==v.y?u.z<v.z:u.y<v.y:u.x<v.x;}
bool cmp2(const node &u,const node &v) {return u.y==v.y?u.z<v.z:u.y<v.y;}

void CDQ(int l,int r) {
    if (l==r) return ;
    int mid=(l+r)>>1;
    CDQ(l,mid); CDQ(mid+1,r);
    sort(a+l,a+mid+1,cmp2);
    sort(a+mid+1,a+r+1,cmp2);
    int j=l;
    for (int i=mid+1;i<=r;i++){
        while (j<=mid&&a[j].y<=a[i].y) add(a[j].z,a[j].w),j++;
        a[i].ans+=sum(a[i].z);
    }
    for (int i=l;i<j;i++) add(a[i].z,-a[i].w);
}
void solve(){
    // 可以先对 A 离散化
    sort(A+1,A+n+1,cmp1);
    int cnt=0; vector<int> id; // 去重
    for (int i=1,j=1;i<=n;i++,j++) {
        id.push_back(A[i].id[0]);
        if (A[i].x!=A[i+1].x||A[i].y!=A[i+1].y||A[i].z!=A[i+1].z)
            a[++cnt]=A[i],a[cnt].w=j,a[i].id=id,j=0,id.clear();
    }
    CDQ(1,cnt);
    for (int i=1;i<=cnt;i++) {
        for (auto j:a[i].id) ans[j]=a[i].ans+a[i].w-1;
    }
    for (int i=1;i<=n;i++) cout<<ans[i]<<' ';
}
```

## 莫队


### 普通莫队

```cpp
const int N = 50005;
int n,m,siz,nw,c[N],ans[N];

struct query {
    int l,r,id;
    bool operator<(const query &x) const {
        if (l/siz!=x.l/siz) return l<x.l;
        return (l/siz)&1?r<x.r:r>x.r;
    }
}a[N];
void add(int i) {}
void del(int i) {}

void solve() {
    siz=sqrt(n);
    sort(a+1,a+m+1);
    for (int i=0,l=1,r=0;i<m;i++) {
        while (l>a[i].l) add(--l);
        while (r<a[i].r) add(++r);
        while (l<a[i].l) del(l++);
        while (r>a[i].r) del(r--);
        ans[a[i].id]=nw;
    }
}
```


### 带修莫队

块长 $O(n^{\frac{2}{3}})$，复杂度 $O(n^{\frac{5}{3}})$

```cpp
const int N = 50005;
int n,m,siz,nw,c[N],ans[N];

struct query {
    int l,r,t;
    bool operator<(const query &x) const {
        if (l/siz!=x.l/siz) return l<x.l;
        if (r/siz!=x.r/siz) return r/siz<x.r/siz;
        else return t<x.t;
    }
}a[N];
struct modify{int pos,val;}r[N];
void add(int i) {}
void del(int i) {}
void solve() {
    siz=pow(n,2.0/3);
    sort(a+1,a+m+1);
    for (int i=0,l=1,r=0;i<m;i++) {
        while (l>a[i].l) add(--l);
        while (r<a[i].r) add(++r);
        while (l<a[i].l) del(l++);
        while (r>a[i].r) del(r--);
        while (t<a[i].t) {
            ++t;
            if (r[t].pos>=l&&r[t].pos<=r) // ... modify
            swap(r[t].val,c[r[t].pos]);
        }
        while (t>a[i].t) {
            if (r[t].pos>=l&&r[t].pos<=r) // ... modify
            swap(r[t].val,c[r[t].pos]); --t;
        }
        ans[a[i].id]=nw;
    }
}
```


### 二维莫队

记询问次数为 $q$，当前矩阵的左上角坐标为 $(x_1,\ y_1)$，右下角坐标为 $(x_2,\ y_2)$，取块长为 $B=n\cdot q^{-\frac 14}$ 即可。

注意这样计算 $B$ 的结果 可能为 $0$，注意特判。

最终，计算部分时间复杂度是 $\Theta(n^2\cdot q^{\frac 34})$，加上对询问的排序过程，总时间复杂度为 $\Theta(n^2\cdot q^{\frac 34}+q\log q)$。

### 树上莫队

dfs 一棵树，然后如果 dfs 到 x 点，就 push_back(x)，dfs 完 x 点，就直接 ``push_back(-x)``，然后我们在挪动指针的时候，

- 新加入的值是 x ---> ``add(x)``
- 新加入的值是 - x ---> ``del(x)``
- 新删除的值是 x ---> ``del(x)``
- 新删除的值是 - x ---> ``add(x)``

这样的话，我们就把一棵树处理成了序列。在序列上分类讨论做莫队

### 回滚莫队

在只有增加不可实现或者只有删除不可实现的时候，就可以使用回滚莫队在 $O(n \sqrt m)$ 的时间内解决问题

对原序列进行分块，对询问按以左端点所属块编号升序为第一关键字，右端点升序为第二关键字的方式排序。

按顺序处理询问：

- 如果询问左端点所属块 $B$ 和上一个询问左端点所属块的不同，那么将莫队区间的左端点初始化为 $B$ 的右端点加 $1$, 将莫队区间的右端点初始化为 B 的右端点；
- 如果询问的左右端点所属的块相同，那么直接扫描区间回答询问；
- 如果询问的左右端点所属的块不同：
  - 如果询问的右端点大于莫队区间的右端点，那么不断扩展右端点直至莫队区间的右端点等于询问的右端点；
  - 不断扩展莫队区间的左端点直至莫队区间的左端点等于询问的左端点；
  - 回答询问；
  - 撤销莫队区间左端点的改动，使莫队区间的左端点回滚到 $B$ 的右端点加 $1$。
  
序列长度 $n$，询问 $m$，取块长 $\frac{n}{\sqrt{m}}$，复杂度 $O(n\sqrt{m})$

```cpp
int n, q, x[N], t[N], m;
struct Query {int l, r, id;} Q[N];
int pos[N], L[N], R[N], sz, tot, cnt[N], __cnt[N], ans[N];
bool cmp(const Query& A, const Query& B) {
    if (pos[A.l] == pos[B.l]) return A.r < B.r;
    return pos[A.l] < pos[B.l];
}
void build() {
    sz = sqrt(n);
    tot = n / sz;
    for (int i = 1; i <= tot; i++) {
        L[i] = (i - 1) * sz + 1;
        R[i] = i * sz;
    }
    if (R[tot] < n) {
        ++tot;
        L[tot] = R[tot - 1] + 1;
        R[tot] = n;
    }
}
void Add(int v, ll& Ans) {
  ++cnt[v]; // ...
}
void Del(int v) { --cnt[v]; }

int main() {
    int l = 1, r = 0, last_block = 0, __l;
    ll Ans = 0, tmp;
    for (int i = 1; i <= q; i++) {
        if (pos[Q[i].l] == pos[Q[i].r]) {
            // 询问的左右端点同属于一个块则暴力扫描回答
            continue;
        }
        // 访问到了新的块则重新初始化莫队区间
        if (pos[Q[i].l] != last_block) {
            while (r > R[pos[Q[i].l]]) Del(x[r]), --r;
            while (l < R[pos[Q[i].l]] + 1) Del(x[l]), ++l;
            Ans = 0; last_block = pos[Q[i].l];
        }
        // 扩展右端点
        while (r < Q[i].r) ++r, Add(x[r], Ans);
        __l = l;
        tmp = Ans;
        // 扩展左端点
        while (__l > Q[i].l) --__l, Add(x[__l], tmp);
        ans[Q[i].id] = tmp;
        // 回滚
        while (__l < l) Del(x[__l]), ++__l;
    }
}
```
> Written with [StackEdit中文版](https://stackedit.cn/).
