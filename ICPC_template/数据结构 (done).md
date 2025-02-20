
# 数据结构

## 并查集

### 路径压缩

```cpp
const int N=1e5+10;
int n,fa[N];
void init() {for (int i=1;i<=n;i++) fa[i]=i;}
int find(int x) {return fa[x]==x?fa[x]:fa[x]=find(fa[x]);}
void merge(int x,int y) {
    int fx=find(x),fy=find(y);
    if (fx!=fy) fa[fy]=fx;
}
```

### 按秩合并

```cpp
const int N=1e5+10;
int n,fa[N],siz[N];
void init() {for (int i=1;i<=n;i++) fa[i]=i,siz[i]=1;}
int find(int x) {return fa[x]==x?fa[x]:find(fa[x]);}
void merge(int x,int y) {
    int fx=find(x),fy=find(y);
    if (fx!=fy) {
        if (siz[x]<siz[y]) swap(x,y);
        fa[y]=x; siz[x]+=siz[y];
    }
}
```


### 可撤销并查集

```cpp
const int N=1e5+10;
int n,fa[N],siz[N];
stack<pair<int *,int>> s;

void init() {for (int i=1;i<=n;i++) fa[i]=i,siz[i]=1;}
int find(int x) {return fa[x]==x?fa[x]:find(fa[x]);}
void merge(int x,int y) {
    int fx=find(x),fy=find(y);
    if (fx!=fy) {
        if (siz[x]<siz[y]) swap(x,y);
        s.push({fa+y,fa[y]}),fa[y]=x;
        s.push({siz+x,siz[x]}),siz[x]+=siz[y];
    }
}
unsigned tag() {return s.size();} // 获得此时的 tag
void undo(unsigned tag) { // 回溯到 tag 时刻
    while (s.size()>tag)
        *s.top().first=s.top().second,s.pop();
}
```

### 带权并查集

```cpp
void init() {for (int i=1;i<=n;i++) fa[i]=i,siz[i]=1;}
int find(int x){
    if (fa[x]==x) return x;
    int fx=find(fa[x]); dis[x]+=dis[fa[x]];
    return fa[x]=fa;
}
void merge(int x,int y,int c) {
    int fx=find(x),fy=find(y);
    if (fx==fy) return;
    f[fx]=fy,dis[fx]=c+dis[y]-dis[x];
}
```


## ST 表

```cpp
const int N=1e5+10; 
int a[N],f[N][21],lg[N],n,m;

void getlg(int n) {
    lg[1]=0,lg[2]=1;
    for(int i=3;i<=n;i++) lg[i]=lg[i/2]+1;
}
void init() {
    getlg(n);
    for (int i=1;i<=n;i++) f[i][0]=a[i];
    for (int j=1;j<=20;j++)
        for (int i=1;i+(1<<j)-1<=n;i++)
            f[i][j]=max(f[i][j-1],f[i+(1<<(j-1))][j-1]);
}
int query(int l,int r) {
    int log=lg[r-l+1];
    return max(f[l][log],f[r-(1<<log)+1][log]);
}
```


## 树状数组

1. 单点加区间查询

2. 区间加单点查询
    
    维护差分序列 $b_i=a_i-a_{i-1}$

    对于 $l\sim r$ 的区间加 $s$，$b_l\leftarrow b_l+s,b_{r+1}\leftarrow b_{r+1}-s$

    对于询问 $p$，$\sum_{i=1}^p b_i$

3. 区间加区间查询

    维护差分序列 $b_i=a_i-a_{i-1},c_i=i\cdot(a_i-a_{i-1})$

    对于 $l\sim r$ 的区间加 $s$，$b_l+s,b_{r+1}-s,c_l+l\cdot s, c_{r+1}-(r+1)\cdot s$

    对于查询 $l\sim r$，$((r+1)\sum_{i=1}^r b_i-\sum_{i=1}^rc_i)-(l\sum_{i=1}^{l-1}b_i-\sum_{i=1}^{l-1}c_i)$

```cpp
const int N=1e5+10;
int a[N],n;
int lowbit(int x) {return x&-x;}
void add(int pos,int sum) {
    while (pos<=n) {
        a[pos]+=sum;
        pos+=lowbit(pos);
    }
}
void query(int pos) {
    int ans=0;
    while (pos>0) {
        ans+=a[pos];
        pos-=lowbit(pos);
    }
    return ans;
}
```


## 线段树

```cpp
const int N=1e5+10;
int sum[N<<2],add[N<<2],a[N];

#define mid ((l+r)>>1)
#define ls (k<<1)
#define rs (k<<1|1)

void pushdown(int l,int r,int k) {
    add[ls]+=add[k],add[rs]+=add[k];
    sum[ls]+=add[k]*(mid-l+1),sum[rs]+=add[k]*(r-mid);
    add[k]=0;
} 
void pushup(int k) {
    sum[k]=sum[ls]+sum[rs];
}
void build(int l,int r,int k) {
    if (l==r) {sum[k]=a[l];return ;}
    build(l,mid,ls);
    build(mid+1,r,rs);
    pushup(k);
}
void update(int l,int r,int k,int L,int R,int ad) {
    if (r<L||R<l) return ;
    if (l>=L&&r<=R) {add[k]+=ad,sum[k]+=(r-l+1)*ad;return ;}
    pushdown(l,r,k);
    update(l,mid,ls,L,R,ad);
    update(mid+1,r,rs,L,R,ad);
    pushup(k);
}
int query(int l,int r,int k,int L,int R) {
    if (r<L||R<l) return 0;
    if (l>=L&&r<=R) return sum[k];
    pushdown(l,r,k);
    return query(l,mid,ls,L,R)+query(mid+1,r,rs,L,R);
}
```


### 区间最值修改

[HDU5306 Gorgeous Sequence] 维护一个序列 $a$

1. $l,r,t:\forall l\le i\le r,\ a_i=\min(a_i,t)$

2. $l,r$ 输出区间 $[l,r]$ 中的最大值
  
3. $l,r$ 输出区间 $[l,r]$ 的和

考虑每个结点维护最大值 $Mx$、严格次大值 $Se$、区间和 $Sum$、最大值的个数 $Cnt$ 

接下来我们考虑区间对 $t$ 取 $\min$ 的操作

- 如果 $Mx\le t$，直接返回

- 如果 $Se<t<Mx$ ，$Sum\leftarrow Sum+Cnt\cdot(t-Mx), Mx\leftarrow t$，并打一个标记

- 如果 $t\le Se$，暴力递归向下操作，然后上传信息

仅考虑第三种情况，对于每一个维护的区间，每一个 $\Theta(1)$ 的递归操作都会使该区间中 $Cnt$ 至少增加 $1$. 当 $Cnt$ 与区间长度相等时，$Se$ 恒为 $-\inf$，第三种情况将不会出现. 复杂度即为区间长度和. 

总复杂度：$\Theta((n+m)\log n)$

```cpp
const int N=1e6+10;
int sum[N<<2],mx[N<<2],se[N<<2],cnt[N<<2],tag[N<<2],a[N];

#define mid ((l+r)>>1)
#define ls (k<<1)
#define rs (k<<1|1)

int n,m;
void pushup(int k) {
    sum[k]=sum[ls]+sum[rs];
    mx[k]=max(mx[ls],mx[rs]);
    if (mx[ls]<mx[rs]) se[k]=max(mx[ls],se[rs]),cnt[k]=cnt[rs];
    else if (mx[ls]==mx[rs]) se[k]=max(se[rs],se[ls]),cnt[k]=cnt[ls]+cnt[rs];
    else se[k]=max(mx[rs],se[ls]),cnt[k]=cnt[ls];
}
void pushdown(int k) {
    if (tag[k]==-1) return ;
    if (mx[ls]>tag[k])
        sum[ls]-=(mx[ls]-tag[k])*cnt[ls],mx[ls]=tag[ls]=tag[k];
    if (mx[rs]>tag[k])
        sum[rs]-=(mx[rs]-tag[k])*cnt[rs],mx[rs]=tag[rs]=tag[k];
    tag[k]=-1;
}
void build(int l,int r,int k) {
    if (l==r) {
        mx[k]=sum[k]=a[l],cnt[k]=1,se[k]=-1,tag[k]=-1;
        return ;
    }
    build(l,mid,ls);build(mid+1,r,rs);
    pushup(k); tag[k]=-1;
}
void update(int l,int r,int k,int L,int R,int mn) {
    if (r<L||R<l) return ;
    if (mn>=mx[k]) return ;
    if (L<=l&&r<=R&&se[k]<mn) {
        sum[k]-=(mx[k]-mn)*cnt[k];
        mx[k]=tag[k]=mn; return ;
    }
    pushdown(k);
    update(l,mid,ls,L,R,mn); update(mid+1,r,rs,L,R,mn);
    pushup(k);
}
int querymax(int l,int r,int k,int L,int R) {
    if (r<L||R<l) return -1;
    if (L<=l&&r<=R) return mx[k];
    pushdown(k);
    return max(querymax(l,mid,ls,L,R),querymax(mid+1,r,rs,L,R));
}
int querysum(int l,int r,int k,int L,int R) {
    if (r<L||R<l) return 0;
    if (L<=l&&r<=R) return sum[k];
    pushdown(k);
    return querysum(l,mid,ls,L,R)+querysum(mid+1,r,rs,L,R);
}
```

### 区间历史版本和

设 $t$ 为当前时间，$h_i$ 为 $a_i$ 的历史版本和，维护一个值 $c_i=h_i-t\cdot a_i$。

一个把 $[l,r]$ 加上 $v$ 的区间加操作，发现对于 $i\in [l,r], c_i\leftarrow c_i-t\cdot v$（这里的 $t$ 是操作之前的时间）。于是 $c_i$ 也用区间加维护即可。

### 历史最大值

维护区间当前最大值 $mx$，区间历史最大值 $mx'$，还有这个节点当前加了多少值 $add$，以及这个节点在标记下放前加的值的历史最大值 $add'$。那么标记下传时，我们考虑更新儿子节点的值和标记：

$$mx_s'=\max(mx_s',mx_s+add_r')$$

$$add_s'=\max(add_s',add_s+add_r')$$


### 线段树分裂与合并

```cpp
const int N=1e5+10;
int cnt,tot,rt[N],ls[N<<5],rs[N<<5],val[N<<5],bac[N<<5];
// rt[i] 记录每颗线段树的根

int nw() {return (cnt?bac[cnt--]:++tot);}
void del(int p) {bac[++cnt]=p,ls[p]=rs[p]=val[p]=0;}
void update(int l,int r,int &k,int pos,int v) {
    if (pos<l||r<pos) return ;
    if (!k) {k=nw();}
    if (l==r) {val[k]+=v; return ;}
    val[k]+=v; int mid=(l+r)>>1;
    update(l,mid,ls[k],pos,v); update(mid+1,r,rs[k],pos,v);
}
int query(int l,int r,int k,int L,int R) {
    if (r<L||R<l||!k) return 0;
    if (L<=l&&r<=R) return val[k];
    int mid=(l+r)>>1;
    return query(l,mid,ls[k],L,R)+query(mid+1,r,rs[k],L,R);
}
int merge(int x,int y) {
    if (!x||!y) return x|y;
    val[x]+=val[y];
    ls[x]=merge(ls[x],ls[y]);
    rs[x]=merge(rs[x],rs[y]);
    del(y); return x;
}
// 分裂后，左侧线段树权值和等于 k
void split(int x,int &y,int k) { 
    if (!x) return ;
    y=nw(); int v=val[ls[x]];
    if (k>v) split(rs[x],rs[y],k-v);
    else swap(rs[x],rs[y]);
    if (k<v) split(ls[x],ls[y],k);
    val[y]=val[x]-k; val[x]=k;
}
```


### 李超线段树

设当前区间的中点为 $m$，我们拿新线段 $f$ 在中点处的值与原最优线段 $g$ 在中点处的值作比较。

如果新线段 $f$ 更优，则将 $f$ 和 $g$ 交换。那么现在考虑在中点处 $f$ 不如 $g$ 优的情况：

- 若在左端点处 $f$ 更优，那么 $f$ 和 $g$ 必然在左半区间中产生了交点，$f$ 只有在左区间才可能优于 $g$，递归到左儿子中进行下传
- 若在右端点处 $f$ 更优，那么 $f$ 和 $g$ 必然在右半区间中产生了交点，$f$ 只有在右区间才可能优于 $g$，递归到右儿子中进行下传
- 若在左右端点处 $g$ 都更优，那么 $f$ 不可能成为答案，不需要继续下传

在范围 $[L,R]$ 插入直线 $y=kx+b$

```cpp
#define fi first
#define se second
#define mk make_pair
const double eps=1e-8;
const int N=1e5+10;
int rt,n,tot,cnt,id[N],ls[N],rs[N];
pair<double,double> s[N];

bool cmp(int i,int j,int x){
    double ans1=x*s[i].fi+s[i].se;
    double ans2=x*s[j].fi+s[j].se;
    if(abs(ans1-ans2)<eps) return i>j;
    return ans1<ans2;
}
void insert(int l,int r,int &k,int p){
    if (!k) k=++cnt;
    if (cmp(id[k],p,l)&&cmp(id[k],p,r)) {id[k]=p; return;}
    if (cmp(p,id[k],l)&&cmp(p,id[k],r)) return;
    if (l==r) return ;
    int mid=(l+r)>>1;
    if (cmp(id[k],p,mid)) swap(id[k],p);
    if (cmp(id[k],p,l)) insert(l,mid,ls[k],p);
    else insert(mid+1,r,rs[k],p);
}
int find(int l,int r,int k,int x){
    if (l==r) return id[k];
    if (!k) return 0;
    int mid=(l+r)>>1,ans;
    if (x<=mid) ans=find(l,mid,ls[k],x);
    else ans=find(mid+1,r,rs[k],x);
    if (cmp(ans,id[k],x)) ans=id[k];
    return ans;
}
void solve() {
    s[0]=mk(0,-1e9);
    const int L=1,R=50000;
    // 添加
    s[++tot]=mk(k,b);
    insert(L,R,rt,tot);
    // 查询
    int id=find(L,R,rt,x);
    cout<<s[id].fi*x+s[id].se;
}
```

插入线段 $[l_i,r_i]$，查询相同。

```cpp
void insert(int l,int r,int &k,int L,int R,int p){
    if (!k) k=++cnt;
    if (r<L||R<l) return ;
    if (L<=l&&r<=R) {
        if (cmp(id[k],p,l)&&cmp(id[k],p,r)) {id[k]=p; return;}
        if (cmp(p,id[k],l)&&cmp(p,id[k],r)) return;
        if (l==r) return ;
        int mid=(l+r)>>1;
        if (cmp(id[k],p,mid)) swap(id[k],p);
        if (cmp(id[k],p,l)) insert(l,mid,ls[k],L,R,p);
        else insert(mid+1,r,rs[k],L,R,p);
        return ;
    }
    int mid=(l+r)>>1;
    insert(l,mid,ls[k],L,R,p);
    insert(mid+1,r,rs[k],L,R,p);
}
void add(int x1,int y1,int x2,int y2) {
    int L=1,R=50000;
    if (x1>x2) swap(x1,x2),swap(y1,y2);
    if (x1==x2) s[++tot]=mk(0,max(y1,y2));
    else {
        double k=(y2-y1)*1.0/(x2-x1);
        s[++tot]=mk(k,y1-x1*k);
    }
    insert(L,R,rt,x1,x2,tot);
}
```


## 平衡树

### FHQ treap

```cpp
const int N=1e6+1e5+10;
int val[N],key[N],ls[N],rs[N],siz[N],fa[N];
int n,cnt,rt;

void create(int &id,int v) {id=++cnt;val[id]=v,key[id]=rand(),siz[id]=1;}
void update(int x) {siz[x]=siz[ls[x]]+siz[rs[x]]+1;fa[ls[x]]=fa[rs[x]]=x;}
void pushdown(int x) {}
// val split 的平衡树；split 后，左侧 treap 的 val 均小于等于 v
void split(int id,int v,int &x,int &y) {
    if (!id) {x=y=0;return;} pushdown(id);
    if (val[id]<=v) {x=id;split(rs[id],v,rs[id],y);}
    else {y=id;split(ls[id],v,x,ls[id]);}
    update(id);
}
// siz split 的平衡树；split 后，左侧 treap 的大小为 v
void split(int id,int v,int &x,int &y) {
    if (!id) {x=y=0;return ;} pushdown(id);
    if (v>siz[ls[id]]) x=id,split(rs[id],v-siz[ls[id]]-1,rs[id],y);
    else y=id,split(ls[id],v,x,ls[id]);
    update(id);
}
// 不论 val 还是 siz split 的平衡树，merge 均一致
int merge(int x,int y) {
    if (!x||!y) return x|y;
    if (key[x]<key[y]) {pushdown(x);rs[x]=merge(rs[x],y);update(x);return x;}
    else {pushdown(y);ls[y]=merge(x,ls[y]);update(y);return y;}
}

void push(int v) {int x,y,z;split(rt,v,x,z);create(y,v);rt=merge(merge(x,y),z);}
void pop(int v) {int x,y,z;split(rt,v,x,z);split(x,v-1,x,y);y=merge(ls[y],rs[y]);rt=merge(merge(x,y),z);}
void pop2(int v) {int x,y,z;split(rt,v,x,z);split(x,v-1,x,y);rt=merge(x,z);}
int find(int v) {int x,y;split(rt,v-1,x,y);int tmp=siz[x]+1;rt=merge(x,y);return tmp;}
// 按siz合并，查找位置的函数。x 为第 x 个插入的数
int sizfind(int x) {
    int ans=siz[ls[x]]+1;
    while (x!=rt) {
        if (rs[fa[x]]==x) ans+=siz[ls[fa[x]]]+1;
        x=fa[x];
    }
    return ans;
}
int _findth(int r,int k) {
    while(r) {
        if (siz[ls[r]]+1==k) return r;
        else if (k<=siz[ls[r]]) r=ls[r];
        else k=k-siz[ls[r]]-1,r=rs[r];
    }
    return 0;
}
int findth(int k) {return val[_findth(rt,k)];}
int pre(int v) {int x,y;split(rt,v-1,x,y);int tmp=_findth(x,siz[x]);rt=merge(x,y);return val[tmp];}
int sub(int v) {int x,y;split(rt,v,x,y);int tmp=_findth(y,1);rt=merge(x,y);return val[tmp];}
int pre2(int v) {int x,y;split(rt,v,x,y);int tmp=_findth(x,siz[x]-1);rt=merge(x,y);return tmp;}
int sub2(int v) {int x,y;split(rt,v-1,x,y);int tmp=_findth(y,2);rt=merge(x,y);return tmp;}

// 可持久化
int clone(int rt) {
    val[++cnt]=val[rt],ls[cnt]=ls[rt],rs[cnt]=rs[rt],siz[cnt]=siz[rt],key[cnt]=key[rt],fa[cnt]=fa[rt];
    return cnt;
}
void pesplit(int id,int v,int &x,int &y) {
    if (!id) {x=y=0;return;} pushdown(id);
    if (val[id]<=v) {x=clone(id);pesplit(rs[x],v,rs[x],y);update(x);}
    else {y=clone(id);pesplit(ls[y],v,x,ls[y]);update(y);}
}
int pemerge(int x,int y) {
    if (!x||!y) return x|y;
    if (key[x]<key[y]) {pushdown(x);x=clone(x);rs[x]=pemerge(rs[x],y);update(x);return x;}
    else {pushdown(y);y=clone(y);ls[y]=pemerge(x,ls[y]);update(y);return y;}
}
```


## 笛卡尔树

一颗 treap，中序遍历节点编号为 $1,2,\cdots,n$，且是按照 key 的小根堆

```cpp
const int N=1e7+10;
int n,key[N],s[N],ls[N],rs[N];
void build() {
    int top=0,x,last;
    for (int i=1;i<=n;i++) {
        x=i,last=0;
        while (top&&key[s[top]]>key[x]) last=s[top],s[top--]=0;
        if (top) rs[s[top]]=x;
        ls[x]=last,s[++top]=x;
    }
}
void solve() {
    for (int i=1;i<=n;i++) key[i]=read();
    build();
}
```


## 主席树

关于区间加、区间求和，需要维护一个永久懒标记 $add$

对于询问，我们再开一个懒标记 $al$，根节点初值赋为 ``al[rt]=add[rt]``，并在询问过程中不断将 $add_i$ 相加、下传，即 ``al[ls[i]]=add[ls[i]]+al[i]``

下面代码为静态区间第 $k$ 大。

```cpp
const int N=1e5+10;
int cnt,sum[N],ls[N],rs[N];

void clone(int x,int y) {sum[x]=sum[y],ls[x]=ls[y],rs[x]=rs[y];}
int build(int l,int r) {
    int dir=++cnt;
    if (l==r) return dir;
    int mid=(l+r)>>1;
    ls[dir]=build(l,mid); rs[dir]=build(mid+1,r);
    return dir;
}
int update(int l,int r,int k,int pos,int v) {
    int dir=++cnt; clone(cnt,k); sum[cnt]+=v;
    if (l==r) return dir;
    int mid=(l+r)>>1;
    if (pos<=mid) ls[dir]=update(l,mid,ls[k],pos,v);
    else rs[dir]=update(mid+1,r,rs[k],pos,v);
    return dir;
}
int query(int l,int r,int L,int R,int k) {
    if (l==r) return l;
    int mid=(l+r)>>1,x=sum[ls[R]]-sum[ls[L-1]];
    if (k<=x) return query(l,mid,ls[L],ls[R],k);
    else return query(mid+1,r,rs[L],rs[R],k);
}
```


## Trie 与可持久化 Trie

```cpp
const int N=1e5+10;
int cnt,vis[N][2],val[N];

int insert(int num,int lst) {
    int nw=++cnt;
    while (i=30;i>=0;i--) {
        val[nw]=val[lst]+1;
        int x=(num>>i)&1;
        vis[nw][x^1]=vis[lst][x^1];
        vis[nw][x]=++cnt;
        lst=vis[lst][x]; nw=vis[nw][x];
    }
    val[nw]=val[lst]+1;
}
int insert(int num) {
    int rt=1;
    for (int i=30;i>=0;i--) {
        ++val[rt];
        int x=(num>>i)&1;
        if (!vis[rt][x]) vis[rt][x]=++cnt;
        rt=vis[rt][x];
    } ++val[rt];
}
```


## 可并堆

```cpp
struct heap{ // 小根堆
    int rt[N],v[N],ls[N],rs[N],dis[N],del[N];
    void init() {
        dis[0]=-1;
        for (int i=1;i<=n;i++) rt[i]=i;
        for (int i=1;i<=n;i++) cin>>v[i];
    }
    int find(int x) {
        return x==rt[x]?x:rt[x]=find(rt[x]);
    }
    int Merge(int x,int y) {
        if (!x||!y) return x|y;
        if (v[x]>v[y]) swap(x,y);
        rs[x]=Merge(rs[x],y);
        if (dis[rs[x]]>dis[ls[x]]) swap(rs[x],ls[x]);
        dis[x]=dis[rs[x]]+1;
        return x;
    }
    void merge(int x,int y) {
        if (del[x]||del[y]) return ;
        x=find(x),y=find(y);
        if (x!=y) rt[x]=rt[y]=Merge(x,y);
    }
    int top(int x) {
        if (del[x]) return -1;
        x=find(x); return v[x];
    }
    int pop(int x) {
        if (del[x]) return -1;
        x=find(x); del[x]=1;
        int ans=v[x];
        rt[ls[x]]=rt[rs[x]]=rt[x]=Merge(ls[x],rs[x]);
        rs[x]=ls[x]=dis[x]=0;
        return ans;
    }
};
```


## LCT

Splay 函数：

- ``Splay(x)`` 通过和 Rotate 操作联动实现把 x 旋转到 当前 Splay 的根。

- ``Rotate(x)`` 将 x 向上旋转一层的操作。

LCT 函数：

- ``Access(x)`` 把从根到 x 的所有点放在一条实链里，使根到 x 成为一条实路径，并且在同一棵 Splay 里。

- ``IsRoot(x)`` 判断 x 是否是所在树的根。

- ``MakeRoot(x)`` 使 x 点成为其所在树的根。

- ``Link(x, y)`` 在 x, y 两点间连一条边。

- ``Cut(x, y)`` 把 x, y 两点间边删掉。

- ``Find(x)`` 找到 x 所在树的根节点编号。

- ``Split(x, y)`` 提取出 x, y 间的路径

```cpp
int val[N];
struct Link_Cut_Tree{
    int top,c[N][2],fa[N],q[N],rev[N];
    int xr[N]; // 维护的值
    inline void pushup(int x){xr[x]=xr[c[x][0]]^xr[c[x][1]]^val[x];}
    inline void pushdown(int x){
        int l=c[x][0],r=c[x][1];
        if(rev[x]){
            rev[l]^=1;rev[r]^=1;rev[x]^=1;
            swap(c[x][0],c[x][1]);
        }
    }
    inline bool isroot(int x){return c[fa[x]][0]!=x&&c[fa[x]][1]!=x;}
    void rotate(int x){
        int y=fa[x],z=fa[y],l,r;
        if(c[y][0]==x)l=0;else l=1;r=l^1;
        if(!isroot(y)){if(c[z][0]==y)c[z][0]=x;else c[z][1]=x;}
        fa[x]=z;fa[y]=x;fa[c[x][r]]=y;
        c[y][l]=c[x][r];c[x][r]=y;
        pushup(y);pushup(x);
    }
    void splay(int x){
        top=1;q[top]=x;
        for(int i=x;!isroot(i);i=fa[i])q[++top]=fa[i];
        for(int i=top;i;i--)pushdown(q[i]);
        while(!isroot(x)){
            int y=fa[x],z=fa[y];
            if(!isroot(y)){
                if((c[y][0]==x)^(c[z][0]==y))rotate(x);
                else rotate(y);
            }rotate(x);
        }
    }
    void access(int x){for(int t=0;x;t=x,x=fa[x])splay(x),c[x][1]=t,pushup(x);}
    void makeroot(int x){access(x);splay(x);rev[x]^=1;}
    int find(int x){access(x);splay(x);while(c[x][0])x=c[x][0];return x;}
    void split(int x,int y){makeroot(x);access(y);splay(y);}
    void cut(int x,int y){split(x,y);if(c[y][0]==x&&c[x][1]==0)c[y][0]=0,fa[x]=0;}
    void link(int x,int y){makeroot(x);fa[x]=y;}
}T;
```


> Written with [StackEdit中文版](https://stackedit.cn/).
