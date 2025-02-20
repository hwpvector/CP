

# 图论

## 链式前向星

```cpp
const int N=1e5+10,M=2e5+10;
int head[N],nxt[M],to[M],val[N];
void add(int u,int v,int w) {
    nxt[++cnt]=head[u],head[u]=cnt,to[cnt]=v,val[cnt]=w;
}
void get(int u) {
    for (int i=head[u];~i;i=nxt[i]) {
        int v=to[i],w=val[i];
    }
}
```



## 拓扑排序

若存在至少一个环，则返回 ``Flase``。

```cpp
#define pb push_back
const int N=1e5+10,M=2e5+10;
vector<int> e[N],t;
int n,m,in[N];

void init() {
    for (int i=1;i<=m;i++) {
        int u,v; cin>>u>>v;
        e[u].pb(v); ++in[v];
    }
}
bool topo() {
    queue<int> q;
    for (int i=1;i<=n;i++)
        if (!in[i]) q.push(i);
    while (!q.empty()) {
        int u=q.front(); q.pop();
        t.pb(u);
        for (auto v:e[u]) {
            --in[v];
            if (!in[v]) q.push(v);
        }
    }
    return t.size()==n;
}
```



## 单源最短路

### Dijkstra

要求：非负权图

```cpp
#define pii pair<int,int>
#define pb push_back
const int N=1e5+10,M=2e5+10;
int dis[N],vis[N];
vector<int> e[N];

void dij(int s) {
    memset(dis,0x3f3f3f3f,sizeof(dis));
    priority_queue<pii,vector<pii>,greater<pii>> q;
    dis[s]=0; q.push(mk(0,s));
    while (!q.empty()) {
        int u=q.top().u; q.pop();
        if (vis[u]) continue;
        vis[u]=1;
        for (auto [v,w]:e[u])
            if (dis[v]>dis[u]+w) {
                dis[v]=dis[u]+w;
                q.push(mk(dis[v],v));
            }
    }
}
```



### SPFA

可以跑负权图。若存在负环，则返回 ``Flase``。~~SPFA已死~~

```cpp
#define pii pair<int,int>
#define pb push_back
const int N=1e5+10,M=2e5+10;
int dis[N],vis[N],cnt[N];
vector<int> e[N];

bool spfa(int s) {
    memset(dis,0x3f3f3f3f,sizeof(dis));
    queue<int> q;
    dis[s]=0,vis[s]=1; q.push(s);
    while (!q.empty()) {
        int u=q.front(); q.pop();
        vis[u]=0;
        for (auto [v,w]:e[u]) 
            if (dis[v]>dis[u]+w) {
                dis[v]=dis[u]+w;
                cnt[v]=cnt[u]+1;
                if (cnt[v]>=n) return 0;
                if (!vis[v]) q.push(v), vis[v]=1;
            }
    }
    return 1;
}
```



## 全源最短路

### Floyd

可以跑负权图。

$f_{k,x,y}$ 表示只允许经过结点 $1\sim k$，结点 $x$ 到结点 $y$ 的最短路长度。注意到第一维状态可以省略。


```cpp
int N=200;
int n,m,f[N][N];

void floyd() {
    memset(f,0x3f3f3f3f,sizeof(f));
    for (int i=1;i<=n;i++) f[i][i]=0;
    for (int i=1;i<=m;i++) {
        int u,v,w; cin>>u>>v>>w;
        f[u][v]=min(f[u][v],w);
    }
    for (int k=1;k<=n;k++) 
        for (int x=1;x<=n;x++)
            for (int y=1;y<=n;y++)
                f[x][y]=min(f[x][y],f[x][k]+f[k][y]);
}
```

### Johnson

可以跑负权图。

我们新建一个虚拟节点（在这里我们就设它的编号为 $0$ ）。从这个点向其他所有点连一条边权为 
$0$ 的边。接下来用 SPFA 算法求出从 $0$ 号点到其他所有点的最短路，记为 $h_i$

假如存在一条从 $u$ 点到 $v$ 点，边权为 $w$ 的有向边，则我们将该边的边权重新设置为 $w+h_u-h_v$。以每个点为起点，跑 $n$ 轮 Dijkstra 即可求出任意两点间的最短路。

对于 $u$ 到 $v$ 的最短路，实际上为 $dis_{u,v}-h_u+h_v$。



## 最小生成树

### Kruskal

```cpp
#define pii pair<int,int>
#define pb push_back
const int N=1e5+10,M=2e5+10;
pair<int,pii> e[M];
vector<pii> t[N];
int n,m,fa[N];

int find(int u) {
    return u==fa[u]?u:fa[u]=find(fa[u]);
}
void kruskal() {
    for (int i=1;i<=m;i++) 
        cin>>e[i].se.fi>>e[i].se.se>>e[i].fi;
    for (int i=1;i<=n;i++) fa[i]=i;
    sort(e+1,e+m+1);
    for (int i=1;i<=m;i++) {
        int u=e[i].se.fi,v=e[i].se.se,w=e[i].fi;
        int fu=find(u),fv=find(v);
        if (fu==fv) continue;
        fa[fu]=fv;
        t[u].pb(mk(v,w)),t[v].pb(mk(u,w));
    }
}
```

### Boruvka

通常题面为：给出一种计算两点 $u,v$ 之间边权的方法，求一个完全图的生成树。通常可以快速计算每一个点连向其他点的最短距离的计算方法（要求复杂度为 $O(n)\sim O(n\log n)$ 数量级）

- 维护当前所有点形成的连通块，初始时没有选择任何边，也就是每个点单独形成一个连通块

- 进行若干轮连边，在每一轮中，为每个连通块找到一条权值最小的连向该连通块外部的边，并将这条边连接起来

- 直到某一轮发现所有点已经在一个连通块内时



### 严格次小生成树

- 求出无向图的最小生成树 $T$，设其权值和为 $M$

- 倍增维护到 $2^i$ 级祖先路径上的最大边权，同时维护严格次大边权

- 遍历每条未被选中的边 $e = (u,v,w)$，找到 $T$ 中 $u$ 到 $v$ 路径上边权最大的一条边 $e' = (s,t,w')$。当 $w=w'$ 时，用 $u$ 到 $v$ 路径上的严格次大边权代替。
  
- 在 $T$ 中以 $e$ 替换 $e'$，可得一棵权值和为 $M' = M + w - w'$ 的生成树 $T'$

- 对所有替换得到的答案 $M'$ 取最小值即可

## Kruskal 重构树

```cpp
#define pii pair<int,int>
#define pb push_back
const int N=1e5+10,M=2e5+10;
pair<int,pii> e[M];
vector<pii> t[N];
int cnt,n,m,fa[N];

int find(int u) {
    return u==fa[u]?u:fa[u]=find(fa[u]);
}
void kruskal() {
    for (int i=1;i<=m;i++) 
        cin>>e[i].se.fi>>e[i].se.se>>e[i].fi;
    for (int i=1;i<=n;i++) fa[i]=i;
    sort(e+1,e+m+1);
    cnt=n;
    for (int i=1;i<=m;i++) {
        int u=e[i].u,v=e[i].v,w=e[i].w;
        int fu=find(u),fv=find(v);
        if (fu!=fv) {
            val[++cnt]=w; fa[fu]=fa[fv]=cnt; fa[cnt]=cnt;
            t[fu].pb(cnt),t[fv].pb(cnt),t[cnt].pb(fu),t[cnt].pb(fv);
        }
    }
}
```

## LCA

### 倍增 LCA

```cpp
const int N=1e5+10;
int n,m,f[N][20],dep[N];
vector<int> e[N];

void dfs(int u,int fa) {
    f[u][0]=fa; dep[u]=dep[fa]+1;
    for (int i=1;i<20;i++) {
        if (dep[u]<(1<<i)) break;
        f[u][i]=f[f[u][i-1]][i-1];
    }
    for (auto v:e[u]) if (v!=fa) dfs(v,u);
}
int lca(int x,int y) {
    if (dep[x]<dep[y]) swap(x,y);
    for (int i=19;i>=0;i--)
        if (dep[x]-(1<<i)>=dep[y]) x=f[x][i];
    if (x==y) return x;
    for (int i=19;i>=0;i--) {
        if ((1<<i)>dep[x]) continue;
        if (f[x][i]!=f[y][i]) x=f[x][i],y=f[y][i];
    }
    return f[x][0];
}
```



### Tarjan LCA

```cpp
#define pb push_back
#define pii pair<int,int>
const int N=1e5+10,Q=1e5+10;
int n,q,ans[Q],vis[N],fa[N];
vector<int> e[N];
vector<pii> t[N];

int find(int u) {
    return u==fa[u]?u:fa[u]=find(fa[u]);
}
void dfs(int u,int f) {
    vis[u]=1; fa[u]=u;
    for (auto v:e[u]) if (v!=f) {
        dfs(v,u); fa[v]=u;
    }
    for (auto [v,id]:t[u]) {
        if (vis[v]) ans[id]=find(v);
    }
}
void tarjan() {
    for (int i=1;i<=q;i++) {
        int u,v; cin>>u>>v;
        t[u].pb(mk(v,i)),t[v].pb(mk(u,i));
    }
    dfs(1,0);
    for (int i=1;i<=q;i++) cout<<ans[i]<<'\n';
}
```



## 树链剖分

### 重链剖分

```cpp
const int N=1e5+10;
vector<int> e[N];
int n,A[N],a[N],dep[N],fa[N],son[N],siz[N],top[N],id[N],tot;

void dfs1(int u,int f,int d) {
    dep[u]=d,fa[u]=f,siz[u]=1;
    for (int v:e[u]) if (v!=f) {
        dfs1(v,u,d+1);
        siz[u]+=siz[v];
        if (siz[v]>siz[son[u]]) son[u]=v;
    }
}
void dfs2(int u,int tp) {
    id[u]=++tot,a[tot]=A[u],top[u]=tp;
    if (son[u]==0) return ;
    dfs2(son[u],tp);
    for (int v:e[u]) if (v!=fa[u]&&v!=son[u]) dfs2(v,v);
}
inline int queryuv(int u,int v){
    int ans=0;
    while (top[u]!=top[v]){
        if (dep[top[u]]<dep[top[v]]) swap(u,v);
        ans+=query(1,n,1,id[top[u]],id[u]);
        u=fa[top[u]];
    }
    if (dep[u]>dep[v]) swap(u,v);
    return ans+query(1,n,1,id[u],id[v]);
}
inline void updateuv(int u,int v,int num) {
    while (top[u]!=top[v]){
        if (dep[top[u]]<dep[top[v]]) swap(u,v);
        update(1,n,1,id[top[u]],id[u],num);
        u=fa[top[u]];
    }
    if (dep[u]>dep[v]) swap(u,v);
    update(1,n,1,id[u],id[v],num);
}
inline int querytr(int u) {return query(1,n,1,id[u],id[u]+siz[u]-1);}
inline void updatetr(int u,int num) {update(1,n,1,id[u],id[u]+siz[u]-1,num);}
void solve() {
    for (int i=1;i<=n;i++) cin>>A[i]; // 权值保存到 A[i]
    dfs1(1,0,1); dfs2(1,1);
    build(1,n,1); // 线段树 build 时，权值需使用 a[i]
}
```



### dsu on tree

```cpp
const int N=1e5+10;
int siz[N],son[N],;
int col[N],cnt[N],ans[N],Mx,sum;

void dfs(int u,int fa) {
    siz[u]=1;
    for (int v:e[u]) if (v!=fa) {
        dfs(v,u); siz[u]+=siz[v];
        if (siz[v]>siz[son[u]]) son[u]=v;
    }
}
void add(int u,int fa,int val) {
    cnt[col[u]]+=val;
    if (cnt[col[u]]>Mx) Mx=cnt[col[u]],sum=col[u];
    else if (cnt[col[u]]==Mx) sum+=col[u];
    for (int v:e[u]) if (v!=fa&&v!=s) add(v,u,val);
}
void dfs2(int u,int fa,int op) {
    for (int v:e[u]) if (v!=fa&&v!=son[u]) dfs2(to,u,0);
    if (son[u]) dfs2(son[u],u,1);
    s=son[u]; add(u,fa,1); ans[u]=sum;
    s=0; if(!op) add(x,fa,-1),sum=0,Mx=0;
}
```



### 长链剖分

```cpp
const int N=1e5+10;
vector<int> e[N];
int h[N],*f[N],son[N],g[N],siz;

void dfs1(int u,int fa) {
    for (auto v:e[u]) if (v!=fa) {
        dfs1(v,u);
        if (h[v]>h[u]) h[u]=h[v],son[u]=v; 
    }
    h[u]++;
}
void dfs2(int u,int fa) {
    // 对 f[u][0] 赋初始值
    if (!son[u]) return ;
    f[son[u]]=f[u]+1; dfs2(son[u],u);
    for (auto v:e[u]) if (v!=fa&&v!=son[u]) {
        f[v]=g+siz; siz+=h[v]; dfs2(v,u);
        for(int j=1;j<=h[v];j++) {
            // 对 dp 数组 f 进行维护
        }
    }
}
void solve() {
    dfs1(1,0);
    f[1]=g,siz=h[1]; dfs2(1,0);
}
```



## 虚树

虚树：查询点 + 查询点的两两 LCA，所组成的树

```cpp
bool valid[N];
int rt,cnt,dfn[N],h[N],a[N],len;
vector<int> e[N],t[N],cl;

void dfs(int u,int fa) {
    dfn[u]=++cnt;
    for (auto v:e[u]) if (v!=fa) dfs(v,u);
}
void clear() {
    len=0; dfn[0]=0x3f3f3f3f;
    for (auto i:cl) t[i].clear();
    cl.clear();
}
bool cmp(int x,int y) {return dfn[x]<dfn[y];}
void build() {
    sort(h+1,h+m+1,cmp);
    for (int i=1;i<m;++i)
        a[++len]=h[i],a[++len]=lca(h[i],h[i+1]);
    a[++len]=h[m];
    sort(a+1,a+len+1,cmp);
    len=unique(a+1,a+len+1)-a-1;
    for (int i=1;i<len;++i) {
        int lc=lca(a[i],a[i+1]);
        t[lc].pb(a[i+1]); cl.pb(lc);
        if (dfn[rt]>dfn[lc]) rt=lc;
    }
}
```



## 树分治

### 点分治

点分治适合处理大规模的树上路径信息问题。

计算树上两点距离为 $k$ 的个数，$q$ 个询问：

- 选取树的重心，作为根节点
- 计算节点到根的距离 $w_i$（$u$ 和 $v$ 的距离为 $w(u)+w(v)$），计算以根节点 LCA 时的答案（复杂度 $O(n)$）
- 递归子树，以子树中的树的重心作为根，计算答案

### 点分树

按照点分树的形式，调整树的形态，每个节点连向其各个子树的重心

对于任意两点 $u,v$，唯一可以确定是 $u,v$ 在点分树上的 LCA 一定在 $u\rightarrow v$ 的路径上。或者，$dis(u,v)=dis(u,lca)+dis(lca,v)$

```cpp
const int N=1e5+10;
vector<int> e[N],t[N];
int n,siz[N],mxsiz[N],del[N],rt;

void getsiz(int u,int fa) {
    siz[u]=1;
    for (auto v:e[u]) if (v!=fa&&!del[v]) {
        getsiz(v,u); siz[u]+=siz[v];
    }
}
void getmxsiz(int u,int fa,int sz) {
    mxsiz[u]=sz-siz[u];
    for (auto v:e[u]) if (v!=fa&&!del[v]) mxsiz[u]=max(mxsiz[u],siz[v]);
    for (auto v:e[u]) if (v!=fa&&!del[v]) getmxsiz(v,u,sz);
    if (!rt||mxsiz[rt]>mxsiz[u]) rt=u;
}
void findrt(int u) {
    getsiz(u,0);
    rt=0; getmxsiz(u,0,siz[u]);
}
int build(int u) {
    findrt(u); u=rt; del[u]=1;
    for (auto v:e[u]) if (!del[v]) {
        int s=build(v);
        t[u].pb(s);
    }
    return u;
}
```



## 树哈希

对于无根树 $T_1(V_1, E_1)$ 和 $T_2(V_2,E_2)$，先分别找出它们的所有重心。

- 如果这两棵无根树重心数量不同，那么这两棵树不同构
- 如果这两颗无根树重心数量都为 $1$，分别记为 $c_1$ 和 $c_2$，那么如果有根树 $T_1(V_1,E_1,c_1)$ 和有根树 $T_2(V_2,E_2,c_2)$ 同构，那么无根树 $T_1(V_1, E_1)$ 和 $T_2(V_2,E_2)$ 同构，反之则不同构
- 如果这两颗无根树重心数量都为 $2$，分别记为 $c_1,c'_1$ 和 $c_2,c'_2$，那么如果有根树 $T_1(V_1,E_1,c_1)$ 和有根树 $T_2(V_2,E_2,c_2)$ 同构或者有根树 $T_1(V_1,E_1,c'_1)$ 和 $T_2(V_2,E_2,c_2)$ 同构，那么无根树 $T_1(V_1, E_1)$ 和 $T_2(V_2,E_2)$ 同构，反之则不同构

```cpp
typedef unsigned long long ull 
const ull mask = std::chrono::steady_clock::now().time_since_epoch().count();
const int N=1e5+10;
int n,sub[N],root[N];
vector<int> e[N];

ull shift(ull x) {x^=mask;x^=x<<13;x^=x>>7;x^=x<<17;x^=mask;return x;}
void getsub(int u) {
    sub[u]=1;
    for (int v:e[u]) {
        getsub(v); sub[u]+=shift(sub[v]);
    }
} 
void getroot(int x) {
    for (int v:e[u]) {
        root[v]=sub[v]+shift(root[u]-shift(sub[v]));
        getroot(v);
    }
}
ull hash1(int rt) { // 有根树的 Hash
    return getsub(rt);
}
ull hash2() { // 无根树的 Hash
    getsub(1);
    root[1]=sub[1]; getroot(1);
    ull h=1;
    for (int i=1;i<=n;i++) h+=shift(rt[i]);
    return h;
}
```



## 连通性相关

### 缩点

```cpp
const int N=1e5+10;
int dfn[N],low[N],dfncnt,s[N],in_stack[N],tp;
int scc[N],sz[N],sc;
// 结点 i 所在 SCC 的编号；强连通 i 的大小
vector<int> e[N];

void tarjan(int u) {
    low[u]=dfn[u]=++dfncnt,s[++tp]=u,in_stack[u]=1;
    for (auto v:e[u]) {
        if (!dfn[v])
            tarjan(v),low[u]=min(low[u],low[v]);
        else if (in_stack[v])
            low[u]=min(low[u], dfn[v]);
    }
    if (dfn[u]==low[u]) {
        ++sc;
        while (s[tp]!=u) {
            scc[s[tp]]=sc; sz[sc]++;
            in_stack[s[tp]]=0; --tp;
        }
        scc[s[tp]]=sc; sz[sc]++;
        in_stack[s[tp]]=0; --tp;
    }
}
```



### 割点

```cpp
const int N=1e5+10;
int n,m,dfn[N],low[N],flag[N],dfncnt,res;
vector<int> e[N];

void Tarjan(int u,int fa) {
    low[u]=dfn[u]=++dfncnt;
    int child=0;
    for (auto v:e[u]) {
        if (!dfn[v]) {
            child++; Tarjan(v,u);
            low[u]=min(low[u],low[v]);
            if (fa!=u&&low[v]>=dfn[u]&&!flag[u]) 
                flag[u]=1,res++;
        } 
        else if (v!=fa) low[u]=min(low[u],dfn[v]);
    }
    if (fa==u&&child>=2&&!flag[u])
        flag[u]=1,res++;
}
int main() {
    for (int i=1;i<=n;i++)
        if (!dfn[i]) dfncnt=0,Tarjan(i,i);
    cout<<res<<'\n';
    for (int i=1;i<=n;i++)
        if (flag[i]) cout<<i<< " ";
}
```



### 桥

当 ``isbridge[x]`` 为 ``1`` 时，``(fa[x],x)`` 为一条割边

```cpp
const int N=1e5+10;
int low[N],dfn[N],dfncnt;
int res,isbridge[N],fa[N];
vector<int> e[N];

void tarjan(int u,int f) {
    fa[u]=f; low[u]=dfn[u]=++dfncnt;
    for (auto v:e[u]) {
        if (!dfn[v]) {
            tarjan(v,u); low[u]=min(low[u],low[v]);
            if (low[v]>dfn[u]) {
                isbridge[v]=1; ++res;
            }
        }
        else if (v!=f) low[u]=min(low[u],dfn[v]);
    }
}
```



### 边双连通分量

在一张连通的无向图中，对于两个点 $u$ 和 $v$，如果无论删去哪条边（只能删去一条）都不能使它们不连通，我们就说 $u$ 和 $v$ 边双连通。

边双连通具有传递性，即，若 $x$,$y$ 边双连通，$y$,$z$ 边双连通，则 $x$,$z$ 边双连通。

我们考虑先求出所有的桥，将桥删除，再 DFS 求出边双连通分量。

### 点双连通分量

在一张连通的无向图中，对于两个点 $u$ 和 $v$，如果无论删去哪个点（只能删去一个，且不能删 $u$ 和 $v$ 自己）都不能使它们不连通，我们就说 $u$ 和 $v$ 点双连通。

点双连通不具有传递性。

```cpp
const int N=1e5+10,;
int low[N],dfn[N],s[N],top,bcc,dfncnt,n,m;
vector<int> e[N],ans[N];

inline void tarjan(int u,int fa) {
    int son=0; low[u]=dfn[u]=++dfncnt; s[++top]=u;
    for (auto v:e[u]) {
        if (!dfn[v]) {
            son++; tarjan(v,u);
            low[u]=min(low[u],low[v]);
            if (low[v]>=dfn[u]) {
                bcc++;
                while(s[top+1]!=v) ans[bcc].push_back(s[top--]);
                ans[bcc].push_back(u);
            }
        }
        else if (v!=fa) low[u]=min(low[u],dfn[v]);
    }
    if (fa==0&&son==0) ans[++bcc].push_back(u);
}
void solve() {
    for(int i=1;i<=n;i++) if (!dfn[i]) {
        top=0; dfncnt=0; tarjan(i,0);
    }
}
```




## 2-SAT

2-SAT，给出 $n$ 个集合，每个集合有 $2$ 个元素，已知若干个 $\langle a,b \rangle$，表示 $a$ 与 $b$ 矛盾（其中 $a$ 与 $b$ 属于不同的集合）。从每个集合选择一个元素，判断能否一共选 $n$ 个两两不矛盾的元素

我们对 $x$ 和 $y$ 连接有向边 $x \rightarrow y$，当且仅当选择 $x$ 后必定要选择 $y$

对该图缩点。如果每一个集合的 $x_i$ 和 $x_i'$ 在缩点后不在同一连通块中，则至少存在一种方案。

对于一种构造方案，可以通过构造 DAG 的反图后在反图上进行拓扑排序实现，能选则选；也可以根据 tarjan 缩点后，所属连通块编号越小，节点越靠近叶子节点这一性质，优先对所属连通块编号小的节点进行选择。



## 网络流

### 最大流

#### Dinic

Dinic 有时间复杂度上界 $O(n^2m)$

- 当所有边的容量为 $1$ 时，复杂度 $O(m\min(n^{\frac{2}{3}},m^{\frac{1}{2}}))$
- 如果存在一个割集使得每条边的容量为 $1$，则复杂度围为 $O(m^{\frac{3}{2}})$
  - 割集：连通图边集的子集，除去割集的边则图不连通
- 在单位网络中，复杂度为 $O(\sqrt{n}m)$
  - 单位网络：除了源点和汇点的所有带你都满足：只有一条入边且容量为 $1$、只有一条出边且容量为 $1$
  
```cpp
const int N=210,M=5010,INF=0x3f3f3f3f3f3f;

// 需要初始化 init(); 需要输入源点 S, 汇点 T, 边数 n
struct MF {
    struct edge{int v,nxt,cap,flow;} e[M<<1];
    int fir[N],dep[N],cur[N],cnt,maxflow,n,S,T;
    void init() {
        memset(fir,-1,sizeof(fir));
        cnt=0; maxflow=0;
    }
    void addedge(int u,int v,int w) {
        e[cnt]={v,fir[u],w,0}; fir[u]=cnt++;
        e[cnt]={u,fir[v],0,0}; fir[v]=cnt++;
    }
    void output() {
        for (int u=1;u<=n;++u)
            for (int j=fir[u];~j;j=e[j].nxt) if (e[j].cap)
                cout<<u<<' '<<e[j].v<<' '<<e[j].flow<<'\n'<<flush;
    }
    bool bfs() {
        queue<int> q; q.push(S);
        memset(dep,0,sizeof(int)*(n+1)); dep[S]=1; 
        while (q.size()) {
            int u=q.front(); q.pop();
            for (int i=fir[u];~i;i=e[i].nxt) {
                int v=e[i].v;
                if ((!dep[v])&&(e[i].cap>e[i].flow)) {
                    dep[v]=dep[u]+1; q.push(v);
                }
            }
        }
        return dep[T];
    }
    int dfs(int u,int flow) {
        if ((u==T)||(!flow)) return flow;
        int ret=0;
        for (int& i=cur[u];~i;i=e[i].nxt) {
            int v=e[i].v,d; cur[u]=i;
            if ((dep[v]==dep[u]+1)&&(d=dfs(v,min(flow-ret,e[i].cap-e[i].flow)))) {
                ret+=d; e[i].flow+=d; e[i^1].flow-=d;
                if (ret==flow) return ret;
            }
        }
        return ret;
    }
    void dinic() {
        while (bfs()) {
            memcpy(cur,fir,sizeof(int)*(n+1));
            maxflow+=dfs(S,INF);
        }
    }
}mf;
```



#### HLPP

复杂度 $O(n^2\sqrt{m})$

```cpp
const int N=1210,M=120010,INF=0x3f3f3f3f;
struct edge {int nex,t,v;}e[M<<1];
int n,m,s,t,h[N],ht[N],ex[N],gap[N],level=0,cnt=1;
stack<int> B[N];

void add_path(int f,int t,int v) {e[++cnt]={h[f],t,v},h[f]=cnt;}
void add_flow(int f,int t,int v) {add_path(f,t,v); add_path(t,f,0);}
int push(int u) {
    bool init=(u==s);
    for (int i=h[u];i;i=e[i].nex) {
        const int &v=e[i].t,&w=e[i].v;
        if (!w||(init==false&&ht[u]!=ht[v]+1)||ht[v]==INF) continue;
        int k=init?w:min(w,ex[u]);
        if (v!=s&&v!=t&&!ex[v]) B[ht[v]].push(v),level=max(level,ht[v]);
        ex[u]-=k,ex[v]+=k,e[i].v-=k,e[i^1].v+=k;
        if (!ex[u]) return 0;
    }
    return 1;
}
void relabel(int u) {
    ht[u]=INF;
    for (int i=h[u];i;i=e[i].nex)
        if (e[i].v) ht[u]=min(ht[u],ht[e[i].t]);
    if (++ht[u]<n) {
        B[ht[u]].push(u);
        level = max(level, ht[u]);
        ++gap[ht[u]];
    }
}
bool bfs_init() {
    memset(ht,0x3f,sizeof(ht));
    queue<int> q; q.push(t),ht[t] = 0;
    while (q.size()) {
        int u=q.front(); q.pop();
        for (int i=h[u];i;i=e[i].nex) {
            const int &v=e[i].t;
            if (e[i^1].v&&ht[v]>ht[u]+1) ht[v]=ht[u]+1,q.push(v);
        }
    }
    return ht[s]!=INF;
}
int select() {
    while (level>-1&&B[level].size()==0) level--;
    return level==-1?0:B[level].top();
}

int hlpp() {
    if (!bfs_init()) return 0;
    memset(gap,0,sizeof(gap));
    for (int i=1;i<=n;i++)
        if (ht[i]!=INF) gap[ht[i]]++;
    ht[s]=n; push(s);
    int u;
    while ((u=select())) {
        B[level].pop();
        if (push(u)) {
            if (!--gap[ht[u]])
                for (int i=1;i<=n;i++)
                    if (i!=s&&ht[i]>ht[u]&&ht[i]<n+1)
                        ht[i]=n+1;
            relabel(u);
        }
    }
    return ex[t];
}
```

#### 多源汇最大流

```cpp
const int N = 10010, M = (100000 + N) * 2, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    int sc, tc;
    scanf("%d%d%d%d", &n, &m, &sc, &tc);
    S = 0, T = n + 1;
    memset(h, -1, sizeof h);
    while (sc -- )
    {
        int x;
        scanf("%d", &x);
        add(S, x, INF);
    }
    while (tc -- )
    {
        int x;
        scanf("%d", &x);
        add(x, T, INF);
    }

    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    printf("%d\n", dinic());
    return 0;
}
```




### 最小割

对于一个网络流图 $G=(V,E)$，**割** 为一种点的划分方式：将所有的点划分为 $S$ 和 $T=V-S$ 两个集合，其中源点 $s\in S$，汇点 $t\in T$

割 $(S,T)$ 的**容量** $c(S,T)$ 表示所有从 $S$ 到 $T$ 的边的容量之和，即 $c(S,T)=\sum_{u\in S,v\in T}c(u,v)$

**最小割**就是求得一个割 $(S,T)$ 使得割的容量 $c(S,T)$ 最小

**最小割等于最大流**。对于方案，我们可以通过从源点 $s$ 开始 DFS，每次走残量大于 $0$ 的边，找到所有 $S$ 点集内的点。

如果需要在最小割的前提下最小化割边数量，那么先求出最小割，把没有满流的边容量改成 $\infty$，满流的边容量改成 $1$，重新跑一遍最小割就可求出最小割边数量

**最大权值闭合图**，即给定一张有向图，每个点都有一个权值（可以为正或负或 $0$），你需要选择一个权值和最大的子图，使得子图中每个点的后继都在子图中。

做法：建立超级源点 $s$ 和超级汇点 $t$，若节点 $u$ 权值为正，则 $s$ 向 $u$ 连一条有向边，边权即为该点点权；若节点 $u$ 权值为负，则由 $u$ 向 $t$ 连一条有向边，边权即为该点点权的相反数。原图上所有边权改为 $\infty$ 。跑网络最大流，将所有正权值之和减去最大流，即为答案。



### 费用流

#### SSP

最坏情况下 $O(mnf)$，其中 $f$ 为最大流

```cpp
const int N=5e3+10,M=1e5+10,INF=0x3f3f3f3f;
int n,m,tot=1,lnk[N],cur[N],ter[M],nxt[M],cap[M],cost[M],dis[N],ret;
bool vis[N];

void add(int u,int v,int w,int c) {ter[++tot]=v,nxt[tot]=lnk[u],lnk[u]=tot,cap[tot]=w,cost[tot]=c;}
void addedge(int u,int v,int w,int c) {add(u,v,w,c),add(v,u,0,-c);}
bool spfa(int s, int t) {
    memset(dis,0x3f,sizeof(dis));
    memcpy(cur,lnk,sizeof(lnk));
    queue<int> q;
    q.push(s), dis[s]=0,vis[s]=1;
    while (!q.empty()) {
        int u=q.front(); q.pop(),vis[u]=0;
        for (int i=lnk[u];i;i=nxt[i]) {
            int v=ter[i];
            if (cap[i]&&dis[v]>dis[u]+cost[i]) {
                dis[v]=dis[u]+cost[i];
                if (!vis[v]) q.push(v),vis[v]=1;
            }
        }
    }
    return dis[t]!=INF;
}
int dfs(int u,int t,int flow) {
    if (u==t) return flow;
    vis[u]=1; int ans=0;
    for (int &i=cur[u];i&&ans<flow;i=nxt[i]) {
        int v=ter[i];
        if (!vis[v]&&cap[i]&&dis[v]==dis[u]+cost[i]) {
            int x=dfs(v,t,min(cap[i],flow-ans));
            if (x) ret+=x*cost[i],cap[i]-=x,cap[i^1]+=x,ans+=x;
        }
    }
    vis[u]=0; return ans;
}
int mcmf(int s,int t) {
    int ans=0;
    while (spfa(s,t)) {
        int x; while ((x=dfs(s,t,INF))) ans+=x;
    }
    return ans;
}
```



#### Primal-Dual 原始对偶

最坏情况下 $O(nm+fm\log m)$，其中 $f$ 为最大流

```cpp
struct mypair {
    int dis,id;
    bool operator<(const mypair& a) const {return dis>a.dis;}
    mypair(int d,int x) {dis=d,id=x;}
};
const int N=5010,M=1e5+10;
struct edge{int v,f,c,next;}e[M];
struct node{int v,e;}p[N];
int head[N],dis[N],vis[N],h[N];
int n,m,s,t,cnt=1,maxf, minc;

void addedge(int u,int v,int f,int c) {e[++cnt].v=v;e[cnt].f=f;e[cnt].c=c;e[cnt].next=head[u];head[u]=cnt;}
void add(int u,int v,int f,int c) {addedge(u,v,f,c);addedge(u,v,0,-c);}
bool dijkstra() {
    priority_queue<mypair> q;
    for (int i=1;i<=n;i++) dis[i]=INF;
    memset(vis,0,sizeof(vis));
    dis[s]=0;
    q.push(mypair(0,s));
    while (!q.empty()) {
        int u=q.top().id; q.pop();
        if (vis[u]) continue;
        vis[u]=1;
        for (int i=head[u];i;i=e[i].next) {
            int v=e[i].v,nc=e[i].c+h[u]-h[v];
            if (e[i].f&&dis[v]>dis[u]+nc) {
                dis[v]=dis[u]+nc; p[v].v=u; p[v].e=i;
                if (!vis[v]) q.push(mypair(dis[v],v));
            }
        }
    }
    return dis[t]!=INF;
}
void spfa() {
    queue<int> q; q.push(s);
    memset(h,63,sizeof(h));
    h[s]=0,vis[s]=1;
    while (!q.empty()) {
        int u=q.front(); q.pop();
        vis[u]=0;
        for (int i=head[u];i;i=e[i].next) {
            int v=e[i].v;
            if (e[i].f&&h[v]>h[u]+e[i].c) {
                h[v]=h[u]+e[i].c;
                if (!vis[v]) {
                    vis[v]=1;
                    q.push(v);
                }
            }
        }
    }
}
void GGGG() {
    spfa();
    while (dijkstra()) {
        int minf=INF;
        for (int i=1;i<=n;i++) h[i]+=dis[i];
        for (int i=t;i!=s;i=p[i].v) minf=min(minf,e[p[i].e].f);
        for (int i=t;i!=s;i=p[i].v) {
            e[p[i].e].f-=minf;
            e[p[i].e^1].f+=minf;
        }
        maxf+=minf;
        minc+=minf*h[t];
  }
}
```

#### 有负环的费用流

```cpp
constexpr int inf=1000000000;
namespace net{
    int cnt,lim,h[205],hc[205],dis[205],f[100005],w[100005],to[100005],nxt[100005];
    bool on[205];
    queue<int> q;
    void lockstar(int x){lim=x;}//封锁下标大于 x 的边 
    void reset(){cnt=1,lim=inf;}
    inline void addstar(int x,int y,int _f,int _w){
        f[++cnt]=_f,w[cnt]=_w,to[cnt]=y,nxt[cnt]=h[x],h[x]=cnt;
        f[++cnt]=0,w[cnt]=-_w,to[cnt]=x,nxt[cnt]=h[y],h[y]=cnt;
    }
    bool SPFA(int x,int y){
        memset(dis,63,sizeof(dis)),dis[x]=0;
        for(q.push(x);!q.empty();q.pop()){
            int now=q.front();
            on[now]=0;
            for(int i=h[now];i;i=nxt[i]) if(i<=lim&&f[i]&&dis[to[i]]>dis[now]+w[i]){
                dis[to[i]]=dis[now]+w[i];
                if(!on[to[i]]) on[to[i]]=1,q.push(to[i]);
            }
        }
        return dis[y]<inf;
    }
    int dfs(int x,int flow,int aim){
        if(x==aim||!flow) return flow;
        int ret=0;
        on[x]=1;
        for(int &i=hc[x];i;i=nxt[i]) if(i<=lim&&!on[to[i]]&&dis[to[i]]==dis[x]+w[i]){
            int now=dfs(to[i],min(flow,f[i]),aim);
            ret+=now,f[i]-=now,f[i^1]+=now,flow-=now;
            if(!flow) break;
        }
        on[x]=0;
        return ret;
    }
    pair<int,int> SSP(int S,int T){
        pair<int,int> ret;
        while(SPFA(S,T)){
            memcpy(hc,h,sizeof(hc));
            int flow=dfs(S,inf,T);
            ret.first+=flow;
            ret.second+=dis[T]*flow;
        }
        return ret;
    }
}
//强制满流负权边 连反向正权边退流 限制退流流量
//有源汇上下界最小费用可行流解决
int n,m,S,T,SS,TT,ans,mem,maxflow,w[205];
int main(){
    scanf("%d%d%d%d",&n,&m,&S,&T);
    net::reset(); 
    for(int i=1;i<=m;i++){
        static int x,y,f,_w;
        scanf("%d%d%d%d",&x,&y,&f,&_w);
        if(!f) continue;//要你何用
        if(_w>=0) net::addstar(x,y,f,_w);
        else{//下界0 上界f 费用取反
            w[x]-=f,w[y]+=f,ans+=_w*f;
            net::addstar(y,x,f,-_w);
        }
    }
    SS=n+1,TT=SS+1,mem=net::cnt;
    for(int i=1;i<=n;i++){
        if(w[i]>0) net::addstar(SS,i,w[i],0);
        if(w[i]<0) net::addstar(i,TT,-w[i],0);
    }
    net::addstar(T,S,inf,0);
    auto SSP=net::SSP(SS,TT);//可行流
    maxflow=net::f[net::cnt];//拿到可行流实际流量
    ans+=SSP.second;//拿到可行流费用
    net::lockstar(mem);//封边
    SSP=net::SSP(S,T);//在原图上跑
    maxflow+=SSP.first,ans+=SSP.second;
    printf("%d %d\n",maxflow,ans);
}
```

### 上下界网络流

#### 无源汇上下界可行流

```cpp
const int N = 210, M = (10200 + N) * 2, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], l[M], ne[M], idx;
int q[N], d[N], cur[N], A[N];

void add(int a, int b, int c, int d)
{
    e[idx] = b, f[idx] = d - c, l[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    scanf("%d%d", &n, &m);
    S = 0, T = n + 1;
    memset(h, -1, sizeof h);
    for (int i = 0; i < m; i ++ )
    {
        int a, b, c, d;
        scanf("%d%d%d%d", &a, &b, &c, &d);
        add(a, b, c, d);
        A[a] -= c, A[b] += c;
    }

    int tot = 0;
    for (int i = 1; i <= n; i ++ )
        if (A[i] > 0) add(S, i, 0, A[i]), tot += A[i];
        else if (A[i] < 0) add(i, T, 0, -A[i]);

    if (dinic() != tot) puts("NO");
    else
    {
        puts("YES");
        for (int i = 0; i < m * 2; i += 2)
            printf("%d\n", f[i ^ 1] + l[i]);
    }
    return 0;
}
```

#### 有源汇上下界最大流


```cpp
const int N = 210, M = (N + 10000) * 2, INF = 1e8;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N], A[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    int s, t;
    scanf("%d%d%d%d", &n, &m, &s, &t);
    S = 0, T = n + 1;
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c, d;
        scanf("%d%d%d%d", &a, &b, &c, &d);
        add(a, b, d - c);
        A[a] -= c, A[b] += c;
    }

    int tot = 0;
    for (int i = 1; i <= n; i ++ )
        if (A[i] > 0) add(S, i, A[i]), tot += A[i];
        else if (A[i] < 0) add(i, T, -A[i]);

    add(t, s, INF);
    if (dinic() < tot) puts("No Solution");
    else
    {
        int res = f[idx - 1];
        S = s, T = t;
        f[idx - 1] = f[idx - 2] = 0;
        printf("%d\n", res + dinic());
    }

    return 0;
}
```

#### 有源汇上下界费用流

建给出的边时加入费用，额外的边时加入费用 $0$。跑费用流。

#### 有源汇上下界最小流

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 50010, M = (N + 125003) * 2, INF = 2147483647;

int n, m, S, T;
int h[N], e[M], f[M], ne[M], idx;
int q[N], d[N], cur[N], A[N];

void add(int a, int b, int c)
{
    e[idx] = b, f[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
    e[idx] = a, f[idx] = 0, ne[idx] = h[b], h[b] = idx ++ ;
}

bool bfs()
{
    int hh = 0, tt = 0;
    memset(d, -1, sizeof d);
    q[0] = S, d[S] = 0, cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int ver = e[i];
            if (d[ver] == -1 && f[i])
            {
                d[ver] = d[t] + 1;
                cur[ver] = h[ver];
                if (ver == T) return true;
                q[ ++ tt] = ver;
            }
        }
    }
    return false;
}

int find(int u, int limit)
{
    if (u == T) return limit;
    int flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int ver = e[i];
        if (d[ver] == d[u] + 1 && f[i])
        {
            int t = find(ver, min(f[i], limit - flow));
            if (!t) d[ver] = -1;
            f[i] -= t, f[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

int dinic()
{
    int r = 0, flow;
    while (bfs()) while (flow = find(S, INF)) r += flow;
    return r;
}

int main()
{
    int s, t;
    scanf("%d%d%d%d", &n, &m, &s, &t);
    S = 0, T = n + 1;
    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c, d;
        scanf("%d%d%d%d", &a, &b, &c, &d);
        add(a, b, d - c);
        A[a] -= c, A[b] += c;
    }

    int tot = 0;
    for (int i = 1; i <= n; i ++ )
        if (A[i] > 0) add(S, i, A[i]), tot += A[i];
        else if (A[i] < 0) add(i, T, -A[i]);

    add(t, s, INF);

    if (dinic() < tot) puts("No Solution");
    else
    {
        int res = f[idx - 1];
        S = t, T = s;
        f[idx - 1] = f[idx - 2] = 0;
        printf("%d\n", res - dinic());
    }

    return 0;
}
```


## 欧拉图

- 欧拉回路：通过图中每条边恰好一次的回路
- 欧拉通路：通过图中每条边恰好一次的通路
- 欧拉图：具有欧拉回路的图
- 半欧拉图：具有欧拉通路但不具有欧拉回路的图

无向图是欧拉图当且仅当：
- 非零度顶点是连通的
- 顶点的度数都是偶数

无向图是半欧拉图当且仅当：
- 非零度顶点是连通的
- 恰有 2 个奇度顶点

有向图是欧拉图当且仅当：
- 非零度顶点是强连通的
- 每个顶点的入度和出度相等

有向图是半欧拉图当且仅当：
- 非零度顶点是弱连通的（本身不是强连通，且如果有向边是无向的则连通）
- 至多一个顶点的出度与入度之差为 1
- 至多一个顶点的入度与出度之差为 1
- 其他顶点的入度和出度相等

```cpp
const int N = 2e5 + 5;
int n, m, lar, top, stc[N], in[N], hd[N];
vector<int> e[N];
void dfs(int id) {
  for(int &i = hd[id]; i < e[id].size(); ) dfs(e[id][i++]);
  stc[++top] = id;
}
int main() {
  cin >> n >> m;
  for(int i = 1; i <= m; i++) {
    int u, v;
    scanf("%d%d", &u, &v);
    e[u].push_back(v), in[v]++;
  }
  for(int i = 1; i <= n; i++) {
    sort(e[i].begin(), e[i].end());
    if(abs((int) e[i].size() - in[i]) > 1) puts("No"), exit(0);
    if(e[i].size() > in[i])
      if(lar) puts("No"), exit(0);
      else lar = i;
  }
  dfs(lar ? lar : 1);
  if(top != m + 1) puts("No");
  else {
    reverse(stc + 1, stc + top + 1);
    for(int i = 1; i <= top; i++) cout << stc[i] << " ";
  }
}
```



## 二分图相关

### 最大匹配

转换成网络流模型。

将源点连上左边所有点，右边所有点连上汇点，容量皆为 $1$。原来的每条边从左往右连边，容量也皆为 $1$，最大流即最大匹配。

如果使用 Dinic 算法 求该网络的最大流，可在 $O(\sqrt{n}m) $ 求出。是网络流中讨论情况的第三种。


也可以使用匈牙利算法，复杂度 $O(n^3)$

#### 匈牙利

```cpp
#define pb push_back
const int N=510;
int n,m,e,ans,vis[N],match[N];
vector<int> a[N];

inline bool dfs(int u) {
    for (auto v:a[u]) if (!vis[v]) {
        vis[v]=1;
        if (!match[v]||dfs(match[v])){
            match[v]=u;
            return 1;
        }
    }
    return 0;
}
void solve() {
    for (int i=1;i<=m;i++) {
        int u,v; cin>>u>>v;
        e[u].pb(v);
    }
    for (int i=1;i<=n;i++) {
        memset(vis,0,sizeof(vis));
        dfs(i);
    }
}
```



### 最小点覆盖

对于图 $G=(V, E)$，若 $V'\subseteq V$ 且 $\forall e\in E$ 满足 $e$ 的至少一个端点在 $V'$ 中，则称 $V'$ 是图 $G$ 的一个点覆盖。

二分图的最小点覆盖等于最大匹配数。对于方案，对于每一个任意选取一个点作为点覆盖集合中的点。

### 最大独立集

对于图 $G=(V, E)$，若 $V'\subseteq V$ 且 $V'$ 中任意两点都不相邻，则 $V'$ 是图 $G$ 的一个独立集。

最大独立集等于点数减去最大匹配数。二分图的最大独立集是最小点覆盖所选取的点集的补集。



### 最大权匹配

转换成网络流模型。

首先，在图中新增一个源点和一个汇点。从源点向二分图的每个左部点连一条流量为 $1$，费用为 $0$ 的边，从二分图的每个右部点向汇点连一条流量为 $1$，费用为 $0$ 的边。

接下来对于二分图中每一条连接左部点 $u$ 和右部点 $v$，边权为 $w$ 的边，则连一条从 $u$ 到 $v$，流量为 $1$，费用为 $w$ 的边。

另外，考虑到最大权匹配下，匹配边的数量不一定与最大匹配的匹配边数量相等，因此对于每个左部点，还需向汇点连一条流量为 $1$，费用为 $0$ 的边。

求这个网络的 最大费用最大流 即可得到答案。此时，该网络的最大流量一定为左部点的数量，而最大流量下的最大费用即对应一个最大权匹配方案。

用 SSP 算法求解费用流，复杂度为 $O(n^2m)$；用 Primal-Dual 原始对偶算法求解，复杂度为 $O(n^3\log n)$

KM 算法，复杂度为 $O(n^3)$。

考虑到二分图中两个集合中的点并不总是相同，为了能应用 KM 算法解决二分图的最大权匹配，需要先作如下处理：将两个集合中点数比较少的补点，使得两边点数相同，再将不存在的边权重设为 $0$，这种情况下，问题就转换成求最大权完美匹配问题，从而能应用 KM 算法求解。

#### KM

```cpp
#define int long long
const int N=505,inf=1e18;
int n,m,e[N][N],matched[N],slack[N],pre[N],ex[N],ey[N];
bool visx[Maxn],visy[Maxn];

void match(int u) { 
    int x,y=0,yy=0,delta;
    memset(pre,0,sizeof(pre));
    for (int i=1;i<=n;i++) slack[i]=inf;
    matched[y]=u;
    while (1) { 
        x=matched[y];delta=inf;visy[y]=1;
        for (int i=1;i<=n;i++) {    
            if (visy[i]) continue;
            if (slack[i]>ex[x]+ey[i]-e[x][i]) { 
                slack[i]=ex[x]+ey[i]-e[x][i];
                pre[i]=y;
            }
            if (slack[i]<delta) {delta=slack[i];yy=i;}
        }
        for (int i=0;i<=n;i++) {    
            if(visy[i])ex[matched[i]]-=delta,ey[i]+=delta;
            else slack[i]-=delta;
        }
        y=yy;
        if (matched[y]==-1) break;
    }
    while (y) {matched[y]=matched[pre[y]];y=pre[y];}
}
int KM() {  
    memset(matched,-1,sizeof(matched));
    memset(ex,0,sizeof(ex));
    memset(ey,0,sizeof(ey));
    for(int i=1;i<=n;i++) { 
        memset(visy,0,sizeof(visy));
        match(i);
    }
    int res=0;
    for (int i=1;i<=n;i++)
        if (matched[i]!=-1) res+=e[matched[i]][i];
    return res;
}
void solve() {
    memset(e,sizeof(e),-inf);
    for (int i=1;i<=m;i++) {
        int u,v,w; cin>>u>>v>>w;
        e[u][v]=w;
    }
    KM(); // match[i]: i 匹配的人
}
```



## 杂

### 树的直径

树上任意两节点之间最长的简单路径即为树的直径。可以通过 DP 求解。

若树上所有边边权均为正，则树的所有直径中点重合

### 树的重心

如果在树中选择某个节点并删除，这棵树将分为若干棵子树，统计子树节点数并记录最大值。取遍树上所有节点，使此最大值取到最小的节点被称为整个树的重心。

- 树的重心如果不唯一，则至多有两个，且这两个重心相邻
- 以树的重心为根时，所有子树的大小都不超过整棵树大小的一半
- 树中所有点到某个点的距离和中，到重心的距离和是最小的；如果有两个重心，那么到它们的距离和一样
- 把两棵树通过一条边相连得到一棵新的树，那么新的树的重心在连接原来两棵树的重心的路径上
- 在一棵树上添加或删除一个叶子，那么它的重心最多只移动一条边的距离

求树的重心可以预先设置一个根节点，求所有点子树的大小。对于节点 ``u`` 为根的情况，另一边子树的大小为 ``n-siz[u]``。

### 基环树找环

```cpp
const int N=1e5+10;
int in[N];
vector<int> e[N],loop;

void topo() {
    for (int i=1;i<=n;i++) {
        int u,v; cin>>u>>v; ++in[u],++in[v];
        e[u].pb(v),e[v].pb(u);
    }
    queue<int> q;
    for (int i=1;i<=n;i++) if (in[i]==1) q.push(i);
    while (!q.empty()) {
        int u=q.top(); q.pop();
        for (int v:e[u]) {
            --in[v];
            if (in[v]==1) q.push(v);
        }
    }
    for (int i=1;i<=n;i++) if (in[v]>=2) loop.pb(i);
}
```
