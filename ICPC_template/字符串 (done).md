
# 字符串

## Hash

### 低碰撞质数

| $p\le 10^3$ | $p\le 10^4$ | $p\le 10^5$ | 
| --- | --- | --- |
| $131,163,193,389,769$ | $1543,3079,6151$ | $12289,13331,24593,49157,98317$  |

| $p\le 10^6$ | $p\le 10^7$ | 
| --- | --- |
| $196613,393241,786433$ | $1572869,3145739,6291469,9905411$ |

| $p\le 10^8$ | $p\le 10^9$ | $p\le 10^{10}$ |
| --- | --- | --- |
| $12582917,25165843,50331653$ | $402653189,998244353$ | $10^9+7,10^9+9,1610612741$ |

### 封装多哈希

```cpp
const int N=1e5+10;

struct hash{
    int p[N],h[N];
    void init(int base,int mod,string s) {
        int n=s.size(); --n; p[0]=1;
        for (int i=1;i<=n;i++) p[i]=p[i-1]*base%mod;
        for (int i=1;i<=n;i++) h[i]=(h[i-1]*base+s[i])%mod;
    }
    int gethash(int l,int r) {
        return ((h[r]-h[l-1]*p[r-l+1])%mod+mod)%mod;
    }
}h[2];
bool equ(int l1,int r1,int l2,int r2) {
    return h[0].gethash(l1,r1)==h[0].gethash(l2,r2)&&h[1].gethash(l1,r1)==h[1].gethash(l2,r2);
}
```

### 取模优化

```cpp
struct Bart{
	int I,m;
	void ini(int Mod) {m=Mod; I=(1ll<<62)/m;}
	int operator()(int x) {
		int tp=x-((__int128)x*I>>62)*m;
		while (tp>=m) tp-=m;
		while (tp<0) tp+=m;
		return tp;
	}
};
```

### 二分哈希

```cpp
int lcp(int x,int y) {
	if (s[x]!=s[y]) return 0;
	int l=0,r=min(n-x,n-y);
	while (l<r) {
		int mid=r-((r-l)>>1);
		if (equ(x,x+mid,y,y+mid)) l=mid;
		else r=mid-1;
	} return l+1;
}
bool cmp(int x,int y) {
	if (s[x]!=s[y]) return s[x]<s[y];
	int l=lcp(x,y);
	if (l!=min(n-x,n-y)+1) return s[x+l]<s[y+l];
	else return x<y;
}
```

## KMP

求模式串 $s$  中 $s[1,i]$ 的 border

求匹配串 $str$ 出现 $s$ 的位置

```cpp
const int N=1e6+10;
string s,str;
int n,m,fail[N];

void kmp() {
	for (int i=2,j=0;i<=n;i++) {
		while (j&&s[i]!=s[j+1]) j=fail[j];
		if (s[j+1]==s[i]) ++j;
		fail[i]=j;
	}
	for (int i=1,j=0;i<=m;i++) {
		while (j&&str[i]!=s[j+1]) j=fail[j];
		if (s[j+1]==str[i]) ++j;
		if (j==n) j=fail[j]; // A match complete in str[j-n+1,j]
	}
}
```

## Border Theory & 区间 Border

若 $0< p \leq |s|$，对于 $1\leq i\leq |s| - p$ 有 $s_i=s_{i+p}$，就称 $p$ 为 $s$ 的一个周期，最小周期写作 $per(u)$。

若 $0\leq r < |s|$，$pre(s, r) = suf(s, r)$，就称 $pre(s, r)$ 是 $s$ 的 $\text{border}$。

一个显然的结论：

> $s_{1..k}$ 为 $s$ 的 $\text{border}\rightleftharpoons|s| - k$ 为 $s$ 的周期。

$\textbf{Weak Periodicity Lemma}$

> 若 $p$ 和 $q$ 是 $s$ 的周期，$p+q\leq |s|$，则 $\gcd(p, q)$ 也是 $s$ 的周期。

证明：令 $d=q-p\ (q > p)$， $i - p < 0$ 和 $i + q \leq |s|$ 至少有一个成立，于是要么得到 $s_i=s_{i-p}=s_{i-p+q}=s_d$ 要么得到 $s_i=s_{i+q}=s_{i+q-p}=s_d$。

于是 $d$ 和 $p$ 都是 $s$ 的周期，$d+p\leq |s|$，发现这是一个更相减损的过程，最后便会得到 $\gcd(p, q)$ 也为 $s$ 的周期。

$\textbf{Periodicity Lemma}$

> 若 $p$ 和 $q$ 是 $s$ 的周期，$p+q-\gcd(p, q)\leq |s|$，则 $\gcd(p, q)$ 也是 $s$ 的周期。

字符串匹配引理：

> 字符串 $u$ 和 $v$ 满足 $2|u| \geq |v|$，则 $u$ 在 $v$ 中的所有匹配位置组成一个等差数列，公差为 $per(u)$。

证明：只考虑匹配次数大于 $2$ 的情况。$2|u|\geq|v|$ 表明了两次匹配之间 $u$ 的位置必有交，而两次位置之差必定为 $u$ 的一个周期。

考虑第一次第二次和最后一次匹配，根据 $\textbf{Weak Periodicity Lemma}$ 可知 $\gcd$(第一次和第二次距离之差，第二次和最后一次距离之差) 为 $u$ 的周期。

通过简单的反证法可以得出它便是 $u$ 的最小周期。从而得到 $v$ 在此的结构为 $u$ 的最小周期的不断重复。

$\textbf{Border Theory}$

> 字符串 $s$ 的所有 $\text{border}$ 长度排序后可分成 $O(\log |s|)$ 段，每段是一个等差数列。

证明：

**引理**

> 字符串 $s$ 的所有不小于 $\frac{|s|}2$ 的 $\text{border}$ 长度组成一个等差数列。

证明：有等价的说法为本质不同的长度小于 $\frac{|s|}2$ 的 $s$ 的周期至多只有一个。用简单的反证法与 $\textbf{Weak Periodicity Lemma}$ 可以得到。

现在我们将 $\text{border}$ 按照长度分类，分为 $[1, 2), [2, 4), [4, 8), \ldots [2^{k-1}, 2^k), [2^k, |s|)$ 这些集合。

对于长度相同的两个字符串 $u, v$，记 $PS(u, v)=\{k|pre(u, k)=suf(v, k)\}$。

记 $LargePS(u, v)=\{k\in PS(u, v)|k \geq \frac{|u|}2\}$。

则一个 $\text{border}$ 集合可以表示为 。$LargePS(pre(s, 2^i), suf(s, 2^i))$。

**引理**

> $LargePS(u, v)$ 组成一个等差数列。

证明：$\text{border}$ 的 $\text{border}$ 是 $\text{border}$。对于这个集合里最长的 $\text{border}$ 而言，不小于它长度一半的 $\text{border}$ 长度组成一个等差数列。

### 区间 Border

我们设给定的子串为 $s$。

先考虑 Border 长度为 $[2^{i-1}, 2^i)$ 的集合。

Border 的集合即为 $pre(s, 2^{i-1})$ 在 $suf(s, 2^i)$ 中的匹配位置与 $suf(s, 2^{i-1})$ 在 $pre(s, 2^i)$ 中的匹配位置的交集（移位后）。

而匹配位置为一个等差数列。我们只用求第一次匹配第二次匹配与最后一次匹配便可知道这个等差数列的首项、公差、项数。相当于实现一个 $succ(v, i)$ 和 $pred(v, i)$ 表示求 $v$ 从位置 $i$ 开始向前/后的第一次匹配位置。

问题变为 $str$ 中所有长度为 $2^{i-1}$ 的串有哪些位置能够与之匹配。

发现这就是倍增求后缀数组的时候把每一轮结束后的结果都记录下来。

最后是长度为 $[2^k, |s|)$ 的集合。做法其实一样，把上面所有 $2^i$ 替换为 $|s|$ 即可。

于是我们就有了 $O(n\log n)$ 预处理，单次询问 $O(log^2n)$ 的优秀做法。

如何优化？考虑将 $O(log^2n)$ 降低。这时瓶颈为求出等差数列、等差数列求交。

等差数列求交的优化肯定建立在字符串性质上。

**引理**：四个字符串满足 $|x_1|=|y_1|\geq|x_2|=|y_2|$，且 $x_1$ 在 $y_2y_1$ 中至少匹配 $3$ 次，$y_1$ 在 $x_1x_2$ 中至少匹配 $3$ 次，则 $x_1$ 和 $y_1$ 的最小周期相等。

证明：否则不妨设 $per(x_1) > per(y_1)$，考虑 $x_1$ 在 $y_2y_1$ 中的最右边一次匹配, 设它与 $y_1$ 的重叠部分为 $z$。

则 $|z| \geq 2per(x_1) > per(x_1) + per(y_1)$, 则 $z$ 拥有周期 $d = \gcd (per(x_1), per(y_1))|per(x_1)$, 于是 $d$ 也是 $x_1$ 的周期。但 $d < per(x_1)$, 矛盾。

同样，$y_1$ 在 $x_1x_2$ 里至少匹配 $3$ 次保证了 $per(x_1)<per(y_1)$ 矛盾。

所以我们合并的两个等差数列要么长度不足 $3$，要么公差相同。所以可以 $O(1)$ 合并。

求出等差数列的优化如下。

注意到在求 $succ(v, i)$ 的时候，$|v|$ 是二的幂，我们只在意起点在 $[i, i+|v|]$ 的匹配。

我们把原串按 $2^k$ 为段长分成若干段，考虑长度为 $2^k$ 的串的匹配信息。这样 $[i, i + |v|]$ 的匹配最多跨越两个段。我们可以分别得到这两个段的信息，直接合并即可。用 $\text{hashtable}$ 维护。

这样就可以做到 $O(1)$ 求出等差数列。


## Z 函数

$z_i$ 表示 $s[i,n]$ 和 $s[1,n]$ 的 LCP

$p_i$ 表示 $s[1,n]$ 和 $s'[i,n]$ 的 LCP

```cpp
const int N=1e6+10;
string s,str;
int n,m,z[N],p[N];

void z_function() {
	z[1]=n;
	int l=0,r=0;
	for (int i=2;i<=n;i++) {
		if (i<=r) z[i]=min(z[i-l+1],r-i+1);
		while (i+z[i]<=n&&s[i+z[i]]==s[z[i]+1]) ++z[i];
		if (i+z[i]-1>r) l=i,r=i+z[i]-1;
	}
}
void exkmp() {
	int l=0,r=0;
	for (int i=1;i<=m;i++) {
		if (i<=r) p[i]=min(z[i-l+1],r-i+1);
		while (i+p[i]<=m&&p[i]+1<=n&&str[i+p[i]]==s[p[i]+1]) ++p[i];
		if (i+p[i]-1>r) l=i,r=i+p[i]-1;
	}
}
```

## Manacher

$p_i$ 表示以 $i$ 为中心的最长回文串为 $s[i-p_i,i+p_i]$

```cpp
const int N=1e6+10;
string str,_str;
int n,p[N<<1];

void manacher() {
	cin>>_str; n=_str.size();
	str.pb('~');
	for (auto c:_str) str.pb('#'),str.pb(c);
	str.pb('#'); n=n<<1;
	int r=0;
	for (int i=1;i<=n;i++) {
		if (p[r]+r>=i) p[i]=min(p[r*2-i],r+p[r]-i)
		else p[i]=0;
		while (str[i-p[i]-1]==str[i+p[i]+1]) p[i]++;
		if (i+p[i]>r+p[r]) r=i;
	}
}
```

## 序列自动机

向前添加、字符集大小 $|\Sigma^*|$ 较小

```cpp
int n,a[N],nxt[N][26];

void solve(){
    for (int i=0;i<26;i++) nxt[n][i]=n+1;
    for (int i=n-1;i>=1;i--) {
        for (int j=0;j<26;j++) nxt[i][j]=nxt[i+1][j];
        nxt[i][a[i+1]]=i+1;
    }
}
```

向后添加、字符集大小 $|\Sigma^*|$ 较大

```cpp
const int N=1e5+10;
vector<int> v[N];
int n,a[N];

void pre() {
    // 字符集 可以离散化
    for (int i=1;i<=n;i++) v[a[i]].pb(i);
}
void ist(int x) {v[x].pb(++n);}
int nxt(int i,int x) {
    if (v[x].back()<=i) return n+1;
    int l=0,r=v[x].size()-1;
    while (l<r) {
        int mid=(l+r)>>1;
        if (v[x][mid]<=i) l=mid+1;
        else r=mid;
    }
    return v[x][l];
}
```

可以用主席树维护 $nxt_{i,s}$。从后往前做，每次单点修改并新建一个版本，单点查询

## ACAM

```cpp
const int N=1e5+10;
int cnt,n,vis[N][26],fail[N],match[N],siz[N];
vector<int> e[N];

void insert(string str,int id) {
	int rt=0;
	for (auto s:str) {
		if (!vis[rt][s-'a']) vis[rt][s-'a']=++cnt;
		rt=vis[rt][s-'a'];
	}
	match[id]=rt;
}
void get() {
	fail[0]=0; queue<int> q;
	for (int i=0;i<26;i++)
		if (vis[0][i]) fail[vis[0][i]]=0,q.push(vis[0][i]);
	while (!q.empty()) {
		int x=q.front(); q.pop();
		for (int i=0;i<26;i++) {
			if (vis[x][i]) fail[vis[x][i]]=vis[fail[x]][i],q.push(vis[x][i]);
			else vis[x][i]=vis[fail[x]][i];
		}
	}
}
void dfs(int u) {
	for (auto v:e[u]) {
		dfs(v); siz[u]+=siz[v];
	}
}
void solve(string str) {
	int rt=0;
	for (auto s:str) rt=vis[rt][s-'a'],++siz[rt];
	for (int i=1;i<=cnt;i++) e[fail[i]].push_back(i);
	dfs(0);
	for (int i=1;i<=n;i++) cout<<siz[match[i]];
}
```

## SA

$sa_i$ 表示排名为 $i$ 的后缀从 $sa_i$ 开始；$rk_i$ 表示从 $i$ 开始后缀的排名为 $rk_i$

$height_i$ 表示 $str[sa_i,n]$ 和 $str[sa_{i-1},n]$ 的 LCP

$sa_i$ 和 $sa_j$ 的 LCP 为 $\min(height_{i+1},\cdots,height_j)$

```cpp
const int N=1e5+10;
string str;
int n,rk[N*2],oldrk[N],sa[N*2],id[N],cnt[N],height[N];

void getsa() {
	int m=128; // m 为字符集值域
	for (int i=1;i<=n;i++) cnt[rk[i]=str[i]]++;
	for (int i=1;i<=m;i++) cnt[i]+=cnt[i-1];
	for (int i=n;i>=1;i--) sa[cnt[rk[i]]--]=i;
  	int p=0;
 	for (int w=1;;w<<=1,m=p) {
 		int cur=0;
 		for (int i=n-w+1;i<=n;i++) id[++cur]=i;
 		for (int i=1;i<=n;i++)
 			if (sa[i]>w) id[++cur]=sa[i]-w;
 		memset(cnt,0,sizeof(cnt));
 		for (int i=1;i<=n;i++) cnt[rk[i]]++;
 		for (int i=1;i<=m;i++) cnt[i]+=cnt[i-1];
 		for (int i=n;i>=1;i--) sa[cnt[rk[id[i]]]--]=id[i];
 		p=0; 
 		memcpy(oldrk,rk,sizeof(oldrk));
    	for (int i=1;i<=n;i++) {
    		if (oldrk[sa[i]]==oldrk[sa[i-1]]&&oldrk[sa[i]+w]==oldrk[sa[i-1]+w]) rk[sa[i]]=p;
    		else rk[sa[i]]=++p;
    	}
    	if (p==n) break;
    }
}
void getheight() {
	for (int i=1,k=0;i<=n;++i) {
		if (rk[i]==0) continue;
		if (k) --k;
		while (str[i+k]==str[sa[rk[i]-1]+k]) ++k;
		height[rk[i]]=k;
	}
}
```

## SAM

```cpp
#define int long long
const int N=1e5+10,M=30;
int tot=1,last=1,link[N<<1],ch[N<<1][M],len[N<<1],endpos[N<<1];
// 总点数tot，点的index属于[1-tot]，空串/根为1 
// link 为点的 parent 树父节点/最长出现位置与自己不同的后缀
// ch[n][s] 指节点n末尾加字符s所转移到的点
// len 指该节点的串的最长长度，最短长度等于 len[link[n]] + 1
// endpos[n] 每个点作为“终结点”的次数
void clear() {
    for (int i=0;i<=tot;i++){
        link[i]=len[i]=endpos[i]=0;
        for (int k=0;k<M;k++) ch[i][k]=0;
    }
    tot=1; last=1;
}
void extend(int w) {
    int p=++tot,x=last,r,q;
    endpos[p]=1;
    for (len[last=p]=len[x]+1;x&&!ch[x][w];x=link[x]) ch[x][w]=p;
    if (!x) link[p]=1;
    else if (len[x]+1==len[q=ch[x][w]]) link[p]=q;
    else {
        link[r=++tot]=link[q];
        memcpy(ch[r],ch[q],sizeof ch[r]);
        len[r]=len[x]+1;
        link[p]=link[q]=r;
        for(;x&&ch[x][w]==q;x=link[x]) ch[x][w]=r;
    }
}
vector<int> p[N<<1];
void dfs(int u) {
    for (auto v:p[u]) {
        dfs(v); endpos[u]+=endpos[v];
    }
}
// 使用该方法后，endpos[] 指在串中出现总次数，即原数组的子树求和
void get_endpos(){
    for (int i=1;i<=tot;i++) p[i].clear();
    for (int i=2;i<=tot;i++){
        p[link[i]].push_back(i);
    }
    dfs(1);
}

const int STC=998244353;
void self_test(){
    clear();
    for (int i=1;i<=1000;i++) extend(i*i%26+1);
    int tmp=107*last+301*tot;
    for (int i=1;i<=tot;i++) {
        tmp=(tmp*33+link[i]*101+len[i]*97)%STC;
        for (int k=1;k<M;k++) tmp=(tmp+k*ch[i][k])%STC;
    }
    assert("stage 1"&&tmp==393281314); // stage1 : 检查建树是否正确
    tmp=0; get_endpos();
    for(int i=1;i<=tot;i++) tmp=(tmp*33+endpos[i])%STC;
    assert("stage 2"&&tmp==178417668); // stage2 : 检查endpos计算是否正确
    cout<<"Self Test Passed. Remember to delete this function's use."<<endl;
    clear();
}
```

## PAM

回文串子串的一些性质：

- $t$ 是回文串 $s$ 的后缀，$t$ 是 $s$ 的 ``border`` 当且仅当 $t$ 是回文串
- $t$ 是回文串 $s$ 的 ``border``，则 $|s|-|t|$ 是 $s$ 的周期；$|s|-|t|$ 为 $s$ 的最小周期，当且仅当 $t$ 是 $s$ 的最长回文真后缀
- $t$ 是 $s$ 的最长回文真后缀，若 $|s|\mod |s|-|t|=0$，则 $|s|-|t|$ 为 $s$ 的最小完全周期；否则，$s$ 的最小完全周期为 $1$
- $s$ 的所有回文后缀按照长度排序后，可以划分成 $\log |s|$ 段等差数列

求解回文划分方案数。我们维护两个数组，$diff$ 和 $slink$
- $diff_u$ 表示节点 $u$ 和 $fail_u$ 所代表的回文串的长度差
- $slink_u$ 表示 $u$ 一直沿着 $fail$ 向上跳到第一个节点 $v$，使得 $diff_v \neq diff_u$

暴力跳 $slink_u$ 仅有 $\log n$ 的复杂度

我们在更新 $dp$ 数组的同时，更新数组 $g_u$，表示 $u$ 所在等差数列的 $dp$ 和，且 $u$ 是这个等差数列中长度最长的节点

```cpp
const int N=5e5+10;

struct pam {
    int cnt,to[N],fail[N],len[N],nxt[N][26],dep[N];
    string s; int siz;
    int create(int l) {
        ++cnt; memset(nxt[cnt],0,sizeof(nxt[cnt]));
        len[cnt]=l; return cnt;
    } 
    void init() {
        cnt=-1; siz=0; s.clear(); s.pb(' ');
        create(0); fail[0]=1; dep[0]=0;
        create(-1); fail[1]=1; dep[1]=0;
    }
    int getfail(int x) {
        while (s[siz-len[x]-1]!=s[siz]) x=fail[x];
        return x;
    }
    int ist(char ch) {
        ++siz; s.pb(ch); int i=getfail(to[siz-1]);
        if (!nxt[i][ch-'a']) {
            fail[cnt+1]=nxt[getfail(fail[i])][ch-'a'];
            nxt[i][ch-'a']=create(len[i]+2);
            dep[cnt]=dep[fail[cnt]]+1;
        }
        to[siz]=nxt[i][ch-'a'];
        return dep[to[siz]];
    }
    // 回文划分方案数
    int diff[N],slink[N],g[N],dp[N];
    void ist(char ch) {
        ++siz; s.pb(ch); int i=getfail(to[siz-1]);
        if (!nxt[i][ch-'a']) {
            fail[cnt+1]=nxt[getfail(fail[i])][ch-'a'];
            nxt[i][ch-'a']=create(len[i]+2);
            diff[cnt]=len[cnt]-len[fail[cnt]];
            if (diff[cnt]==diff[fail[cnt]]) slink[cnt]=slink[fail[cnt]];
            else slink[cnt]=fail[cnt];
        }
        to[siz]=nxt[i][ch-'a'];
        for (int j=to[siz];j>1;j=slink[j]) {
            if (slink[j]==fail[j]) g[j]=dp[siz-len[j]];
            else g[j]=g[fail[j]]+dp[siz-diff[j]-len[slink[j]]];
            g[j]%=mod; if (siz%2==0) dp[siz]+=g[j];
        } dp[siz]%=mod;
    }
}p;
```

> Written with [StackEdit中文版](https://stackedit.cn/).
