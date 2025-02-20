# 数论

## 巴雷特约减

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

## 上下取整与负数取模

记 $[x]$ 表示取整：

- 当 $x>0$ 时，$[x]=\lfloor x\rfloor$
- 当 $x<0$ 时，$[x]=\lceil x\rceil$

$[\frac{a}{b}]$ 即为 C++ 中的 ``a/b``

上下取整的转换：

- $a\ge 0,b>0$ 时，$\lfloor\frac{a}{b}\rfloor=[\frac{a}{b}],\lceil\frac{a}{b}\rceil=[\frac{a+b-1}{b}]$
- $a\le 0,b<0$ 时，$\lfloor\frac{a}{b}\rfloor=[\frac{a}{b}],\lceil\frac{a}{b}\rceil=[\frac{a+b+1}{b}]$
- $a\le 0,b>0$ 时，$\lfloor\frac{a}{b}\rfloor=[\frac{a-b+1}{b}],\lceil\frac{a}{b}\rceil=[\frac{a}{b}]$
- $a\ge 0,b<0$ 时，$\lfloor\frac{a}{b}\rfloor=[\frac{a-b-1}{b}],\lceil\frac{a}{b}\rceil=[\frac{a}{b}]$

```cpp
int floor(int a,int b) {
	if (a<0) return floor(-a,-b);
	if (b<0) return (a-b-1)/b;
	return a/b;
}
int ceil(int a,int b) {
	if (a<0) return ceil(-a,-b);
	if (b<0) return a/b;
	return (a+b-1)/b;
}
```

负数取模的定义：

- $a\% b=a\%(-b)$
- $a\% b=-(-a)\%b$

$a\% b$ 即为 C++ 中的 ``a%b``

## 龟速乘

```cpp
int mul(int x,int y,int mod) {
	int ans=0; while (y)  {
		if (y&1) ans=(ans+x)%mod;
		y>>=1; x=(x+x)%mod;
	} return ans;
}
```

## 快速幂

```cpp
int qpow(int x,int k,int mod) {
	int ans=1; while (k) {
		if (k&1) ans=(ans*x)%mod;
		k>>=1; x=(x*x)%mod;
	} return ans;
}
```

## 光速幂

固定模数 $p$ 和 $a$ 时，$O(\sqrt{p})$ 预处理，$O(1)$ 求 $a^b\mod p$

令 $b<2\varphi(p)$；当 $b\ge2\varphi(p)$ 时，根据欧拉定理，显然有 $a^b \equiv a^{\varphi(p)+b\mod\varphi(p)}$

当然，若 $p$ 是质数时，可以令 $b\le p-2$

设 $b=ks+t$，其中 $k=\lceil\sqrt{p}\rceil$

于是，我们只需要预处理出 $a^0,a^1,\cdots,a^{k-1}$，以及 $a^k,a^{2k},\cdots, a^{k^2}$ 即可

## EXGCD

求解 $ax+by=\gcd(a,b)(ab\not= 0)$ 的一组解 $(x_0,y_0)$

$$\begin{aligned}
    ({b\cdot[\frac{a}{b}]}+a\% b)x+by&=\gcd(b,a \% b)\\
    b\cdot({[\frac{a}{b}}]x+y)+(a\% b) x&=\gcd(b,a \% b)
\end{aligned}$$

边界：

$$
b=0 \Rightarrow x=1,y\in \mathbb{Z}
$$

递归即可。复杂度 $O(\log n)$

可以证明，边界取 $y=0$ 时，得到的解为 $|x_0|,|y_0|$ 均最小的解，且对于递归的任意时刻都有 $|x|\le \frac{1}{2}|b|, |y|\le \frac{1}{2}|a|$

通解为：

$$x=x_0+kb, y=y_0-ka, k\in \mathbb{Z}$$

对于方程 $ax+by=c$，有解当且仅当 $gcd(a,b)|c$。令 $d=\frac{c}{\gcd(a,b)}$，则有特解 $x_0'=dx_0,y_0'=dy_0$

$x$ 有最小非负整数解 $(x_0\% b+b)\%b$，$y$ 有最小非负整数解 $(y_0\% a+a)\%a$

若 $a,b>0$，求 $x,y$ 同为非负整数解的个数，注意到 $y_{\min}$ 时 $x_{\max}=\frac{c-by_{\min}}{a}$，可以求出 $x_{\min}$ 和 $x_{\max}$，解的个数为 $\frac{x_{\max}-{x_{\min}}}{b}+1$

```cpp
void exgcd(int &x,int &y,int a,int b) {
    if(!b) {x=1,y=0;return;}
    exgcd(x,y,b,a%b);
    int t=x; x=y; y=t-a/b*y;
}
```

## 逆元

当 $p$ 为质数时，$a^{p-1}\equiv 1\mod p$，$a^{-1}=a^{p-2}$

```cpp
int qpow(int x,int k,int mod) {
	int ans=1; while (k) {
		if (k&1) ans=(ans*x)%mod;
		k>>=1; x=(x*x)%mod;
	} return ans;
}
int inv(int x,int mod) {return qpow(a,mod-2,mod);}
```

当 $p$ 非质数时，$aa^{-1}\equiv1\mod p$，$aa^{-1}+py=1$，$\exist a^{-1}\iff \gcd(a,p)=1$

exgcd 求解方程 $ax+py=1$，$x=a^{-1}$

```cpp
void exgcd(int &x,int &y,int a,int b) {
    if(!b) {x=1,y=0;return;}
    exgcd(x,y,b,a%b);
    int t=x; x=y; y=t-a/b*y;
}
int inv(int a,int p) {
    if (__gcd(x,p)!=1) return -1;
    int x,y; exgcd(x,y,a,p); return (x%p+p)%p;
}
```

## CRT

求解如下形式的一元线性同余方程组（其中 $r_1, r_2, \cdots, r_k$ 两两互质）：

 
$$\begin{cases}
x &\equiv a_1 \pmod {r_1} \\
x &\equiv a_2 \pmod {r_2} \\
  &\vdots \\
x &\equiv a_k \pmod {r_k} \\
\end{cases}$$


```cpp
void exgcd(int &x,int &y,int a,int b) {
    if(!b) {x=1,y=0;return;}
    exgcd(x,y,b,a%b);
    int t=x; x=y; y=t-a/b*y;
}
int n,a[N],r[N];
int crt() {
    int p=1,ans=0;
    for (int i=1;i<=n;i++) p=p*r[i];
    for (int i=1;i<=n;i++) {
        int m=p/r[i],inv,y;
        exgcd(inv,y,m,r[i]);
        inv%=r[i];
        ans=(ans+a[i]*m*inv%p)%p;
    }
    return (ans+p)%p;
}
```

### EXCRT

$r_i$ 不互质的情况

```cpp
void exgcd(int &x,int &y,int a,int b) {
    if(!b) {x=1,y=0;return;}
    exgcd(x,y,b,a%b);
    int t=x; x=y; y=t-a/b*y;
}
ll n, a[N], r[N]
int excrt() {
    ll ans = a[1]; M = r[1];
    for(int i = 2; i <= n; ++i){
        ll B = ((a[i] - ans) % r[i] + r[i]) % r[i];
        ll GCD = exgcd(x, y, M, r[i]);
        x *= __int128(B / GCD) % r[i];
        ans += M * x;
        M *= r[i] / GCD;
        ans = (ans + M) % M;
    }
}
```

## BSGS

对 $a,b,m\in\mathbf{Z}^+$，该算法可以在 $O(\sqrt{m})$ 的时间内求解 $a^x \equiv b \pmod m$，其中 $a\perp m$。

令 $x = A \left \lceil \sqrt m \right \rceil - B$，其中 $0\le A,B \le \left \lceil \sqrt m \right \rceil$，则有 $a^{A\left \lceil \sqrt m \right \rceil -B} \equiv b \pmod m$，稍加变换，则有 $a^{A\left \lceil \sqrt m \right \rceil} \equiv ba^B \pmod m$.

```cpp
map<int,int> mp;
int bsgs(int a, int b, int p) {
    if(a % p == b % p) return 1;
    if(a % p == 0 && b) return -1;
    ll unit = (ll)ceil(sqrt(p)), tmp = qpow(a, unit, p);
    for(int i = 0; i <= unit; ++i)
        mp[b] = i, b = (b * a) % p;
    b = 1;
    for(int i = 1; i <= unit; ++i) {
        b = (b * tmp) % p;
        if(mp[b]) return i * unit - mp[b];
    }
    return -1;
}
```

## EXBSGS

$(a,p)!=1$ 时，设 $\gcd(a,p)=g$，有：

$$\frac{a}{g}\cdot a^{x-1}\equiv \frac{b}{g} \mod \frac{p}{g}$$

不断迭代，直至 $(a,\frac{p}{\prod g_i})=1$

```cpp
map<int,int> mp;
ll exgcd(ll a, ll b, ll& x, ll& y) {
    if(!b) {x = 1, y = 0;return a;}
    ll ans = exgcd(b, a % b, y, x);
    y -= a / b * x; return ans;
}
ll inv(ll a, ll p) {
    ll x, y; exgcd(a, p, x, y);
    return (x % p + p) % p;
}
ll qpow(ll a, ll b, ll p) {
    ll ans = 1; while(b) {
        if(b & 1) ans = (ans * a) % p;
        a = (a * a) % p; b >>= 1;
    } return ans;
}
ll BSGS(ll a, ll b, ll p) { 
    ll unit = (ll)ceil(sqrt(p)), tmp = qpow(a, unit, p);
    for(int i = 0; i <= unit; ++i)
        mp[b] = i, b = (b * a) % p;
    b = 1;
    for(int i = 1; i <= unit; ++i) {
        b = (b * tmp) % p;
        if(mp[b]) return i * unit - mp[b];
    }
    return -1;
}
ll exBSGS(ll a, ll b, ll p) {
    ll x, y, g = exgcd(a, p, x, y), k = 0, tmp = 1;
    while(g != 1) {
        if(b % g) return -1;
        ++k, b /= g, p /= g, tmp = tmp * (a / g) % p;
        if(tmp == b) return k;
        g = exgcd(a, p, x, y);
    }
    ll ans = BSGS(a, b * inv(tmp, p) % p, p);
    if (ans == -1) return -1;
    return ans + k;
}
```

## Lucas

质数 $p$，有 $\binom{n}{m}\bmod p = \binom{\left\lfloor n/p \right\rfloor}{\left\lfloor m/p\right\rfloor}\cdot\binom{n\bmod p}{m\bmod p}\bmod p$

```cpp
long long Lucas(long long n, long long m, long long p) {
    if (m == 0) return 1;
    return (C(n % p, m % p, p) * Lucas(n / p, m / p, p)) % p;
}
```

## 原根

模 $m$ 的原根为 $g$（$\gcd(g,m)=1$），是使得 $g^{x}\equiv 1 \mod m$ 的最小正整数解为 $x=\varphi(m)$

相当于 $g$ 是模 $m$ 剩余系所构成的最大循环群的生成元

原根存在定理：一个数 $m$ 存在原根当且仅当 $m=2,4,p^{\alpha},2p^{\alpha}$，其中 $p$ 是奇素数，$\alpha$ 是正整数

最小原根在 $O(m^{\frac{1}{4}})$ 范围，可以暴力找到一个原根

根据欧拉定理，$a^x\mod m\equiv a^{x\mod \varphi(m)}\mod m$，如果找到了一个原根 $g$，那么对于所有 $x$ 满足 $\gcd(x,\varphi(m))=1$，$g^x$ 都是原根，且这是模 $m$ 的所有原根（原根个数为 $\varphi(\varphi(m))$ 个）

```cpp
int phi(int x) {
    int ans=x;
    for (int i=2;i*i<=x;i++) {
	    if (x%i==0) ans=ans/i*(i-1)
        while (x%i==0) x/=i;
    }
    if (x>1) ans=ans/x*(x-1);
    return ans;
}
inline bool check(int x){
    if (x==2||x==4) return 1;
    if (!(x&1)) x>>=1;
    if (!(x&1)) return 0;
    for (int i=2;i*i<=x;i++){
        if (x%i==0){
            while (x%i==0) x/=i;
            return (x==1);
        }
    }
    return 1;
}
int ord(int m) {
    if (!check(m)) return -1;
    vector<int> pm;  int p,ph=p=phi(m);
    for (int i=2;i*i<=ph;i++)
        if (ph%i==0) {
            pm.pb(i);
            while (ph%i==0) ph/=i;
        }
    if (ph>1) pm.pb(ph);
    for (int g=1;g<=m;g++) {
        if (gcd(g,m)!=1) continue;
        int flag=1;
        for (auto it:pm)
            if (qpow(g,p/it,m)==1) {flag=0;break;}
        if (flag) return g;
    }
}
```

## 一些公式

$n=\sum_{d|n}\varphi(d)$

$\varphi(n) = n \times \prod_{i = 1}^s{\dfrac{p_i - 1}{p_i}}$

$\varphi(mn)\varphi(\gcd(m,n))=\varphi(m)\varphi(n)\gcd(m,n)$

$a^b\equiv
\begin{cases}
a^{b\bmod\varphi(p)},\,&\gcd(a,\,p)=1\\
a^b,&\gcd(a,\,p)\ne1,\,b<\varphi(p)\\
a^{b\bmod\varphi(p)+\varphi(p)},&\gcd(a,\,p)\ne1,\,b\ge\varphi(p)
\end{cases}
\pmod p$

素数 $p$ 有 $(p-1)!\equiv -1\pmod p$

## 线性筛

### 筛质数

```cpp
const int N=1e6+10;
int n,npr[N];
vector<int> pr;

void pre() {
	npr[1]=1;
	for (int i=2;i<=n;i++) {
		if (!npr[i]) pr.pb(i);
		for (auto p:pr) {
			if (i*p>n) break;
			if (i%p==0) {npr[i*p]=1; break;}
			else npr[i*p]=1;
		}
	}
}
```

### 筛积性函数

需要积性函数 $f$ 有转移 $f(np)=g(p,f(n))$，其中 $p$ 表示 $n$ 的最小质因子，$g$ 表示递推方程

```cpp
const int N=1e6+10;
int n,npr[N],phi[N],mu[N];
vector<int> pr;

void init() {
	npr[1]=1; phi[1]=mu[1]=1;
	for (int i=2;i<=n;i++) {
		if (!npr[i])
			pr.pb(i),phi[i]=i-1,mu[i]=-1;
			// f(x) x 为质数时的初值
		for (auto p:pr) {
			if (i*p>n) break;
			if (i%p==0) {
				npr[i*p]=1,phi[i*p]=phi[i]*p,mu[i*p]=0;
				// n 最小质数为 p 时的转移 f(np)=g(p,f(n))
				break;
			} else {
				npr[i*p]=1;
				phi[i*p]=phi[i]*phi[p];
				mu[i*p]=mu[i]*mu[p];
			}
		}
	}
}
```

需要能快速求出 $f(p^c)$ 的值

```cpp
const int N=1e6+10;
int n,npr[N],phi[N],mu[N],sp[N],exc[N];
vector<int> pr;

void init() {
    npr[1]=1;
    for (int i=2;i<=n;i++) {
        if (!npr[i]) pr.pb(i);
        for (auto p:pr) {
            if (i*p>n) break;
            if (i%p==0) {npr[i*p]=1; break;}
            else npr[i*p]=1;
        }
    }
    phi[1]=mu[1]=1;
    // 快速算出 f(p^c)
    for (auto p:pr) {
        phi[p]=p-1; mu[p]=-1;
        for (int j=p*p;j<=n;j*=p) phi[j]=phi[j/p]*p,mu[j]=0;
    }
    for (int i=2;i<=n;i++) {
        if (!npr[i]) sp[i]=i,exc[i]=1;
        for (auto p:pr) {
            if (i*p>n) break;
            if (i%p==0) {
                sp[i*p]=sp[i]*p; exc[i*p]=exc[i];
                phi[i*p]=phi[exc[i]]*phi[sp[i*p]];
                mu[i*p]=mu[exc[i]]*mu[sp[i*p]];
                break;
            }
            else {
                sp[i*p]=p; exc[i*p]=i;
                phi[i*p]=phi[i]*phi[p];
                mu[i*p]=mu[i]*mu[p];
            }
        }
    }
}
```


## 数论分块

对于常数 $n$，使得式子 $\left\lfloor\dfrac ni\right\rfloor=\left\lfloor\dfrac nj\right\rfloor$ 成立且满足 $i\leq j\leq n$ 的 $j$ 值最大为 $\left\lfloor\dfrac n{\lfloor\frac ni\rfloor}\right\rfloor$，即值 $\left\lfloor\dfrac ni\right\rfloor$ 所在块的右端点为 $\left\lfloor\dfrac n{\lfloor\frac ni\rfloor}\right\rfloor$。

对于常数 $n$，使得式子 $\left\lceil\dfrac ni\right\rceil=\left\lceil\dfrac nj\right\rceil$ 成立且满足 $i\leq j\leq n$ 的 $j$ 值最大为 $\left\lfloor\dfrac{n-1}{\lfloor\frac{n-1}i\rfloor}\right\rfloor$，即值 $\left\lceil\dfrac ni\right\rceil$ 所在块的右端点为 $\left\lfloor\dfrac{n-1}{\lfloor\frac{n-1}i\rfloor}\right\rfloor$。当 $i=n$ 时，上式会出现分母为 $0$ 的错误，需要特殊处理。

## 二次剩余

对奇素数 $p$ 和满足 $(a,p)=1$ 的整数 $a$，
 
$a^{\frac{p-1}{2}}\equiv\begin{cases}
    1 \pmod p,  & (\exists x\in\mathbf{Z}),~~a\equiv x^2\pmod p,\\
    -1 \pmod p, & \text{otherwise}.\\
\end{cases}$

即对上述的 $p$ 和 $a$，

$a$ 是 $p$ 的二次剩余当且仅当  $a^{\frac{p-1}{2}}\equiv 1 \pmod p$.

$a$ 是 $p$ 的二次非剩余当且仅当  $a^{\frac{p-1}{2}}\equiv -1 \pmod p$.

```cpp
int mod,I_mul_I; // 虚数单位的平方
struct complex {
    int real, imag;
    complex(int real = 0, int imag = 0): real(real), imag(imag) { }
};
inline bool operator == (complex x, complex y) {
    return x.real == y.real and x.imag == y.imag;
}
inline complex operator * (complex x, complex y) {
    return complex((x.real * y.real + I_mul_I * x.imag % mod * y.imag) % mod,
            (x.imag * y.real + x.real * y.imag) % mod);
}
complex power(complex x, int k) {
    complex res = 1;
    while(k) {
        if(k & 1) res = res * x;
        x = x * x;
        k >>= 1;
    }
    return res;
}
bool check_if_residue(int x) {
    return power(x, (mod - 1) >> 1) == 1;
}
void solve(int n, int p, int &x0, int &x1) {
    mod = p;
    int a = rand() % mod;
    while(!a or check_if_residue((a * a + mod - n) % mod))
        a = rand() % mod;
    I_mul_I = (a * a + mod - n) % mod;
    x0 = int(power(complex(a, 1), (mod + 1) >> 1).real);
    x1 = mod - x0;
}
```


## Miller-Rabin 素性测验

``unsigned int`` 范围内仅需判定 $2,7,61$；

``unsigned long long`` 范围内仅需判定 $2,325,9375,28178,450775,9780504,1795265022$

```cpp
int qpow(int x,int k,int mod) {
	int ans=1; while (k) {
		if (k&1) ans=(__int128)ans*x%mod;
		k>>=1; x=(__int128)x*x%mod;
	} return ans;
}

const int mr[3]={2,7,61};
bool Miller_Rabin(int p) {
	if (p<2) return 0;
	if (p==2||p==3) return 1;
	int d=p-1,r=0;
	while (!(d&1)) ++r,d>>=1;
	for (int k=0;k<3;k++) {
		int u=mr[k]%p;
		if (u==0||u==1||u==p-1) continue;
		int x=qpow(u,d,p);
		if (x==1||x==p-1) continue;
		for (int i=0;i<r-1;++i) {
			x=(__int128)x*x%p;
			if (x==p-1) break;
		}
		if (x!=p-1) return 0;
	}
	return 1;
}
```

## Pollard-Rho 质因数分解

```cpp
int qpow(int x,int k,int mod) {
	int ans=1; while (k) {
		if (k&1) ans=(__int128)ans*x%mod;
		k>>=1; x=(__int128)x*x%mod;
	} return ans;
}

const int mr[7]={2,325,9375,28178,450775,9780504,1795265022};
bool Miller_Rabin(int p) {
	if (p<2) return 0;
	if (p==2||p==3) return 1;
	int d=p-1,r=0;
	while (!(d&1)) ++r,d>>=1;
	for (int k=0;k<7;k++) {
		int u=mr[k]%p;
		if (u==0||u==1||u==p-1) continue;
		int x=qpow(u,d,p);
		if (x==1||x==p-1) continue;
		for (int i=0;i<r-1;++i) {
			x=(__int128)x*x%p;
			if (x==p-1) break;
		}
		if (x!=p-1) return 0;
	}
	return 1;
}

int Pollard_Rho(int x) {
	int s=0,t=0,step=0,goal=1,val=1;
	int c=(int)rand()%(x-1)+1;
	for (goal=1;;goal*=2,s=t,val=1) {
		for (step=1;step<=goal;++step) {
			t=((__int128)t*t%x+c)%x;
			val=(__int128)val*abs(t-s)%x;
			if ((step%127)==0) {
				int d=__gcd(val,x);
				if (d>1) return d;
			}
		}
		int d=__gcd(val,x);
		if (d>1) return d;
	}
} 

vector<int> pfac;

void fac(int x,int t) {
	if (x<2) return ;
	if (Miller_Rabin(x)) {
		while (t--) pfac.push_back(x);
		return ;
	}
	int p=x,t2=0;
	while (p>=x) p=Pollard_Rho(x);
	while ((x%p)==0) x/=p,++t2;
	fac(x,t),fac(p,t*t2);
}

void get(int x) {
	fac(x,1);
	sort(pfac.begin(),pfac.end());
	for (auto p:pfac) cout<<p<<' ';
}
```

