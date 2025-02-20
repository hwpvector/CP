# 式子与筛法

## 常见积性函数

- 单位函数：$\varepsilon(n)=[n=1]$（完全积性）

- 恒等函数：$\operatorname{id}_k(n)=n^k，\operatorname{id}_{1}(n)$ 通常简记作 $\operatorname{id}(n)$（完全积性）

- 常数函数：$1(n)=1$（完全积性）

- 除数函数：$\sigma_{k}(n)=\sum_{d\mid n}d^{k}。\sigma_{0}(n)$ 通常简记作 $\tau(n)$，$\sigma_{1}(n)$ 通常简记作 $\sigma(n)$

- 欧拉函数：$\varphi(n)=\sum_{i=1}^n[(i,n)=1]$

- 莫比乌斯函数：

$\mu(n)=\begin{cases}1&n=1\\0&\exists d>1,d^{2}\mid n\\(-1)^{\omega(n)}&\text{otherwise}\end{cases}$，其中 $\omega(n)$ 表示 $n$ 的本质不同质因子个数

| $n\le$ | $10^1$ | $10^2$ | $10^3$ | $10^4$ | $10^5$ | $10^6$ | $10^7$ | $10^8$ | $10^9$
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\max\{\omega(n)\}$| $2$ | $3$ | $4$ | $5$ | $6$ | $7$ | $8$ | $8$ | $9$ |
| $\max\{d(n)\}$ | $4$ | $12$ | $32$ | $64$ | $128$ | $240$ | $448$ | $768$ | $1344$ |

| $n\le$ | $10^{10}$ | $10^{11}$ | $10^{12}$ | $10^{13}$ | $10^{14}$ | $10^{15}$ | $10^{16}$ | $10^{17}$ | $10^{18}$
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\max\{\omega(n)\}$| $10$ | $10$ | $11$ | $12$ | $12$ | $13$ | $13$ | $14$ | $15$ |
| $\max\{d(n)\}$ | $2304$ | $4032$ | $6720$ | $10752$ | $17280$ | $26880$ | $41472$ | $64512$ | $103680$ |


## 狄利克雷卷积

$h(x)=\sum_{d|x}f(d)g(\frac{x}{d})$，记 $h=f*g$

一些性质：

- 交换律：$f*g=g*f$
- 结合律：$(f*g)*h=f*(g*h)$
- 分配律：$(f+g)*h=f*h+g*h$
- $f=g \iff f*h=g*h, h(1)\ne 0$
- 单位元：$\varepsilon$，$\varepsilon*f=f*\varepsilon=f$，$\varepsilon(n)=[n=1]$
- 逆元：$f*g=\varepsilon$；逆元唯一，可以构造出：

$$
g(x)=\dfrac {\varepsilon(x)-\sum_{d\mid x,d\ne 1}{f(d)g\left(\dfrac {x}{d} \right)}}{f(1)}
$$

- 两个积性函数的狄利克雷卷积也是积性函数
- 积性函数的逆元也是积性函数

## 莫比乌斯反演
 
$$
f(n)=\sum_{d\mid n}g(d) \iff g(n)=\sum_{d\mid n}\mu(d)f(\frac{n}{d})
$$

$$
f(n)=\sum_{n\mid d}g(d) \iff g(n)=\sum_{n\mid d}\mu(\frac{d}{n})f(d)
$$
 
$$
f(n)=\sum_{i=1}^nt(i)g\left(\left\lfloor\frac{n}{i}\right\rfloor\right) \iff g(n)=\sum_{i=1}^n\mu(i)t(i)f\left(\left\lfloor\frac{n}{i}\right\rfloor\right)
$$

一些常见的反演 trick：

$$
n=\sum_{d\mid n}\varphi(d) \iff \varphi(n)=\sum_{d\mid n}\mu(d)\frac{n}{d}
$$

$$
\begin{aligned}
f(\gcd(a_1,a_2,\cdots))&=\sum_k\sum_{a_i} f(k)[\gcd(a_1,a_2,\cdots)=k]\\
&=\sum_{k}\sum_{a_i,k|a_i,a_i'=\frac{a_i}{k}} f(k)[\gcd(a_1',a_2',\cdots)=1]\\
&=\sum_{k}\sum_{a_i,k|a_i,a_i'=\frac{a_i}{k}}  f(k) \sum_{d|a_i'}\mu(d)\\
&=\sum_{k}\sum_d\sum_{a_i,dk|a_i,a_i'=\frac{a_i}{dk}}  f(k)\mu(d)
\end{aligned}
$$

$$
\begin{aligned}
f(\gcd(a,b))&=\sum_k f(k)[\gcd(a,b)=k]\\
&=\sum_{d|a,b}f(d)\sum_{k|\frac{a}{d},\frac{b}{d}} \mu(k)\\
&=\sum_{dk|a,b} f(d)\mu(k)\\
&=\sum_{d|a,b} \sum_{k|d} f(\frac{d}{k}) \mu(k)\\
\end{aligned}
$$

$$
d(ab)=\sum_{d_1|a}\sum_{d_2|b}[\gcd(a,b)=1]
$$

$$
\sum_{d|n,k|d}f(k)g(\frac{d}{k})=\sum_{d|n,k|d}f(k)g(\frac{n}{d})
$$

$$
    \prod_i\prod_jf(\gcd(i,j))=\prod_i\prod_jf(d)^{[\gcd(i,j)=d]}
$$

$$
    \prod_i\prod_j f(i)^{g(i,j)}=\prod_i f(i)^{\sum_j g(i,j)}
$$

### Gym 105423 D

> 题意：给两个序列 $\{a_n\},\{b_n\}$，求 $\sum_{i=1}^n\sum_{j=1}^n|a_i-a_j|\frac{\text{lcm}(b_ib_j)}{\gcd(b_i,b_j)}$。$1\le a_i,b_i,n\le 10^5$

$$
\begin{aligned}
    \frac{1}{\gcd(a,b)^2}&=\sum_{d|a,b}\frac{1}{d^2}\sum_{k|\frac{a}{d},\frac{b}{d}}\mu(k)\\
    &=\sum_{d\cdot k|a,b}\frac{\mu(k)}{d^2}\\
    &=\sum_{d|a,b}\sum_{k|d}\frac{\mu(k)}{(\frac{d}{k})^2}\\
    &=\sum_{d|a,b}\frac{1}{d^2}\sum_{k|d}\mu(k)k^2
\end{aligned}
$$

记 $f(d)=\sum_{k|d}\mu(k)k^2$，$f(d)$ 可以通过线性筛求出

对 $a_i$ 排序，钦定 $i,j\rightarrow a_i>a_j$

$$
\begin{aligned}
    \sum_i\sum_j|a_i-a_j|\frac{b_ib_j}{\gcd(b_i,b_j)^2}&=2\sum_i\sum_{j<i}(a_i-a_j)b_ib_j\sum_{d|b_i,b_j}\frac{1}{d^2}f(d)\\
    &=2\sum_i\sum_{j<i}\sum_{d}[d|b_i][d|b_j]\frac{(a_i-a_j)b_ib_j}{d^2}f(d)\\
    &=2\sum_i\sum_d[d|b_i]\frac{b_if(d)}{d^2}(a_i\sum_{j<i}[d|b_j]b_j-\sum_{j<i}[d|b_j]a_jb_j)
\end{aligned}
$$

考虑枚举 $i$ 和 $d|b_i$，并对每个 $d$ 动态维护前缀和 $s_{d,i}=\sum_{j<i}[d|b_j]b_j$，$s'_{d,i}=\sum_{j<i}[d|b_j]a_jb_j$

### Luogu P1829

$$
\begin{aligned}
    \sum_i\sum_j\frac{ij}{\gcd(i,j)}&=\sum_i\sum_j ij\sum_d\frac{1}{d}\sum_{k|\frac{i}{d},\frac{j}{d}}\mu(k)\\
    &=\sum_i\sum_j ij \sum_{dk|i,j}\frac{1}{d}\mu(k)\\
    &=\sum_i\sum_j ij \sum_{d|i,j}\sum_{k|d}\frac{\mu(k)}{\frac{d}{k}}\\
    &=\sum_i\sum_j ij \sum_{d|i,j}\frac{1}{d} \sum_{k|d} k\mu(k)\\
    &=\sum_d\frac{1}{d} f(d)\sum_i[d|i]i\sum_j[d|j]j\\
    &=\sum_d\frac{1}{d} f(d) (d\cdot g(\lfloor\frac{n}{d}\rfloor))(d\cdot g(\lfloor\frac{m}{d}\rfloor))\\
    &=\sum_d df(d)g(\lfloor\frac{n}{d}\rfloor) g(\lfloor\frac{m}{d}\rfloor)
\end{aligned}
$$

其中，$f(d)=\sum_{k|d} k\mu(k), g(n)=\frac{n(n+1)}{2}$

显然 $f(d)$ 可以线性筛筛出，枚举 $d$ 算贡献即可

### Luogu P5518

#### 式子 1

$$
\begin{aligned}
    \prod_{i=1}^A\prod_{j=1}^B\prod_{k=1}^C \frac{ij}{\gcd(i,j)\gcd(i,k)}
\end{aligned}
$$

分子：

$$
\begin{aligned}
    \prod_{i=1}^A\prod_{j=1}^B\prod_{k=1}^C i=(A!)^{BC}
\end{aligned}
$$

分母：

$$
\begin{aligned}
    \prod_{i=1}^A\prod_{j=1}^B\prod_{k=1}^C \frac{1}{\gcd(i,j)}&=(\prod_{i=1}^A\prod_{j=1}^B \frac{1}{\gcd(i,j)})^C\\
    &=(\prod_{i=1}^A\prod_{j=1}^B\prod_d (\frac{1}{d})^{[\gcd(i,j)=d]})^C\\
    &=(\prod_d(\frac{1}{d})^{\sum_{i=1}^A\sum_{j=1}^B[\gcd(i,j)=d]})^C\\
    &=(\prod_d\prod_{k}(\frac{1}{d})^{\mu(k)\lfloor\frac{A}{dk}\rfloor\lfloor\frac{B}{dk}\rfloor})^C\\
    &=(\prod_d\prod_{k|d}(\frac{k}{d})^{\mu(k)\lfloor\frac{A}{d}\rfloor\lfloor\frac{B}{d}\rfloor})^C
\end{aligned}
$$

注意到 $f(d)=\prod_{k|d}(\frac{k}{d})^{\mu(k)}$ 可以预处理，计算每个 $k$ 对 $f(1),f(2),\cdots$ 的贡献

于是，式子变为：

$$
\prod_df(d)^{\lfloor\frac{A}{d}\rfloor\lfloor\frac{B}{d}\rfloor}
$$

对 $\lfloor\frac{A}{d}\rfloor$ 和 $\lfloor\frac{B}{d}\rfloor$ 分块即可

复杂度：$O(n\log n+T\sqrt{n}\log n)$

#### 式子 2

处理方法和式子 1 类似

$$
\begin{aligned}
    \prod_{i=1}^A\prod_{j=1}^B\prod_{k=1}^C (\frac{ij}{\gcd(i,j)\gcd(i,k)})^{ijk}
\end{aligned}
$$

分子：

$$
\begin{aligned}
    \prod_{i=1}^A\prod_{j=1}^B\prod_{k=1}^C i^{ijk}&=\prod_{i=1}^A i^{i\sum_{j=1}^Bj\sum_{k=1}^Ck}\\
    &=(\prod_{i=1}^A i^i)^{\frac{B(B+1)}{2}\frac{C(C+1)}{2}}
\end{aligned}
$$

$\prod_{i=1}^A i^i$ 可以 $O(n\log n)$ 预处理，每次询问 $O(\log n)$

分母：

$$
\begin{aligned}
    \prod_{i=1}^A\prod_{j=1}^B\prod_{k=1}^C (\frac{1}{\gcd(i,j)})^{ijk}&=(\prod_{d} \prod_{k|d} (\frac{k}{d})^{d^2\mu(k)S(\lfloor\frac{A}{d}\rfloor)S(\lfloor\frac{B}{d}\rfloor)})^{\frac{C(C+1)}{2}}
\end{aligned}
$$

其中 $S(n)=\frac{n(n+1)}{2}$

$g(d)=\prod_{k|d} (\frac{k}{d})^{d^2\mu(k)}$ 同样预处理，对 $\lfloor\frac{A}{d}\rfloor$ 和 $\lfloor\frac{B}{d}\rfloor$ 同样分块即可

复杂度：$O(n\log n+T\sqrt{n}\log n)$

 1. 这里是列表文本

#### 式子 3

最丑陋的一部分。推的时间最长的一部分。大部分时间浪费在分母注意力不足上。

$$
\begin{aligned}
    \prod_{i=1}^A\prod_{j=1}^B\prod_{k=1}^C (\frac{ij}{\gcd(i,j)\gcd(i,k)})^{\gcd(i,j,k)}
\end{aligned}
$$

分子：

$$
\begin{aligned}
    \prod_{i=1}^A\prod_{j=1}^B\prod_{k=1}^C i^{\gcd(i,j,k)}&=\prod_{i=1}^A i^{\sum_{j=1}^B\sum_{k=1}^C\gcd(i,j,k)}\\
    &=\prod_{i=1}^A i^{\sum_{j=1}^B\sum_{k=1}^C\sum_{d|i}d[\gcd(i,j,k)=d]}\\
    &=\prod_{d}\prod_{i=1}^{\lfloor\frac{A}{d}\rfloor} (id)^{d\sum_{j=1}^{\lfloor\frac{B}{d}\rfloor}\sum_{k=1}^{\lfloor\frac{C}{d}\rfloor}[\gcd(i,j,k)=1]}\\
    &=\prod_{d}\prod_{t}\prod_{i=1}^{\lfloor\frac{A}{td}\rfloor} (id)^{d\lfloor\frac{B}{td}\rfloor\lfloor\frac{C}{td}\rfloor\mu(t)}\\
    &=\prod_{d}\prod_{t|d}\prod_{i=1}^{\lfloor\frac{A}{d}\rfloor} (id)^{\frac{d}{t}\mu(t)\lfloor\frac{B}{d}\rfloor\lfloor\frac{C}{d}\rfloor}\\
    &=\prod_{d} (\lfloor\frac{A}{d}\rfloor! \cdot d^{\lfloor\frac{A}{d}\rfloor})^{\varphi(d)\lfloor\frac{B}{d}\rfloor\lfloor\frac{C}{d}\rfloor}\\
    &=(\prod_{d} (\lfloor\frac{A}{d}\rfloor!)^{\varphi(d)})^{\lfloor\frac{B}{d}\rfloor\lfloor\frac{C}{d}\rfloor} \cdot (\prod_d{d^{\varphi(d)})^{\lfloor\frac{A}{d}\rfloor\lfloor\frac{B}{d}\rfloor\lfloor\frac{C}{d}\rfloor}}
\end{aligned}
$$

于是，我们可以预处理 $\varphi(d)$ 的前缀和（$\mod (p-1)$ 意义下的），$n!$ 的值，$h(d)=d^{\varphi(d)}$ 的前缀积，然后对 $\lfloor\frac{A}{d}\rfloor, \lfloor\frac{B}{d}\rfloor, \lfloor\frac{C}{d}\rfloor$ 整除分块即可

复杂度：$O(n\log n+T\sqrt{n}\log n)$

分母：

$$
\begin{aligned}
    \prod_{i=1}^A\prod_{j=1}^B\prod_{k=1}^C (\frac{1}{\gcd(i,j)})^{\gcd(i,j,k)}&=\prod_{i=1}^A\prod_{j=1}^B\prod_{k=1}^C\prod_{p} (\frac{1}{p})^{\gcd(p,k)[\gcd(i,j)=p]}\\
    &=\prod_{p}\prod_{i=1}^{\lfloor\frac{A}{p}\rfloor}\prod_{j=1}^{\lfloor\frac{B}{p}\rfloor}\prod_{k=1}^{C} (\frac{1}{p})^{\gcd(p,k)[\gcd(i,j)=1]}\\
    &=\prod_{p}\prod_{k=1}^C (\frac{1}{p})^{\gcd(p,k)\sum_{t}\mu(t)\lfloor\frac{A}{pt}\rfloor\lfloor\frac{B}{pt}\rfloor}\\
    &=\prod_{p} (\frac{1}{p})^{\sum_{d|p}\lfloor\frac{C}{d}\rfloor\varphi(d)\sum_{t}\mu(t)\lfloor\frac{A}{pt}\rfloor\lfloor\frac{B}{pt}\rfloor}
\end{aligned}
$$

丑陋无比。简要写一下思考过程。也是走了很多的弯路。发现枚举的项很丑陋，先拆开：

$$
\prod_d \prod_p \prod_t (\frac{1}{dp})^{\lfloor \frac{C}{d}\rfloor \varphi(d)\mu(t)\lfloor\frac{A}{tdp}\rfloor\lfloor\frac{B}{tdp}\rfloor}
$$

我们想着，这几个向下取整总得整除分块做，于是调整一下 $tp$ 的枚举：

$$
\prod_d \prod_p \prod_{t|p} (\frac{t}{dp})^{\lfloor \frac{C}{d}\rfloor \varphi(d)\mu(t)\lfloor\frac{A}{dp}\rfloor\lfloor\frac{B}{dp}\rfloor}
$$

顺着分块的思路去想，搞一搞，我们发现 $\frac{t}{dp}$ 仅有 $t$ 是好分块的。于是，我们将其拆成两部分算贡献：$t$ 和 $\frac{1}{dp}$

第一部分：

$$
\prod_d \prod_p \prod_{t|p} t^{\lfloor \frac{C}{d}\rfloor \varphi(d)\mu(t)\lfloor\frac{A}{dp}\rfloor\lfloor\frac{B}{dp}\rfloor}=\prod_p (\prod_{t|p} t^{\mu(t)})^{ \sum_d\varphi(d)\lfloor \frac{C}{d}\rfloor\lfloor\frac{A}{dp}\rfloor\lfloor\frac{B}{dp}\rfloor}
$$

预处理一下 $f'(p)=\prod_{t|p} t^{\mu(t)}$，及 $f'(p)$ 的前缀积，$\varphi(p)$ 的前缀和。先对 $\lfloor\frac{A}{d}\rfloor,\lfloor\frac{B}{d}\rfloor,\lfloor\frac{C}{d}\rfloor$ 整除分块，再对 $\lfloor\frac{\lfloor\frac{A}{d}\rfloor}{p}\rfloor$ 分块，即可。

复杂度为 $O(n\log n+Tn^{\frac{3}{4}}\log n)$

关于分块套分块复杂度的证明：

分块套分块形如：$\sum_{i,j}f(\lfloor\frac{n}{i}\rfloor,\lfloor\frac{\lfloor\frac{n}{i}\rfloor}{j}\rfloor)$，其中 $f(i,j)$ 能 $O(1)$ 算出。 

观察到 $i>\sqrt{n}$ 的部分枚举了 $\sqrt{n}$ 次，但带来的贡献比 $i<\sqrt{n}$ 少。于是，

$$
\begin{aligned}
T(n)&\le 2((\frac{n}{1})^{1/2}+(\frac{n}{2})^{1/2}+\cdots+(\frac{n}{\sqrt{n}})^{1/2})\\
&\le \int_{0}^{\sqrt{n}} (\frac{n}{x})^{1/2} dx=O(n^{\frac{3}{4}})
\end{aligned}
$$

容易证明的，分块套 $m$ 层，复杂度为 $O(n^{1-{2^{-m}}})$

第二部分（注意到 $\sum_{t|p}\mu(t)=[p=1]$）：

$$
\prod_d \prod_p \prod_{t|p} (\frac{1}{dp})^{\lfloor \frac{C}{d}\rfloor \varphi(d)\mu(t)\lfloor\frac{A}{dp}\rfloor\lfloor\frac{B}{dp}\rfloor}=\prod_{d} (\frac{1}{d})^{\varphi(d)\lfloor\frac{A}{d}\rfloor\lfloor\frac{B}{d}\rfloor\lfloor\frac{C}{d}\rfloor}
$$

预处理一下 $(\frac{1}{d})^{\varphi(d)}$ 的前缀积，然后对 $\lfloor\frac{A}{d}\rfloor,\lfloor\frac{B}{d}\rfloor,\lfloor\frac{C}{d}\rfloor$ 整除分块

## 狄利克雷和

已知 $f$，可以在 $O(n\log\log n)$ 求满足下述条件的 $g$：

$$g=\sum_{d|n}f(d) \text{ or } f=\sum_{d|n}g(d)$$

对于第一个式子，令 $n=\prod_i p_i^{k_i}$，则有 $g(n)=\sum_{k_i'\le k_i} f(\prod_i p_i^{k_i'})$，即对所有的 $p_i$ 做高维前缀和

```cpp
for (auto p:pr)
    for (int j=1;j<=n/p;++j)
        f[j*p]+=f[j];
```

对于第二个式子（通常更常见的是其的反演 $g=\sum_{d|n} f(d)*\mu(\frac{n}{d})$），对所有的 $p_i$ 做高维差分

```cpp
for (auto p:pr)
    for (int j=n/p;j>=1;--j)
        f[j*p]-=f[j];
```

同理，可以处理：

$$g=\sum_{n|d}f(d) \text{ or } f=\sum_{n|d}g(d)$$

```cpp
for (auto p:pr)
    for (int j=n/p;j>=1;--j)
        f[j]+=f[j*p];
for (auto p:pr)
    for (int j=1;j<=n/p;++j)
        f[j]-=f[j*p];
```

## 杜教筛

求和函数为 $f$，需要满足：

- 能构造一个函数 $g$，易求 $f*g$ 在 $\lfloor \frac{n}{i}\rfloor$ 处的前缀和
- 易求 $g$ 在 $\lfloor \frac{n}{i}\rfloor$ 处的前缀和

对于数论函数 $f$，记 $S(n)=\sum_{i=1}^nf(n)$

对于数论函数 $g$，有：

$$
\begin{aligned}
\sum_{i=1}^n(f*g)(i)&=\sum_{i=1}^n\sum_{d|i}g(d)f(\frac{i}{d})\\
&=\sum_{d=1}^n\sum_{1\le i\le n,d|i} g(d)f(\frac{i}{d})\\
&=\sum_{d=1}^n\sum_{i=1}^{\lfloor\frac{n}{d}\rfloor}g(d)f(i)\\
&=\sum_{d=1}^ng(d)\sum_{i=1}^{\lfloor\frac{n}{d}\rfloor}f(i)\\
&=\sum_{i=1}^ng(i)S(\lfloor\frac{n}{i}\rfloor)
\end{aligned}
$$

于是，我们可以得到 $S(n)$ 向 $S(\lfloor\frac{n}{i}\rfloor)$ 的递推式：

$$
\begin{aligned}
g(1)S(n)&=\sum_{i=1}^ng(i)S(\lfloor\frac{n}{i}\rfloor)-\sum_{i=2}^ng(i)S(\lfloor\frac{n}{i}\rfloor)\\
&=\sum_{i=1}^n(f*g)(i)-\sum_{i=2}^ng(i)S(\lfloor\frac{n}{i}\rfloor)
\end{aligned}
$$

如果我们能够构造 $g$，使得能在 $O(1)$ 的时间内算出 $f*g$ 和 $g$ 的前缀和，则该递推式也能够快速计算

不做证明的，时间复杂度为 $T(n)=O(n^{\frac{3}{4}})$

若能以 $T_0(m)$ 的复杂度预处理 $f$ 的 $m$ 项前缀和，则 $T(n)=O(T_0(m)+\frac{n}{\sqrt{m}})$

当 $T_0(m)=O(m)$，即线性筛预处理前 $m$ 项前缀和的话，时间复杂度为 $O(n^{\frac{2}{3}})$，此时 $m=n^\frac{2}{3}$

```cpp
// n ~ 2e9
const int N=1e5+10,M=3e6+10;
int n,sq;
int sp[N],sm[N];
int phi[M],mu[M],npr[M],sphi[M],smu[M];
vector<int> pr;

void init() {
	npr[1]=1; phi[1]=mu[1]=1;
	for (int i=2;i<=3e6;i++) {
		if (!npr[i]) {
			pr.pb(i);
			phi[i]=i-1;
			mu[i]=-1;
		} 
		for (auto p:pr) {
			if (i*p>3e6) break;
			if (i%p==0) {
				phi[i*p]=phi[i]*p;
				mu[i*p]=0;
				npr[i*p]=1;
				break;
			} else {
				phi[i*p]=phi[i]*phi[p];
				mu[i*p]=mu[i]*mu[p];
				npr[i*p]=1;
			}
		}
	}
	for (int i=1;i<=3e6;i++) {
		sphi[i]=sphi[i-1]+phi[i];
		smu[i]=smu[i-1]+mu[i];
	}
}
int getid(int x) {
	if (x<=sq) return x;
	else return sq+n/x;
}

void solve() {
	cin>>n; sq=sqrt(n);
	for (int i=1;i<=n;) {
		int m=n/(n/i);
		if (m<=3e6) {
			sp[getid(m)]=sphi[m];
			sm[getid(m)]=smu[m];
		} else {
			sp[getid(m)]=m*(m+1)/2;
			sm[getid(m)]=1;
			for (int l=2;l<=m;) {
				int r=m/(m/l);
				sp[getid(m)]-=(r-l+1)*sp[getid(m/l)];
				sm[getid(m)]-=(r-l+1)*sm[getid(m/l)];
				l=r+1;
			}
		} 
		i=m+1;
	}
	cout<<sp[getid(n)]<<' '<<sm[getid(n)]<<'\n';
}
```

~~常数为什么这么大~~。多测应该用记搜会快一点（用 ``unodered_map`` 存），或者预处理多一点

### Luogu P3768

$$
\begin{aligned}
\sum_i\sum_j ij\gcd(i,j) &=\sum_i\sum_j ij\sum_{d|i,j} \sum_{k|d} \frac{d}{k} \mu(k)\\
&=\sum_d\sum_{1\le i,j\le n, d|i,j}ij\varphi(d)\\
&=\sum_d(g(\lfloor\frac{n}{d}\rfloor))^2d^2\varphi(d)
\end{aligned}
$$

其中，$g(n)=\frac{n(n+1)}{2}$；$g(\lfloor\frac{n}{d}\rfloor)$ 可以数论分块来做。

对于 $S(n)=\sum_{d=1}^n d^2\varphi(d)$，考虑杜教筛。构造 $g(n)=n^2$，$g$ 的前缀和易算，并且：

$$
\begin{aligned}
\sum_{i=1}^n(f*g)(i)&=\sum_{i=1}^n \sum_{d|i}f(d)g(\frac{i}{d})\\
&=\sum_{i=1}^n\sum_{d|i}i^2\varphi(d)\\
&=\sum_{i=1}^n i^3=\frac{n^2(n+1)^2}{4}
\end{aligned}
$$

所以有：

$$
\begin{aligned}
S(n)&=\sum_{i=1}^n(f*g)(i)-\sum_{i=2}^ng(i)S(\lfloor\frac{n}{i}\rfloor)\\
&=\frac{n^2(n+1)^2}{4}-\sum_{i=2}^n i^2 S(\lfloor\frac{n}{i}\rfloor)
\end{aligned}
$$

我们需要做 $\sqrt{n}$ 次杜教筛。但是，注意到 $g(\lfloor\frac{n}{d}\rfloor)$ 进行的数论分块，$d$ 所需枚举区间的 $[l,r]$ 的右值 $r$ 在集合 $\{\lfloor\frac{n}{1}\rfloor,\lfloor\frac{n}{2}\rfloor,\lfloor\frac{n}{3}\rfloor,\cdots\}$ 中，而在做 $S(n)$ 的杜教筛中恰好能筛出这些点的值。也就是说，我们只做了一遍杜教筛就得到了这些结果，故复杂度为 $O(n^{\frac{2}{3}})$。

结论：数论分块套杜教筛的复杂度为 $O(n^{\frac{2}{3}})$


## PN 筛

求和函数为 $f$，需要满足：

- $f$ 为积性函数
- 能构造一个积性函数 $g$，使得当 $p$ 是质数时，$f(p)=g(p)$
- 易求 $g$ 在 $\lfloor \frac{n}{i}\rfloor$ 处的前缀和

PN 数：$n=\prod_i p_i^{k_i}$，对于任意的 $i$，有 $k_i>1$ 的数，称为 PN 数

$n$ 以内的 PN 数个数为 $O(\sqrt{n})$

求出所有 $\le n$ 的 PN 数，可以先筛出 $\le \sqrt{n}$ 的质数，然后暴力 DFS 各素数的指数

记积性函数 $f$ 的前缀和  $F(n) = \sum_{i=1}^{n} f(i)$。

构造一个积性函数 $g$，满足对于素数 $p$，有 $g(p)=f(p)$。记 $G(n)=\sum_{i=1}^n g(i)$

构造函数 $h$，使得 $f=g*h$，显然 $h$ 也是积性函数。可以发现：

- $h(1)=1$

- $p$ 是素数，则 $h(p)=0$

- 对于所有非 PN 数 $x\not = 1$，$h(x)=0$

于是：

$$
\begin{aligned}
F(n) &= \sum_{i = 1}^{n} f(i)\\
     &= \sum_{i = 1}^{n} \sum_{d|i} h(d) g\left(\frac{i}{d}\right)\\
     &= \sum_{d=1}^{n} \sum_{i=1}^{\lfloor \frac{n}{d}\rfloor} h(d) g(i)\\
     &= \sum_{\substack{d=1 \\ d \text{ is PN}}}^{n}h(d) G\left(\left\lfloor \frac{n}{d}\right\rfloor\right)
\end{aligned}
$$

我们可以找出所有的 PN 数，过程中计算出所有 $h$ 的有效值。

关于计算 $h(p^c)$，可以：

- 推出 $h(p^c)$ 仅与 $p,c$ 有关的表达式
- 推出 $h(p^{c-1})$ 转移到 $h(p^c)$ 的转移方程
- 根据狄利克雷，可推出 $h(p^c) = f(p^c) - \sum_{i=1}^{c}g(p^i)h(p^{c-i})$，于是可以枚举 $p$，递推出所有的 $h(p^c)$；复杂度为 $O(\sqrt{n}\log n)$

注意到，$G$ 可以通过杜教筛筛出，分块套杜教筛的复杂度也是 $O(n^{\frac{2}{3}})$


```cpp
const int mod=1e9+7;
inline int qpow(int x,int k) {
	int ans=1; while (k) {
		if (k&1) ans=(ans*x)%mod;
		k>>=1; x=(x*x)%mod;
	} return ans;
}
inline int inv(int x) {return qpow(x,mod-2);}

// n ~ 1e10
// f(p) = p^k * (p^k - 1)
// g(x) = x * phi(x)

const int N=2e5+10,M=7e6+10,iv6=inv(6);
int n,sq;
int sg[N];
vector<int> h[N],g[N];
int phi[M],npr[M],sp[M];
vector<int> pr;

void init() {
	npr[1]=1; phi[1]=1;
	for (int i=2;i<=7e6;i++) {
		if (!npr[i]) {
			pr.pb(i);
			phi[i]=i-1;
		} 
		for (auto p:pr) {
			if (i*p>7e6) break;
			if (i%p==0) {
				phi[i*p]=phi[i]*p%mod;
				npr[i*p]=1;
				break;
			} else {
				phi[i*p]=phi[i]*phi[p]%mod;
				npr[i*p]=1;
			}
		}
	}
	for (int i=1;i<=7e6;i++) {
		sp[i]=(sp[i-1]+phi[i]*i)%mod;
	}
	for (int i=0;i<pr.size();i++) {
		int p=pr[i];
		if (p>1e5) break;
		h[i].pb(1); h[i].pb(0);
		g[i].pb(1); g[i].pb((p-1)*p%mod);
		for (int j=p*p,c=2;j<=1e10;j*=p,++c) {
			h[i].pb(j%mod*(j%mod-1)%mod);
			g[i].pb(g[i].back()*p%mod*p%mod);
			for (int k=1;k<=c;k++) {
				h[i].back()-=g[i][k]*h[i][c-k]%mod;
			}
			h[i].back()%=mod;
		}
	}
}
int getid(int x) {
	if (x<=sq) return x;
	else return sq+n/x;
}
int sum2(int n) {
	n%=mod; return n*(n+1)%mod*(2*n+1)%mod*iv6%mod;
}
int sum(int n) {
	n%=mod; return n*(n+1)/2%mod;
}

int ans=0;
void pn(int k,int v,int sh) {
	int p=pr[k];
	if (v*p>n||v*p*p>n) return ;
	// corner case: v ~ 1e10, p ~ 1e5, ll is not enough
	pn(k+1,v,sh);
	for (int i=p*p,j=2;v*i<=n;i*=p,++j) {
		ans+=sh*h[k][j]%mod*sg[getid(n/(v*i))]; ans%=mod;
		pn(k+1,v*i,sh*h[k][j]%mod);
	}
}

void solve() {
	cin>>n; sq=sqrt(n);

	for (int i=1;i<=n;) {
		int m=n/(n/i);
		if (m<=7e6) {
			sg[getid(m)]=sp[m];
		} else {
			sg[getid(m)]=sum2(m);
			for (int l=2;l<=m;) {
				int r=m/(m/l);
				sg[getid(m)]-=(sum(r)-sum(l-1))*sg[getid(m/l)];
				sg[getid(m)]%=mod;
				l=r+1;
			}
		} 
		i=m+1;
	}
	ans=sg[getid(n)]; pn(0,1,1);
	cout<<(ans+mod)%mod<<'\n';
}
```



## Min25 筛

求和函数为 $f$，需要满足：

- $f$ 为积性函数
- 当 $p$ 是质数时，$f(p)$ 通常为低阶多项式（或者可以拆成若干个容易求和的完全积性函数的和）
- $f(p^k)(k>1)$ 可以快速求值

下文中，记 $P$ 为质数集，记第 $i$ 个质数为 $p_i$。特别的，$p_0=1$

我们先考虑筛出 $p(n)=\sum_{i=1}^n[i \in P]f(i)$，且 $p(x)$ 中 $x$ 的取值为 $\lfloor \frac{n}{i}\rfloor$

将积性函数 $f$ 拆成若干个容易求和的完全积性函数的和，分开计算，最终求和即可。我们记其中一个积性函数为 $f'(x)$

定义状态 $g(n,j)=\sum_{i=1}^n [i \in P \text{ or } \min_{p|i} p>p_j]f'(i)$

容易看出，$p(n)=g(n,m)$，其中 $m$ 表示小于 $\sqrt{n}$ 的最大质数是第 $m$ 个质数

有转移：

$$
g(n,j)=
\left\{\begin{array}{lc}
    g(n,j-1) &p_j^2>n\\
    g(n,j-1)-f'(p_j)(g(\lfloor\frac{n}{p_j}\rfloor,j-1)-\sum_{i=1}^{j-1}f'(p_j)) &p_j^2\le n
\end{array}\right.
$$

边界：

$$
g(n,0)=\sum_{i=2}^nf'(i)
$$

该转移的复杂度为 $O(\frac{n^{\frac{3}{4}}}{\log n})$

定义求和状态 $S(n,j)=\sum_{i=1}^nf(i)(\min_{p|i}p>p_j)$

最终所求的答案就是 $S(n,0)+f(1)$

有转移：

$$
S(n,j)=p(n)-\sum_{i=1}^jf(p_i)+\sum_{k>j, p_k^2\le n}\sum_{c\ge 1,p_k^c\le n}f(p_k^c)(S(\frac{n}{p_k^{c}},k)+[c>1])
$$

边界：

$$
S(n,j)=0\quad(p_j>n)
$$

该转移复杂度为 $O(n^{1-\epsilon})$，~~虽然跑得飞快~~

考虑另一种转移（也可以通过上面的转移整理得）：

$$
S(n,j)=f(p_{j+1})+S(n,j+1)+\sum_{c\ge 1,p_{j+1}^c\le n} f(p_{j+1}^c)(S(\frac{n}{p_{j+1}^{c}},j+1)+[c>1])\\
$$

此时边界为：

$$
S(n,j)=
\left\{\begin{array}{lc}
    0\quad(p_j>n)\\
    p(n)-\sum_{i=1}^jf(p_i)\quad(p_{j+1}^2>n)
\end{array}\right.
$$

记忆化后复杂度为 $O(\frac{n^{\frac{3}{4}}}{\log n})$。但常数和空间巨大，一般情况不写。

```cpp
const int mod=1e9+7;
inline int qpow(int x,int k) {
	int ans=1; while (k) {
		if (k&1) ans=(ans*x)%mod;
		k>>=1; x=(x*x)%mod;
	} return ans;
}
inline int inv(int x) {return qpow(x,mod-2);}

// n ~ 1e10
// f(p) = p^k * (p^k - 1)
const int N=1e5+10,iv6=inv(6);
int n,sq,g1[N<<1],g2[N<<1],g[N<<1];
int npr[N],sump1[N],sump2[N],sump[N];
vector<int> pr;
vector<int> s[N];

void init() {
	npr[1]=1; pr.pb(0);
	for (int i=2;i<=1e5;i++) {
		if (!npr[i]) pr.pb(i);
		for (int j=1;j<pr.size();j++) {
			int p=pr[j];
			if (i*p>1e5) break;
			if (i%p==0) {
				npr[i*p]=1; break;
			} else npr[i*p]=1;
		}
	}
	for (int i=1;i<pr.size();i++) {
		sump1[i]=(sump1[i-1]+pr[i])%mod;
		sump2[i]=(sump2[i-1]+pr[i]*pr[i])%mod;
		sump[i]=(sump[i-1]+pr[i]*(pr[i]-1))%mod;
	}
}
int getid(int x) {
	if (x<=sq) return x;
	else return sq+n/x;
}
int sum2(int n) {
	n%=mod; return n*(n+1)%mod*(2*n+1)%mod*iv6%mod;
}
int sum(int n) {
	n%=mod; return n*(n+1)/2%mod;
}

// n^{1-\epison}
int S(int n,int j) {
	if (pr[j]>n) return 0;
	int ans=g[getid(n)]-sump[j];
	for (int k=j+1;k<pr.size()&&pr[k]*pr[k]<=n;++k) {
		for (int c=1,p=pr[k],fp=p*(p-1)%mod;p<=n;p*=pr[k],fp=p%mod*(p%mod-1)%mod,++c)
			ans=(ans+fp*(S(n/p,k)+(c>1)))%mod;
	}
	return ans%mod;
}

// n^{1-\epison}
// 记忆化后 n^{3/4}/logn，但常数 & 空间巨大
int S(int n,int j) {
	if (pr[j]>n) return 0;
	if (j==pr.size()-1||pr[j+1]*pr[j+1]>n) return g[getid(n)]-sump[j];
	if (s[getid(n)][j]) return s[getid(n)][j];
	int ans=pr[j+1]*(pr[j+1]-1)%mod+S(n,j+1);
	for (int c=1,p=pr[j+1],fp=p*(p-1)%mod;p<=n;p*=pr[j+1],fp=p%mod*(p%mod-1)%mod,++c) {
		ans=(ans+fp*(S(n/p,j+1)+(c>1)))%mod;
	}
	return ans%mod;
}

void solve() {
	cin>>n; sq=sqrt(n);
	for (int l=1;l<=n;) {
		int m=n/(n/l);
		g1[getid(m)]=sum(m)-1;
		g2[getid(m)]=sum2(m)-1;
		l=m+1;
	}
	for (int i=1;i<pr.size();i++) {
		int p=pr[i];
		if (p*p>n) break;
		for (int l=1;l<=n;) {
			int r=n/(n/l),m=n/l;
			if (p*p>m) break;
			g1[getid(m)]=(g1[getid(m)]-p*(g1[getid(m/p)]-sump1[i-1])%mod)%mod;
			g2[getid(m)]=(g2[getid(m)]-p*p%mod*(g2[getid(m/p)]-sump2[i-1])%mod)%mod;
			l=r+1;
		}
	}
	for (int l=1,i=1;l<=n;)  {
		int m=n/(n/l);
		g[getid(m)]=g2[getid(m)]-g1[getid(m)];
		l=m+1;
	}

	cout<<((S(n,0)+1)%mod+mod)%mod<<'\n';
}
```

## 筛法的实现细节

对于各种情况下 $f(\lfloor\frac{n}{i}\rfloor)$ 的存储，如果使用 ``map`` 会多一只 $\log$；

考虑对 $i$ 进行分块，将 $i\le \sqrt{n}$ 的值存入 ``f[i]``，对 $i>\sqrt{n}$ 的值存入 ``f[sqrt(n)+n/i]``；这样空间就为 $2\sqrt{n}$

筛法递归常数不大，在不卡常的情况下可以写记搜

杜教筛常数略大于线性筛预处理，通常可以多预处理 $\frac{1}{2}$ 左右的数

注意：什么时候要取模，什么时候不取模（可以写一个取模类）

> Written with [StackEdit中文版](https://stackedit.cn/).
