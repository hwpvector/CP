

# Lyndon & Runs 理论

## 0 Preliminaries


1.  我们定义两个字符串 $a$ 和 $b$，如果 $a$ 的字典序 $<b$，则我们称 $a < b$。
2.  如果 $a$ 是 $b$ 的前缀且 $a \ne b$，则我们称 $a \sqsubset b$。
3.  如果 $a$ 是 $b$ 的前缀，则我们称 $a \sqsubseteq b$。
4.  如果 $a < b$ 且 $a$ 不是 $b$ 的前缀，则我们称 $a \triangleleft b$。即 $a \triangleleft b \Longleftrightarrow (a < b) \wedge (a \not\sqsubseteq b)$。Fact：如果 $a \triangleleft b$，则 ${au} < {bv}$。
5.  ${abc}$ 表示拼接 $a, b, c$ 三个字符串。
6.  $a^n$ 表示 $n$ 个 $a$ 拼接在一起。e.g. ${a^2b} = {aab}$
7.  $\epsilon$ 表示空串。
8.  我们定义字符集为 $\Sigma$，组成的字符串为 $\Sigma^*$，$\Sigma^+ = \Sigma^* \setminus \{\epsilon\}$
9.  $\operatorname{pref}(a)$ 表示所有 $a$ 的前缀的集合，$\operatorname{suf}(a)$ 表示所有 $a$ 的后缀的集合（包含 $a$ 和 $\epsilon$）
10.  $\operatorname{pref}^+(a) = \operatorname{pref}(a) \setminus \{a,\epsilon\},\ \operatorname{suf}^+(a) = \operatorname{suf}(a) \setminus \{a, \epsilon\}$
11.  若无特殊定义，字符串 $s$ 是从 $0$ 开始。
12.  $|s|$ 表示 $s$ 的长度，$s[i..j]$ 表示 $s$ 的一个子串，第一个字符的标号为 $i$，最后一个字符的标号为 $j$。
13.  $\hat{w}=w\$$，其中 $\$$ 是一个比字符集里面任何数小的字符。

## 1 Lyndon Words

### 1.1 Definition

**Lyndon Word**：一个字符串是一个 **Lyndon Word** 当且仅当 $\forall a$ 的后缀 $b$，有 $a < b$。

还有一个定义：对于一个 $n$ 的串有 $n$ 个循环同构，则其中严格最小的那个是一个 Lyndon Word。请注意，最小循环表示不一定是 Lyndon Word，因为可能是一个 Lyndon Word 的若干次方，比如 $\text{ababab}$。

比如：$\text{ab}$ 是一个 Lyndon Word，但是 $\text{aba}$ 不是。

$\mathcal L$ 表示所有 Lyndon Word 组成的集合。

### 1.2 Chan-Fox-Lyndon Factorization

又称 Lyndon Decomposition。

定义 $\operatorname{CFL}(s)$ 是一个对于 $s$ 串的划分，即划分成了 ${w_1w_2\cdots w_k} = s$，使得所有 $w_i$ 是 Lyndon Word，并且 $w_1 \ge w_2 \ge \cdots \ge w_n$。

比如：串 $\text{bbababaabaaabaaaab}$ 的 Lyndon 分解是 $\text{b|b|ab|ab|aab|aaab|aaaab}$。

#### Theorem 1.2.1 Lyndon Concatanation

**[Theoreom]** 如果 $a, b \in \mathcal L$，且 $a < b$，则 ${ab} \in \mathcal L$。

**[Proof]**

由于 $a < b$，我们有 ${ab} < b$。接下来我们分两种情况讨论。

1.  当 $a \not \sqsubseteq b$ 时：根据 $a < b$，我们有 $a \triangleleft b$。所以 ${ab} \triangleleft b \implies {ab} < b$。
    
2.  当 $a \sqsubseteq b$ 时：令 $b={ac}$，则 ${ab} = {a^2c}$。因为 $b \in \mathcal L$，所以 ${ab} < b \implies {a^2c} < {ac} \implies {ac} < c$，所以 $b < c$。
    

所以，$\forall d \in \operatorname{suf}^+(b), \ {ab} < b < d \implies \forall c \in \operatorname{suf}^+(a),\ a \triangleleft e \implies {ab} \triangleleft {eb}$。$\blacksquare$

#### Theorem 1.2.2 Existence of CFL

**[Theoreom]** 对于任意的串 $s$，$\operatorname{CFL}(s)$ 一定存在。

**[Proof]** 构造法。我们考虑，单个的字母一定是 Lyndon Word。

根据 **[Theorem 1.2.1 Lyndon Concatanation]**，我们可以把字典序小的两个 Lyndon Word 并起来，所以我们把所有的字典序单增的序列都并起来，剩下的就是一个合法的 CFL。$\blacksquare$

#### Theorem 1.2.3 Uniqueness of CFL

**[Theoreom]** 对于任意的串 $s$，$\operatorname{CFL}(s)$ 一定唯一。

**[Proof]** 反证法，假设有两种方案。我们考虑第一个不同的位置的情况，可以很容易地得到矛盾，和 CFL 的定义矛盾。$\blacksquare$

然后我们就得到了 **CFL 存在且唯一**。由此有两个推论：

#### Theorem 1.2.4 Lyndon Suffixes and Lyndon Prefixes

**[Theoreom]** $w_1$ 是最长的 Lyndon 前缀且 $w_k$ 是最长的 Lyndon 后缀。

**[Proof]** 反证法。因为如果 $w_1$ 不是最长，那么还能再拼，产生了两个合法的 CFL，和 *[Theorem 1.2.3 Uniqueness of CFL]* 矛盾。所以 $w_1$ 是最长的 Lyndon 前缀。

$w_k$ 同理。$\blacksquare$

#### Theorem 1.2.5

**[Theoreom]** 一个字符串 $s$ 的最小后缀是 $w_k$。

**[Proof]**

首先，我们有这样的一个 CFL：

![](/imgs/2024-10-31/XFRQNsCTt9iD0KBQ.png)

首先，我们记 $w_n$ 的起始位置为 $pos$，则显然

![输入图片说明](/imgs/2024-10-31/GNFo5rJeRI2SpxIa.png)

如图，最小后缀的其实位置不可能 $>pos$，因为根据 Lyndon Word 的定义，$w_n$ 的每个后缀都大于他自身。

接下来我们考虑最小后缀在另一个位置的情况，即他在另一个 $w_i$ 之中

![输入图片说明](/imgs/2024-10-31/jXGCSSO7i5EpPvzN.png)

根据 $w_i \ge w_{i+1}\ge \cdots \ge w_n$，而 $w_i$ 的一个后缀 $> w_i$，所以这个后缀大于 $w_n$。

所以唯一可能的最小后缀就是 $w_n$。

简单来说，假设最小后缀是 ${xw_{i+1}w_{i+2}\cdots w_{k}}$ 而不是 $w_k$ 且 $|x| < |w_i|$。我们有 ${x w_{i + 1} \dots w_k} \geq x > w_i \ge w_k$，矛盾。$\blacksquare$

### 1.3 Duval's Algorithm

求出 Lyndon 分解。简要思想：

> ${uav} \in \mathcal{L}$，$u,v,h \in \Sigma^*$，$a<b\in\Sigma$，$k\ge1$
> 
> 1.  ${(uav)^k ub} \in \mathcal{L}$
> 2.  $\operatorname{CFL}({(ubv)^k uah}) = {(ubv)^k} \operatorname{CFL}({uah})$
> 3.  $\operatorname{CFL}({(uv)^k u}) = {(uv)^k} \operatorname{CFL}(u)$

换成代码实现就是：

我们需要维护两个部分：$ubv$ 和 $u$。

*   如果可以拼到当前的串的末尾就拼上去。
*   否则就是一个新的 Lyndon Word。（如果碰到一个比当前的小的东西，则我们更新 $ubv$，否则我们就更新 $u$）。
*   如果不满足 Lyndon Word 字段序递减的条件，则根据 **[Theorem 1.2.1 Lyndon Concatanation]**，我们可以将两个 Lyndon Word 合并。

空间复杂度 $\mathcal O(1)$。接下来证明一下时间复杂度。

最优情况为一个分解走到底，$\mathcal O(n)$。

最坏情况为不停地在重新找，由于至多回退 $n$ 次，每次回退的距离不超过前进的距离，所以是 $\mathcal O(n)$。

### Template

```cpp
vector<pii> lyndon;
int n; string str;

void duval() {
	cin>>str; n=str.size(); str=" "+str;
	for (int i=1,j=1,k=2;i<=n;j=i,k=i+1) {
		while (k<=n&&str[j]<=str[k]) {
			if (str[j]==str[k]) ++j,++k;
			else j=i,++k;
		}
		while (i<=j) lyndon.pb({i,k-j}),i+=k-j;
	}
	for (auto [i,k]:lyndon) {
		cout<<i<<' '<<i+k-1<<'\n';
	}
}
```

### Ex A Minsuf of Prefixes

> 求前缀的最小后缀。

不用 Lyndon 的话传统做法是后缀树，维护一下最右边的路径即可。

根据 **[Theorem 1.2.5 Theorem of Minsuf]**，$w_n$ 就是最小后缀，所以我们可以记忆化一下 CFL 分解：

我们知道，Lyndon 分解的时候是一个 Lyndon 串不停地在重复，如果碰到冲突会重新跳，我们考虑把这个东西记忆化，记为 $\operatorname{minsuf}(i)$。

时间复杂度 $\mathcal O(n)$。

### Ex B Maxsuf of Prefixes

> 求前缀的最大后缀。

首先显然可以把字符集的比较的顺序反过来。

我们考虑一个 Lyndon 分解的结构：$w_1^{p_1} w_2^{p_2} \ldots w_k^{p_k}$，不难发现答案就是 $w_k^{p_k}$。

### Ex C Minimal Rotation

> 最小循环表示

对着 ${x}$ 求一遍 CFL，找到前半串开始的最长 Lyndon Word 就是了。时间复杂度 $\mathcal O(n)$。

### Ex D [Yandex.Algorithm 2015 Round 2.2] Lexicographically Smallest String

> 给你一个字符串 $S$，你可以选择任意一个区间进行翻转，使操作后的字符串字典序最小。$|S| \le 10^7$

来观察性质。

**Lemma**：如果我们在开头有一个不大于其他所有的字符的字符，则我们一定不动这个字符，否则一定会反转某一个前缀。

假设翻转一个前缀 $s[1..t]$ 后的第一个字符比当前的小，则显然 $s[t] < s[1]$，矛盾。

如果第一个字符不是最小，那么我们应该找到最小的字符 $s[t]$ 并且翻转前缀 $s[1..t]$，使得第一个字符成为最小。$\blacksquare$

我们把原串翻转的结果记为 $S^R$。

我们考虑翻转过后的一个前缀是 $S^R$ 中的一个后缀，而我们要求翻转之后的字典序最小，所以我们需要挑选 $S^R$ 中的最小后缀（这里的最小后缀和一般说的有所不同，只要比其他不存在前缀关系都要小即可，所以可能不止一个）。

可以发现，所以符合条件的后缀都是最长的一个的前缀。

我们注意到，一个串 $S'$ 是可以表示成若干的 $w$ 的幂和一个 $w$ 的前缀。

接下来我们考虑什么时候需要翻转。

如果 $|A|=|B|$，则 $A + A + C < A + C + B < C + B + B$ 或 $A+A+C>A+C+B>C+B+B$ 或 $A+A+C=A+C+B=C+B+B$。

我们考虑 $A=w, B=w^R,C=w'$ 的情况，发现满足条件的后缀之中只有最长的和最短的后缀可能对答案产生贡献。

接下来我们应该使用 Duval 算法来找到 $\operatorname{CFL}(S^R)$ 来找到这些后缀。

时间复杂度 $\mathcal O(n)$。

### Ex E [Codeforces 594 E] Cutting the Line

> 题意：给你一个字符串 $S$，你可以把他分成 $k$ 段，并且翻转其中的若干段，使操作后的字符串字典序最小。$1\le k\le |S| \le 5\times 10^6$

## 2 Significant Suffixes

### 2.1 Definition

我们令 $\operatorname{minsuf}(u)$ 为 $u$ 的最小后缀，且 $\operatorname{minsuf}(u, v) = \min _{w \in \operatorname{suf}(u)} wv$。

**Significant Suffixes**：一个字符串 $u$ 的 Significant suffixes 是一个集合：$\Lambda(u) = \{\arg\min_{w\in \operatorname{suf}(u)} wv|v \text{ is a string}\}$。

$\operatorname{minsuf}(s)$ 表示 S 字典序最小的后缀，且

由 $\operatorname{minsuf}$ 的性质可知，$\operatorname{minsuf}(u, \epsilon) = \operatorname{minsuf}(u)$。$\implies \operatorname{minsuf}(u) \in \Lambda(u)$。

所以，显然 $\forall u \in \Lambda(u),\ \operatorname{minsuf}(u) \sqsubseteq u$。

我们注意到一个 CFL 分解中的 Lyndon Words 是存在一定的循环的。因此，我们可以记一个 CFL 为次方的形式。

$$\operatorname{CFL}(u) = {{w_1}^{k_1}{w_2}^{k_2}\cdots {w_n}^{k_n}} $$

我们记 $s_i$ 为一个后缀，即 $s_i = {{w_i}^{k_i}{w_{i+1}}^{k_{i+1}}\cdots{w_n}^{k_n}}$。边界：$s_{n+1} = \epsilon$。

### 2.2 Significant Theorem

#### Lemma 2.2.1

**[Lemma]** 如果 $u^\infty < v$，则 $v > {uv} > {u^2v} > \cdots$。

**[Proof]**

$u^\infty < v \implies u^\infty < {uv}$。

令 $u = {xay}$，$v = {(xay)^k xbh}$，其中 $x,y,h\in\Sigma^*$，$a,b\in\Sigma$，$a<b$。

我们有 $v \succ uv \Longleftrightarrow (xay)^{k - 1} xbh \succ (xay)^k xbh \Longleftrightarrow xbh \succ (xay) xbh$。

$v>{uv} \implies {u^iv} > u^{i+1} \implies \blacksquare$

同理如果 $u^\infty > v$，则 $v < {uv} < {u^2v} < \cdots$。

#### Theorem 2.2.2

**[Theoreom]** $$ \Lambda(u)\subseteq {s_i | i \in [1,n]} $$

**[Proof]**

反证法：如果这个命题不成立，则我们分类讨论

**i)** 假设有一个串 $v = {b{w_i}^ks_{i+1}} \in \Lambda(u)$，$|b| < |w_i|,\ 0 \le k < k_i$。

$w_i \in \mathcal L \implies w_i \triangleleft b \implies s_i = {w_is_{i+1}} < {bs_{i+1}}$，矛盾。

**ii)** 假设有一个串 $v = {{w_i}^ks_{i+1}} \in \Lambda(u)$，$1 < k < k_i$。

根据 **[Theorem 2.2.1 Infinite Theorem]**，如果 ${w_i}^\infty < s_{i+1}$，则 ${{w_i}^{k_i}s_{i+1}} < {{w_i}^{k_i - 1}s_{i+1}}<\cdots<s_{i+1}$，否则 ${{w_i}^{k_i}s_{i+1}} > {{w_i}^{k_i-1}s_{i+1}} > \cdots > s_{i+1}$。

我们令 $\lambda = \min \{i : s_{i+1} \sqsubset s_i\}$。$\forall i \ge \lambda, \ w_i = {s_{i+1}y_i},\ x_i = {y_is_{i+1}}$。$\implies s_i = {{w_i}^{k_i}s_{i+1}}= {(s_{i+1}y_i)^{k_i}s_{i+1}} = {s_{i+1}{x_i}^{k_i}}$。

根据 CFL 的性质，$s_{\lambda} \triangleleft w_{\lambda - 1}$。所以 $\Lambda(u)\subseteq \{s_i | i \in [1,n]\}$。$\blacksquare$

### 2.3

#### Theorem 2.3.1

**[Theoreom]** 如果有 $2$ 个串 $u$ 和 $v$，满足 $|u| \le |v|$，则我们有

$$\begin{aligned}\Lambda(uv) &\subseteq \Lambda(v) \cup \{\operatorname{maxsuf}^R(u, v)\} \\&= \Lambda(v) \cup{\max _{s \in \Lambda(u)}}^R \{sv\}\end{aligned} $$

理由很简单，因为 $\{\operatorname{maxsuf}^R(u, v)\}$ 也是一个 Significant Suffix，随意我们就可以把它展成第二行的式子的形式。$\blacksquare$

#### Theorem 2.3.2

**[Theoreom]** 一个字符串 $S$ 的 Significant Suffixes 至多有 $\log n$ 个。

**[Proof]**

原命题可以很容易地转化为：

如果两个 Significant Suffixes $u$，$v$ 满足 $|u| < |v|$，那么 $2|u| < |v|$。

反证法。设存在 $|u| < |v| < 2|u|$。因为 $u, v \in \operatorname{suf}^+(u)$，所以 $u \in \operatorname{suf}^+(v)$。

所以我们可以非常容易地知道，$u \triangleleft v$。$\implies v$ 有一个长度为 $|v| - |u| < \frac {|v|} 2$ 的周期，记为 $T$。

所以，$u = {Tw}, v = {T^2w}$。

由于 $u$ 是一个 Significant Suffix，因此存在串 $t$，满足 $vt>ut$，即 ${T^2wt} > {Twt} \implies {Twt} > {wt}$。

而 $w \in \operatorname{suf}^+(s)$，所以与 $u$ 是 Significant Suffix 矛盾。$\blacksquare$

### 2.4 Facts

我们知道 $\Lambda(S)$ 中有很多串，其中最短的是 $\operatorname{minsuf}(S)$，而最长的是 $\operatorname{maxsuf}^R(S)$。这里的 $^R$ 代表 reverse。

*   $\Lambda(u) = \{s_{\lambda}, \cdots, s_{n+1}\}$
*   $\operatorname{minsuf}(u) = s_n$
*   $\operatorname{maxsuf}^R(u) = s_\lambda$
*   ${x_\lambda}^\infty > \cdots > {x_m}^\infty$
*   我们有一个串 $v$，${x_i}^\infty > v > {x_{i+1}}^\infty$。则 ${s_\lambda v} > \cdots > {s_{i+1}v} < \cdots < {s_kv}$
*   对于两个串 $u$ 和 $v$，有 $|u|<|v|$，$\Lambda({uv}) \subseteq \{\operatorname{maxsuf}^R(u, v)\} \cup \Lambda(v) = \{\min_{w \in \Lambda(u)}{wv}\} \cup \Lambda(v)$


### Ex F [JSOI2019] 节日庆典

> 题意：给你一个字符串 $S$，求 $S[1,i]$ 的最小表示的起始位置 ，其中 $i=1,2,\cdots,|S|$。$1\le |S|\le 3\times 10^6$

### Ex G [ZJOI2017] 字符串

> 题意：维护动态字符串，值域 $10^9$，区间加，求区间 $\operatorname{minsuf}$ 的位置。$|S| \le 2 \times 10 ^ 5,m \le 3 \times 10 ^ 4$

我们先考虑不带修的情况。

由 **[Theorem 2.3.1 Lambda Subset Theorem]**，我们可以很容易地想到考虑建一棵线段树来维护 Significant Suffixes。

细节：如果线段树的 `mid = l + r >> 1`，则左边的区间比右边长一些。但是上面的这个结论对于 $|u| \le |v|$ 有效，所以我们需要调整一下，使得左儿子比右儿子要长一些（即：`mid = l + r + 1 >> 1`，使得左儿子总不比右儿子短）

可以存一下当前代表的串的所有 Significant Suffixes，然后直接考虑合并（把右边的所有的直接加进来，左边的都循环一遍，字典序最长的加进去）得到父节点的 Significant Suffixes 即可。（看不懂的看代码）

由 **[Theorem 2.3.2 Lambda Log Theorem]** 可知，每一个集合都是 $O(\log n)$ 大小的。这样的话，我们求出了每一个线段树上的区间的 Significant Suffixes。然后查询就在这 $O(\log n)$ 个区间内求 Significant Suffixes 的并，暴力比较即可。所以我们需要一个 $O(1)$ 比较两个串的方法（否则复杂度就挂了）。所以如果不带修的话我们可以考虑 SA。

接下来考虑带修的情况。

我们需要快速地求两个串的 LCP，又有一个线段树，所以可以很自然地想到一个线段树+字符串哈希+二分LCP的算法。复杂度 $O(q \log^4 n)$，慢了点，我这种人傻常数大的就不用想了。

我们考虑分块维护一些哈希，分 $\sqrt n$ 的块。我们维护一下每个点到块的末端的哈希值，然后维护一下每个块到串的末尾的哈希值。然后我们可以记一个块的全局的偏移量，就可以算了。每次查询的时候，我们只需要查 $2$ 次即可，$O(1)$ 查找。最终是 $O(q \log ^3 n + q\sqrt n)$ 的复杂度。

## 3 Lyndon Array

### 3.1 Definition

我们有一个字符串 $s$，则

**Lyndon Array**：$\mathcal L[i] = \max \{j : s_i \cdots s_{j-1} \in L\}$，其中 $L$ 表示 Lyndon 串的集合。在 $\prec_l$ 意义下的 $\mathcal L$ 记为 $\mathcal L_l$。

### 3.2 Non Intersecting Substrings

#### Theorem 3.2.1 Non Intersecting Lyndon Substrings

**[Theoreom]** 最长的 Lyndon 子串是无交集的，即 $i < j < \mathcal L[i]$，我们有 $\mathcal L[j] \le \mathcal L[i]$。

**[Proof]**

假设存在 $i,j$ 使得 $\mathcal L[i] < \mathcal L[j]$。

假设 $u = s_i \cdots s_{j-1}$，$v = s_j \cdots s_{\mathcal L[i] - 1}$，$w = s_{\mathcal L[i]} \cdots s_{\mathcal L[j] - 1}$，且 $u,v,w$ 满足 ${uv}, {vw} \in L$。

$\forall s \in \operatorname{suf}^+({uvw})$，且满足 $|s| \le |v| + |w|$，有 $s \triangleleft {vw} \sqsupseteq v \triangleleft {uv}$。$\implies {uvw} \triangleleft s$。

$\forall s \in \operatorname{suf}^+({uvw})$，且满足 $|s| > |v| + |w|$，有 ${svw} \sqsupseteq {sv} \triangleleft {uv}$。$\implies {uvw} \triangleleft {svw}$。

所以 ${uvw} \in L$，矛盾。$\blacksquare$

### 3.3 Suffix & Lyndon Arrays

我们设 $\operatorname{suf}(i) = s_i \cdots s_{n-1}$，即一个后缀。

而我们有 $s_i \cdots s_{\mathcal L[i] - 1} \triangleleft s_j \cdots s_{\mathcal L[i] - 1}$。

$\implies \operatorname{suf}(i) \triangleleft \operatorname{suf}(j)\quad(i<j<\mathcal L[i])$

于是我们设

$$\operatorname{NSV}(i) = \min\{\{j > i : \neg(\operatorname{suf}(i) \triangleleft \operatorname{suf}(j))\} \cup \{n\}\} $$

显然 $\mathcal L[i] \le \operatorname {NSV}(i)$。

我们还有 $\neg(\operatorname{suf}(i) \triangleleft \operatorname{suf}(j)) \Longleftrightarrow \operatorname{suf}(j) \sqsubseteq \operatorname{suf}(i) \vee \operatorname{suf}(j) \triangleleft \operatorname{suf}(i) \Longleftrightarrow \operatorname{rank}(i) > \operatorname{rank}(j)$。

#### Theorem 3.3.1 NSV Theorem

**[Theoreom]** $\mathcal L[i] = \operatorname{NSV}(i)$

**[Proof]**

原命题可以很方便地转化为 $s_i \cdots s_{\operatorname{NSV}(i) - 1} \in L$。

分类讨论：

1.  如果 $\operatorname{NSV}(i) = n$，$s_i \cdots s_{\operatorname{NSV}(i) - 1} = \operatorname{suf}(i)$
    
2.  否则的话我们肯定有一些 $j$ 使得 $\operatorname{suf}(i) \triangleleft \operatorname{suf}(j)$。
    
    然后我们继续来讨论：
    
    i) 如果 $s_1 \cdot s_{\operatorname{NSV}(i) - 1} \triangleleft s_{j} \cdots s_{\operatorname{NSV}(i) - 1}$，易证。
    
    ii) 反之，结合 $\operatorname{suf}(i + (\operatorname{NSV}(i) - j) - 1) \triangleright \operatorname{suf}(i)>\operatorname{suf}(\operatorname{NSV}(i))$，易证矛盾。（你看不出来？明显与 $\operatorname{suf}(i) \triangleright \operatorname{suf}(j)$ 矛盾）。$\blacksquare$
    

### Ex H 简单字符串

> 给定一个字符串 $S$，每次询问 $S[l,n]$，可以任意分成至多 $k$ 段 $u_1, u_2,\cdots u_p$，当 $\max_{1\le i\le p} u_i$ 最小时，$\max_{1\le i\le p} u_i=S[a,b]$ 的 $a,b$ 的值。如有多个，输出 $a$ 最小的。$|S|,q\le 10^5$

考虑 $\operatorname{CFL}(S[l,n])=w_1^{k_1}w_2^{k_2}\cdots w_n^{k_n}$

若 $k> k_1$，则 $\max_{1\le i\le p} u_i=w_1$

否则，我们尽可能均分 $w_1^{k_1}$，

- 若 $k_1 \equiv 0\mod k$，则 $\max_{1\le i\le p} u_i=w_1^{\frac{k}{k_1}}w_2^{k_2}\cdots w_n^{k_n}$
- 否则，$\max_{1\le i\le p} u_i=w_1^{\lceil\frac{k}{k_1}\rceil}$

我们现在需要求出 $w_1$ 和 $k_1$。显然，$w_1=\mathcal L[l]$。$k_1$ 可以通过 $S[l,n]$ 和 $S[l+|L[l]|,n]$ 的 lcp 算出。

## 4 Runs

### 4.1 Definition

定义一个字符串 $|S|$ 里的一个 run，指其内部一段两侧都不能扩展的**周期子串**，且周期至少完整出现两次。

严格地说，一个 run 是一个 三元组 $(i,j,p)$，满足 $p$ 是 $S[i..j]$ 的最小周期，$j-i+1 \ge 2p$，且满足如下两个条件：

*   要么 $i=1$，要么 $S[i-1]\ne S[i-1+p]$；
*   要么 $j=n$，要么 $S[j+1] \ne S[j+1-p]$。

例如：$S = \text{aababaababb}$ 之中有 7 个 runs：$S[1..2] = \text a^2$，$S[1..10] = (\text{aabab})^2$，$S[2..6] = (\text{ab})^{2.5}$，$S[4..9] = (\text{aba})^2$，$S[6..7] = \text a^2$，$S[7..10] = (\text{ab})^2$，$S[10..11] = \text b^2$。

定义 $Runs(w)$ 表示字符串 $w$ 的所有 runs 的集合。

$\rho(n)$ 表示了在一个长为 $n$ 的字符串之中至多有多少组 runs，而 $\sigma(n)$ 表示了在一个长为 $n$ 的字符串之中所有 runs 的幂之和的最大值。

**Lyndon Root**：令 $r=(i,j,p)$ 是一个run，则他的 Lyndon Root 是一个 $s[i..j]$ 的长度为 $p$ 的 Lyndon 子串。

每一个 run 都有一个 Lyndon root。

### 4.2 Linear Runs

#### Theorem 4.2.1 Linear Runs Theorem

我们假设 $\prec^0$ 表示 $<$，而 $\prec^1$ 表示 $<^R$。（此处的 $^R$ 表示 reverse，给 $\prec$ 标号是为了方便）

$\prec^0$ 和 $\prec^1$ 的对应的 Lyndon Array 是 $\mathcal L^0$ 和 $\mathcal L^1$.

**[Theoreom]** $\rho(n) \le 2n$

**[Proof]**

原命题可以转化为

> 对于每个 runs，我们有存在 $i$ 和 $t$ 使得 $s[i..\mathcal L^t[i] - 1]$ 是 Lyndon root。

我们令 $w$ 是 Lyndon root，$w=s[k..s-1]$。

分类讨论：

1.  如果 $j=|S|$，
    
    我们可以把 $s[k .. |S| - 1]$ 表示成 ${w^pw'} \ (p \in N, w' \in \operatorname{pref}(w))$。
    
    因为 $\operatorname{CFL}({w^pw'}) = {w^p\operatorname{CFL}(w')}$，所以 $w$ 是从 $k$ 开始的最长 Lyndon 前缀。
    
2.  如果 $j<|S|$，
    
    我们可以把 $w$ 表示成 ${uab}$，其中 $a \ne b$。
    
    所以我们可以把 $s_k \cdots s_{|S| - 1}$ 表示成为 ${(uav)^pub}$。
    
    我们不妨假设 $b \prec^t a$、
    
    因为我们有 $\operatorname{CFL}^t({(uav)^pubh}) = (uav)^p\operatorname{CFL}^t({ubh})$，所以 ${uav}$ 是 $\prec^t$ 下的最长 Lyndon 前缀。$\blacksquare$
    

#### Theorem 4.2.2 The "Runs" Theorem

**[Theoreom]** $\rho(n)<n,\sigma(n)\leq3n-3$

**[Proof]**

~几乎从 WC2019 课件搬运的证明~

定义 $Beg(I)$ 表示 $I$ 中所有区间的起始端点的集合。

##### Lemma A

**[Lemma]** 对于一个串的 Lyndon Array $\mathcal L^0[i]$ 和 $\mathcal L^1[i]$，总有 $\mathcal L^{l}[i] = [i..i], \mathcal L^{1-l}[i] = [i..j] (j\ne i)$，其中 $l\in \{0,1\}$。

**[Proof]**

令 $k=\max\{k'\ |\hat{w}_{k'}\ne \hat{w}_i,k'>i\}$。

由 **[Theorem 1.2.1 Lyndon Concatanation]** 可得：

*   若 $\hat w_k < \hat w_i$，则 $\mathcal L^0[i]=[i..i]$，且 $\mathcal L^1[i]=[i..j]\ (j\geq k>i)$。
*   若 $\hat w_k > \hat w_i$，则 $\mathcal L^1[i]=[i..i]$，且 $\mathcal L^0[i]=[i..j]\ (j\geq k>i)$。$\blacksquare$

##### Lemma B

**[Lemma]** 若 $r=(i,j,p)$ 为一个run，则对于 $\hat{w}[j+1]\prec_l \hat{w}[j+1-p]$ 的 $l$，$\forall r$ 的 $\prec_l$ 意义下的 Lyndon Root $\hat w[i_{\lambda}..j_{\lambda}]$ 都与 $\mathcal L^l(i_{\lambda})$相等。

**[Proof]**

$\because \hat{w}[j+1]\ne\hat{w}[j+1-p]$，令 $l\in\{0,1\}$ 满足 $\hat{w}[j+1]\prec_l\hat{w}[j+1-p]$。

令 $\lambda=[i_{\lambda}...j_{\lambda}]$ 为 $r$ 的 $\prec_l$ 意义下的一个 Lyndon Root，由 **[Theorem 1.2.1 Lyndon Concatanation]**，$[i_{\lambda}...j_{\lambda}]=\mathcal L^l(i_{\lambda})$。$\blacksquare$

对于一个run $r=(i,j,p)$，令 $B_r=\{\lambda=[i_{\lambda}...j_{\lambda}]|\lambda$ 为 $r$ 的 $\prec_l$ 意义下的一个 Lyndon Root 且 $i_{\lambda}\ne i\}$。即 $B_r$ 表示所有 $r$ 的关于 $\prec_l$ 的 Lyndon Root 构成的集合，但要除去开头位置 $i$ 处开始的 Lyndon Root。有 $|Beg(B_r)|=|B_r|\geq \lfloor e_r-1\rfloor\geq 1$，其中 $e_r$ 为 $r$ 的指数。

##### Lemma C

**[Lemma]** 两个不同的 run $r,r'$，$Beg(B_r)\cap Beg(B_{r'})$ 为空。

**[Proof]**

反证，假设存在 $i\in Beg(B_r)\cap Beg(B_{r'})$，并且 $\lambda=[i...j_{\lambda}]\in B_r$，$\lambda'=[i...j_{\lambda'}]\in B_{r'}$。

令 $l\in\{0,1\}$ 满足 $\lambda=\mathcal L^l[i]$，由于 $\lambda\ne \lambda'$，有 $\lambda'=\mathcal L^{1-l}[i]$。

由 **Lemma A**，$\lambda$ 和 $\lambda'$ 中有且只有一个为 $[i..i]$。

不妨设 $\lambda=[i..i]$，那么 $j_{\lambda'}>i$。

由于 $w[i...j_{\lambda'}]$ 为一个 Lyndon Word，有 $w[i]\ne w[j_{\lambda'}]$。

由 $B_r$ 和 $B_{r'}$ 的定义，$r$ 和 $r'$ 的开始位置均小于 $i$，这意味着 $w[i-1]=w[i]$（由 $r$ 的周期性），并且 $w[i-1]=w[j_{\lambda'}]$（由 $r'$ 的周期性）。矛盾 $\blacksquare$

任意的一个 run $r$ 可以被赋予一个两两不交的非空位置集合 $Beg(B_r)$。并且，由于 $1\notin Beg(B_r)$ 对于任意的一个 $r$ 均成立，有 $\sum_{r\in Runs(w)}|B_r|=\sum_{r\in Runs(w)}|Beg(B_r)|\leq |w|-1$。

考虑字符串 $w$，由于对于任意 $r\in Runs(w)$，有 $|B_r|\geq1$，由 **Lemma C**，有 $|Runs(w)|\leq\sum_{r\in Runs(w)}|B_r|\leq |w|-1$。

考虑字符串 $w$，令 $e_r$ 表示 $r$ 的指数。由于对于任意 $r\in Runs(w)$，有 $|B_r|\geq \lfloor e_r-1\rfloor>e_r-2$，由 **Lemma C**，有 $\sum_{r\in Runs(w)}(e_r-2)<\sum_{r\in Runs(w)}\lfloor e_r-1\rfloor\leq\sum_{r\in Runs(w)}|B_r|\leq |w|-1$。因为 $|Runs(w)|\leq |w|-1$，可得 $\sum_{r\in Runs(w)}e_r\leq3n-3$。$\blacksquare$

### 4.3 Details about Implementation

根据以上证明中的 **Lemma B**，每一个 runs 都会对应一个 Lyndon root，所以如果我们把 Lyndon Array 算出来了，就可以把每个 runs 对应的 Lyndon root 求出来。

那么我们怎么求 Lyndon Array $\mathcal L^0$ 和 $\mathcal L^1$？

$$\mathcal{L}^0[i] = \mathrm{NSV}(i) = \min\{\{j > i : \neg(\mathrm{suf}(i) \triangleleft \mathrm{suf}(j))\} \cup \{n\}\} $$

而

$$\begin{aligned}\mathcal{L}^1[i] &= \mathrm{NSV}^R(i) \\&= \min\{\{j > i : \neg(\mathrm{suf}(i) \triangleleft^R \mathrm{suf}(j))\} \cup \{n\}\} \\&= \min\{\{j > i : \mathrm{suf}(i) \triangleleft \mathrm{suf}(j) \vee \mathrm{suf}(i) \sqsupseteq \mathrm{suf}(j))\} \cup \{n\}\}\end{aligned} $$

所以我们考虑对于字符串的每个后缀都维护他的 CFL，方法是在头上插入一个新字符，然后判断是否合法。根据 **[Theorem 1.2.1 Lyndon Concatanation]**，如果遇到一个 Lyndon word 大于下一个的情况，合并即可。可以保证正确性。

这里有一个实现上的细节，可能会好写一点。根据 **[Theorem 1.2.4 Lyndon Suffixes and Lyndon Prefixes]**，$w_i$ 是 $w_iw_{i+1}\cdots w_k$ 的最小前缀，所以比较两个 Lyndon word 的字典序相当于比较两个后缀的大小，而这个是比做一个 lcp 要简单多的。

所以至此 Lyndon Array 已经求完了。

接下来我们只要使用 Lyndon Array 扩展出 runs 就可以了，具体的做法是求出 lcp，即如果当前的 Lyndon Array $\mathcal L[i] = (l..r)$，则我们 lcp 求出最长的 $s[l..l+l_1-1]=s[r+1..r+l_1], s[l-l_2..l-1]=s[r-l_2..r-1]$。

根据 runs 的定义，如果 $l_1 + l_2 \ge 2(l - r + 1)$，那么我们就找到了一个 run $(l-l_2, r+l_1-1, r-l+1)$。

如果我们使用 $\mathcal O(n \log n)$ 的 SA 和 $\mathcal O(n \log n)-\mathcal O(1)$ 的 rmq 算法，我们就可以在 $\mathcal O(n \log n)$ 的时间复杂度之内求出所有的 runs。

如果我们使用 SAIS 和 $\mathcal O(n) - \mathcal O(1)$ 的 rmq 算法，我们就可以 $\mathcal O(n)$ 求出所有的 runs。

### Template

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

const int N=1e6+10;
int lyndon[N],lyndonr[N],n; string s;
struct hash{
	int base,mod;
	Bart getmod;
	int p[N],h[N];
	void init(int Base,int Mod,string s) {
		base=Base,mod=Mod; getmod.ini(mod); int n=s.size(); --n; p[0]=1;
		for (int i=1;i<=n;i++) p[i]=getmod(p[i-1]*base);
		for (int i=1;i<=n;i++) h[i]=getmod(h[i-1]*base+s[i]);
	}
	int gethash(int l,int r) {
		return getmod(h[r]-h[l-1]*p[r-l+1]);
	}
}h[2];
bool equ(int l1,int r1,int l2,int r2) {
	return h[0].gethash(l1,r1)==h[0].gethash(l2,r2)&&h[1].gethash(l1,r1)==h[1].gethash(l2,r2);
}
int lcp(int x,int y) {
	if (s[x]!=s[y]) return 0;
	int l=0,r=n-max(x,y);
	while (l<r) {
		int mid=r-((r-l)>>1);
		if (equ(x,x+mid,y,y+mid)) l=mid;
		else r=mid-1;
	} return l+1;
}
int lcpr(int x,int y) {
	if (s[x]!=s[y]) return 0;
	int l=0,r=min(x,y)-1;
	while (l<r) {
		int mid=r-((r-l)>>1);
		if (equ(x-mid,x,y-mid,y)) l=mid;
		else r=mid-1;
	} return l+1;
}
bool cmp(int x,int y) {
	if (s[x]!=s[y]) return s[x]<s[y];
	int l=lcp(x,y);
	return s[x+l]<s[y+l];
}
void getlyn() {
	lyndon[n]=n;
	for (int i=n-1;i>=1;i--) {
		if (cmp(i+1,i)) {lyndon[i]=i;continue;}
		lyndon[i]=n;
		for (int j=lyndon[i+1]+1;j<=n;j=lyndon[j]+1) {
			if (cmp(j,i)) {lyndon[i]=j-1; break;}
		}
	} s[n+1]='z'+1,s[0]='z'+1;
	lyndonr[n]=n;
	for (int i=n-1;i>=1;i--) {
		if (!cmp(i+1,i)) {lyndonr[i]=i;continue;}
		lyndonr[i]=n;
		for (int j=lyndonr[i+1]+1;j<=n;j=lyndonr[j]+1) {
			if (!cmp(j,i)) {lyndonr[i]=j-1; break;}
		}
	} s[n+1]='a'-1,s[0]='a'-1;
}

vector<array<int,3>> ans;
void getans(int l,int r) {
	int R=r+lcp(l,r+1);
	int L=l-lcpr(l-1,r);
	int p=r-l+1;
	if (R-L+1>=p*2) ans.pb({L,R,p});
}

signed main() {
	ios::sync_with_stdio(0);
	for (int i=1;i<=2e5;i++) s.pb('a');
	cin>>s; n=s.size(); s=" "+s; s[0]='a'-1;
	h[0].init(131,1e9+7,s);
	h[1].init(13331,998244353,s);
	s.pb('a'-1);
	getlyn();  
	for (int i=1;i<=n;i++)
		getans(i,lyndon[i]),getans(i,lyndonr[i]);
	sort(ans.begin(),ans.end());
	ans.erase(unique(ans.begin(),ans.end()),ans.end());
	cout<<ans.size()<<'\n';
	for (auto [l,r,p]:ans) cout<<l<<' '<<r<<' '<<p<<'\n';
}
```

## 5 Three Squares Lemma

### 5.1 Definition

**Squares**: 能表示成 $x^2$ 的串。

**Primitive Squares**: 不能再拆的 Squares，即最小周期为自身长度的一半的字符串。如 $x^2+x$ 一定是一个 Primitive Square。

### 5.2 Three Squares Lemma

#### Theorem 5.2.1 Three Squares Lemma

> 我们有 3 个 Primitive Squares，为 $u^2$，$v^2$ 和 $w^2$，满足 $|u|<|v|<|w|$。
> 
> 则我们有 $|u| + |v| \le |w|$。

反证法，假设 $|u|+|v| > |w|$ 成立，则 $|w|-|v|$ 为 $u$ 和 $v$ 的周期。

如果 $2|u| \le |v|$，则 $|w|-|v| < |u|$ 是 $u^2$ 的周期，故 $u^2$ 不是一个 Primitive Square，矛盾。

否则，$|u|$ 和 $|w|-|v|$ 都是 $v$ 的周期，而 $u^2$ 是一个 Primitive Square，所以 $u$ 不是一个周期串，则 $|u|+|w|>2|v|$。

我们令 $w=vs_1$，$u=s_1s_3$，$v=ws_2=s_1s_3s_2$。显然，$|s_2| < |s_1|$。

考虑串 $s_3 s_2$，它有周期 $|s_2|$。由于 $|s_1|$ 是 $u$ 的周期，可得 $s_3s_2$ 是 $u$ 的前缀，所以 $|s_3|$ 也是它的周期。

故 $r=\gcd(|s_2|,|s_3|)$ 是它的周期。而 $|s_2|$ 本身同时是 $u$ 的周期，因此可得 $r$ 是 $u$ 的周期。

接着考虑串 $u=s_1s_3$。它的周期有 $|s_1|$ 和 $r$，而 $r\le|s_3|$，于是 $r'=\gcd(r,|s_1|)$ 也是 $u$ 的周期。然而 $|s_1|$ 和 $|s_3|$ 都是 $r'$ 的倍数，这表示 $u^2$ 也有周期 $r'$，矛盾。$\blacksquare$

#### Theorem 5.2.2 Number of Primitive Squares

> Primitive Squares 的数量级为 $\mathcal O(n\log n)$。

显然。由此有推论：

> 对于一个串的所有 Runs 三元组 $(l,r,p)$，$\sum r-l+1-2p$ 的数量级为 $\mathcal O(n\log n)$

#### Theorem 5.2.3 Different Primitive Squares

> 本质不同的 Primitive Squares 不超过 $2n$ 个。

每个位置最多出现 2 个以这个位置为结尾的 Primitive Squares。$\blacksquare$

#### Theorem 5.2.4 Number of Squares and Different Squares

> 本质不同的 Squares 的数量级为 $\mathcal O(n\log n)$
> Squares 的数量级为 $\mathcal O(n^2)$

考虑在 Runs 上找本质不同的 Squares。对于一个 Runs 的三元组 $[l,r,p]$，本质不同的 Squares 为集合 $\{S[i,i+len-1] : l\le i\le l+p-1, len = 2kp, i+len-1\le r\}$。可以看出该集合元素个数不超过 $r-l+2-2p$，且不少于 $\frac{r-l+2-2p}{2}$。根据 **Theorem 5.2.2 Number of Primitive Squares** 的推论，本质不同的 Squares 的数量级为 $\mathcal O(n\log n)$

而 Squares 的数量级，一个显然的证明是构造一个 $|\Sigma^{*}|=1$ 的字符串

### 5.3 Implementation

首先我们可以想到一种十分显然的方法，因为数量级为 $\mathcal O(n \log n)$，而 Primitive Squares 一定是 Runs 的一部分，所以所以我们可以在每一个 Runs 上面暴力求是否有 Primitive Squares。时间复杂度 $\mathcal O(n \log n)$。

我们可以构造出一个高周期性的字符串，使得其 Primitive Squares 的数量级为  $\mathcal O(n \log n)$ 。定义 **Fibonacci 字符串** 如下：

$$\begin{aligned}t_0&=a,\\
t_1&=b\\
t_i&=t_{i-1}+t_{i-2}\end{aligned}$$

其 Primitive Squares 的数量为 $\mathcal O(n \log n)$。

若仅需 Squares，另一种求法是枚举平方串的长度。设当前枚举的平方串长度为 $2d$，我们将原串分割成 $S[1,d], S[d+1,2d], \cdots S[(\lceil\frac{|S|}{d}\rceil-1) d+1,|S|]$

于是，我们发现长度为 $2d$ 的平方串一定会经过一个区间。枚举区间，与 Runs 类似的方法  lcp 和 lcs 即可。如果用 SA + RMQ 维护，时间复杂度 $\mathcal O(n \log n)$；二分哈希复杂度为 $\mathcal O(n \log^2 n)$

还有一种冷门算法可以求解，**Main–Lorentz 算法**。其思想为分治，我们将串 $S$ 分解成左右两部分 $u=S[1,\lfloor\frac{|S|}{2}\rfloor]$ 和 $v=S[\lfloor\frac{|S|}{2}\rfloor+1,|S|]$。然后，我们仅需计算起始位置在左部，终止在右部的 Squares。

枚举 Squares $S[i,j]$ 的中间字符 $S[\frac{i+j+1}{2}]$ 在 $u$ 还是在 $v$。 若中间字符在 $u$，则 Squares 一定存在一个周期，其开始于 $u$ 内的字符 $u[k]$，终止于 $u[|u|]$。枚举 $u[k]=v[1]$ 的所有 $k$，与 Runs 类似的方法，利用 Z 函数求 =lcp 和 lcs，判断是否存在 Squares 即可。中间字符在 $v$ 同理。

时间复杂度 $\mathcal O(n \log n)$。

### Ex I [集训队作业2018] 串串划分

> 给你个串 $S\ (|S| \le 10^5)$，切成若干段，相邻两个串是不相等的，且每一段 $s$ 不为循环串，即不存在一个串 $u$，使得 $s=u^k$。求方案数模 $998244353$。

首先考虑 dp。$f_i$ 表示 $\operatorname{suf}(i)$ 的划分方案数。可以得到这么一个柿子：（当然，如果考虑划分前缀也是可以的）

$$f_i = \sum _{j<i}f_j \cdot (-1)^{\sigma(j\cdots i) - 1} $$

其中，$\sigma(s)$ 是这样定义的：对于字符串 $s$，他的最小循环节的循环的长度为 $\sigma(s)$。

比如，对于串 $\color{blue}\text{abcabcabc}\color{red}\text{defg}$，他的 $\sigma = 3$，因为 $\color{blue}\text{abc}$ 为最小循环节，循环了 $3$ 次。

这个复杂度是 $\mathcal O(n^2)$，无法接受。

考虑优化：把循环串分成一个初始的循环串和一些循环，分开计算贡献。

我们认为一个循环串初始的时候的最小周期一定是大于等于长度的一半的（否则会重复的）。

$$f_i = \left[\sum_{j<i}f_j\right] + \sum _{j<i} f_j\left[(-1)^{\sigma(j\cdots i)-1}-1\right] $$

然于是我们把 Primitive squares 和 runs 求出来转移就好了。时间复杂度为 $\mathcal O(n \log n)$。

### Ex J [ZJOI2020] 字符串

> 求串 $S$ 的区间本质不同 Squares 数量。$|S|,q\le 2\times 10^5$


