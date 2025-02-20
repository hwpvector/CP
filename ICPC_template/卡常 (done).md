
# 卡常

## 变量运行速度

宏定义（``define``） > 常量（``constexpr``） > 只读量（``const``） = 普通变量

``unsigned`` 效率略高于 ``signed``

## switch-case

``switch-case`` 效率显著高于 ``if-else``

```cpp
switch(x) {
	case 'a': 
		//...
		break;
	case 'b':
		//...
		break;
	//...
	default:
		//...
		break;
}
```

## 快读快写

```cpp
inline int rd() {
	char ch=getchar(); int num=0,b=1;
	for (;ch>'9'||ch<'0';ch=getchar())
		if (ch=='-') b=-1;
	for (;ch<='9'&&ch>='0';ch=getchar())
		num=(num<<3)+(num<<1)+(ch^'0');
	return num*b; 
}
inline void wt(int x) { 
	if (x<0) putchar('-'),x=-x;
	if (x<10) {putchar('0'^x); return ;}
	wt(x/10); putchar((x%10)^'0');
}
```


## 取模优化

当被取模数参与加法/减法运算时，可以使用 ``while(a>=mod) a-=mod`` 和 ``while(a<0) a+=mod`` 代替

### 巴雷特约减

```cpp
struct Bart{
	int I,m;
	void ini(int Mod) {m=Mod;I=(1ll<<62)/m;}
	int operator()(int x) {
		int tp=x-((__int128)x*I>>62)*m;
		while(tp>=m) tp-=m;
		while(tp<0) tp+=m;
		return tp;
	}
}getmod;
```

## 随机算法结束时间

```cpp
void solve() {
	const double MaxTime=0.9; // 一个略小于时限的数
	while ((double)clock()/CLOCKS_PER_SEC<MaxTime) {
		// ...
	}
}
```

## 位运算

比如

```cpp
int l=1,r=n; int mid=(l+r)/2;
```

O2 编译器不会优化。负数下 ``/2`` 和 ``>>1`` 不等价。可以改成 ``unsigned``，或者位运算

常见位运算优化

- 乘上或者除一个 $2$ 的幂，使用左右移
- 判断奇偶数
- 判断大小：``b>a`` 等价于 ``((a-b)>>31)&1`` 或 ``((a-b)>>63)&1``（注意保证是 ``signed`` 类型）
- 求相反数：``(~x+1)``
- 求绝对值：``x^(~(x>>31)+1)+(x>>31)``
- 判断 $2$ 的次幂：  ``(x&(x-1))==0``

## __builtin 类

``__builtin_popcount(x)`` ``int, long long`` 变量在二进制下 $1$ 的数量

``__builtin_ctz(x)/__buitlin_ctzll(x)`` 变量在二进制下末尾 $0$ 的个数

``__builtin_ctz(x)/__buitlin_ctzll(x)`` 变量在二进制下末尾 $0$ 的个数

``__builtin_clz(x)/__buitlin_clzll(x)`` 变量在二进制下前导 $0$ 的个数

``__builtin_ffs(x)`` 变量在二进制下最后一个 $1$ 在第几位（从后往前算）

``__builtin_parity(x)`` 变量在二进制下 $1$ 的个数的奇偶性（偶数为 $0$，奇数为 $1$），快于 `__builtin_popcount(x)`

``__builtin_sqrt(x)`` 更快的开平方

## STL 效率

大量的加入元素前，统计需要的个数，进行 ``reserve`` 操作预留空间

减少不必要的 ``clear`` 后再重新申请内存，或者使用 ``pop_back`` 释放较少的空间

``emplace_back`` 效率略高于 ``push_back``

``lower_bound`` 的常数比较大，卡常可以手写二分。

``deque`` 和衍生的 ``queue`` 和 ``stack`` 常数极大

### bitset

定义 ``bitset<N> bs``

- ``bs.count()`` 返回 ``bs`` 有多少位 $1$
- ``bs._Find_first()`` 寻找第一个 $1$ 的位置
- ``bs._Find_next(i)`` 寻找下一个 $1$ 的位置
- ``bs.reset(), bs.set()`` 将整个 ``bs`` 设置为 $0$ 或 $1$
- ``bs(string), bs(unsigned long)`` 构造函数，构造为 01 串、二进制形式
-  `bs.to_string(), bs.to_ulong()` 返回对应的 01 串、``unsigned long long`` 类型变量

## 加栈空间

并不一定能过编

在头文件前加入：

```cpp
#pragmacomment(linker,"/STACK:1024000000,1024000000")
```

在主函数开始时加入：

```cpp
int __size__=256<<20; 
char *__p__=(char*)malloc(__size__)+__size__;
__asm__("movl %0,%%esp\n"::"r"(__p__));
```

## 其他

把递归写成循环或迭代的形式（比如反复的树形 dp 按 DFS 序算）

内存连续访问：``a[1]++,a[100]++`` 比 ``a[1]++,a[2]++`` 慢

冷代码将 ``if`` 改成 ``while``

把冷代码放一起，热代码放一起，方便编译器跳转

帮助编译器判断冷热代码：``if (__builtin_expect(...,0/1))``，条件极可能为 $0$ 或 $1$




## 火车头

```cpp
#pragma GCC optimize(3)
#pragma GCC target("avx")
#pragma GCC optimize("Ofast")
#pragma GCC optimize("inline")
#pragma GCC optimize("-fgcse")
#pragma GCC optimize("-fgcse-lm")
#pragma GCC optimize("-fipa-sra")
#pragma GCC optimize("-ftree-pre")
#pragma GCC optimize("-ftree-vrp")
#pragma GCC optimize("-fpeephole2")
#pragma GCC optimize("-ffast-math")
#pragma GCC optimize("-fsched-spec")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("-falign-jumps")
#pragma GCC optimize("-falign-loops")
#pragma GCC optimize("-falign-labels")
#pragma GCC optimize("-fdevirtualize")
#pragma GCC optimize("-fcaller-saves")
#pragma GCC optimize("-fcrossjumping")
#pragma GCC optimize("-fthread-jumps")
#pragma GCC optimize("-funroll-loops")
#pragma GCC optimize("-fwhole-program")
#pragma GCC optimize("-freorder-blocks")
#pragma GCC optimize("-fschedule-insns")
#pragma GCC optimize("inline-functions")
#pragma GCC optimize("-ftree-tail-merge")
#pragma GCC optimize("-fschedule-insns2")
#pragma GCC optimize("-fstrict-aliasing")
#pragma GCC optimize("-fstrict-overflow")
#pragma GCC optimize("-falign-functions")
#pragma GCC optimize("-fcse-skip-blocks")
#pragma GCC optimize("-fcse-follow-jumps")
#pragma GCC optimize("-fsched-interblock")
#pragma GCC optimize("-fpartial-inlining")
#pragma GCC optimize("no-stack-protector")
#pragma GCC optimize("-freorder-functions")
#pragma GCC optimize("-findirect-inlining")
#pragma GCC optimize("-fhoist-adjacent-loads")
#pragma GCC optimize("-frerun-cse-after-loop")
#pragma GCC optimize("inline-small-functions")
#pragma GCC optimize("-finline-small-functions")
#pragma GCC optimize("-ftree-switch-conversion")
#pragma GCC optimize("-foptimize-sibling-calls")
#pragma GCC optimize("-fexpensive-optimizations")
#pragma GCC optimize("-funsafe-loop-optimizations")
#pragma GCC optimize("inline-functions-called-once")
#pragma GCC optimize("-fdelete-null-pointer-checks")
```

> Written with [StackEdit中文版](https://stackedit.cn/).
