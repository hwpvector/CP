

# 数值算法

## 拉格朗日插值

横坐标不连续，复杂度为 $O(n^2)$

```cpp
vector<int> lagrange_interpolation(vector<int> x, vector<int> y) {
	const int n = x.size();
	vector<int> M(n + 1), xx(n), f(n);
	M[0] = 1;
	for (int i = 0; i < n; ++i) {
		for (int j = i; j >= 0; --j) {
			M[j + 1] = (M[j] + M[j + 1]) % MOD;
			M[j] = (LL)M[j] * (MOD - x[i]) % MOD;
		}
	}
	for (int i = n - 1; i >= 0; --i) {
		for (int j = 0; j < n; ++j) {
			xx[j] = ((LL)xx[j] * x[j] + (LL)M[i + 1] * (i + 1)) % MOD;
		}
	}
	for (int i = 0; i < n; ++i) {
		LL t = (LL)y[i] * inv(xx[i]) % MOD, k = M[n];
		for (int j = n - 1; j >= 0; --j) {
			f[j] = (f[j] + k * t) % MOD;
			k = (M[j] + k * x[i]) % MOD;
		}
	}
	return f;
}
int main() {
	int n, k;
	cin >> n >> k;
	vector<int> x(n), y(n);
	for (int i = 0; i < n; ++i) cin >> x[i] >> y[i];
	const auto f = lagrange_interpolation(x, y);
	int v = 0;
	for (int i = n - 1; i >= 0; --i) v = ((LL)v * k + f[i]) % MOD;
	cout << v << '\n';
	return 0;
}
```

横坐标连续，复杂度为 $O(n)$

横坐标为 $1,\cdots,n+1$ 的插值公式：

$$f(x)=\sum\limits_{i=1}^{n+1}(-1)^{n+1-i}y_i\cdot\frac{\prod\limits_{j=1}^{n+1}(x-j)}{(i-1)!(n+1-i)!(x-i)}$$

```cpp
int main() {
	scanf("%d%d", &n, &k);
	// 得到 f(i) 在 i=1,2,...,k 的值
	// 下面程序求 f(n) 的值
	if (n <= k) return printf("%d\n", f[n]) & 0;
	pre[0] = suf[k + 1] = 1;
	for (int i = 1; i <= k; i++) pre[i] = 1LL * pre[i - 1] * (n - i) % mod;
	for (int i = k; i >= 1; i--) suf[i] = 1LL * suf[i + 1] * (n - i) % mod;
	fac[0] = inv[0] = fac[1] = inv[1] = 1;
	for (int i = 2; i <= k; i++) {
		fac[i] = 1LL * fac[i - 1] * i % mod;
		inv[i] = 1LL * (mod - mod / i) * inv[mod % i] % mod;
	}
	for (int i = 2; i <= k; i++) inv[i] = 1LL * inv[i - 1] * inv[i] % mod;
	for (int i = 1; i <= k; i++) {
		int P = 1LL * pre[i - 1] * suf[i + 1] % mod;
		int Q = 1LL * inv[i - 1] * inv[k - i] % mod;
		int mul = ((k - i) & 1) ? -1 : 1;
		ans = (ans + 1LL * (Q * mul + mod) % mod * P % mod * f[i] % mod) % mod;
	}
	printf("%d\n", ans);
	return 0;
}
```

## 牛顿插值

```cpp
template <class T>
struct NewtonInterp {
	vector<pair<T, T>> p;
	vector<vector<T>> dy;
	vector<T> base, poly;
	void insert(T const &x, T const &y) {
		p.emplace_back(x, y);
		size_t n = p.size();
		if (n == 1) {
			base.push_back(1);
		} else {
			size_t m = base.size();
			base.push_back(0);
			for (size_t i = m; i; --i) base[i] = base[i - 1];
			base[0] = 0;
			for (size_t i = 0; i < m; ++i)
				base[i] = base[i] - p[n - 2].first * base[i + 1];
		}
		dy.emplace_back(p.size());
		dy[n - 1][n - 1] = y;
		if (n > 1) {
		for (size_t i = n - 2; ~i; --i)
			dy[n - 1][i] = (dy[n - 2][i] - dy[n - 1][i + 1]) / (p[i].first - p[n - 1].first);
		}
		poly.push_back(0);
		for (size_t i = 0; i < n; ++i) poly[i] = poly[i] + dy[n - 1][0] * base[i];
	}

	T eval(T const &x) {
		T ans{};
		for (auto it = poly.rbegin(); it != poly.rend(); ++it) ans = ans * x + *it;
		return ans;
	}
};
int main() {
	NewtonInterp<int> ip;
	int n, k;
	cin >> n >> k;
	for (int i = 1, x, y; i <= n; ++i) {
		cin >> x >> y;
		ip.insert(x, y);
	}
	cout << ip.eval(k);
	return 0;
}
```

## 数值积分

```cpp
double f(double x) {
	// ...
}
double simpson(double l, double r) {
	double mid = (l + r) / 2;
	return (r - l) * (f(l) + 4 * f(mid) + f(r)) / 6;
}
double asr(double l, double r, double eps, double ans, int step) {
  	double mid = (l + r) / 2;
	double fl = simpson(l, mid), fr = simpson(mid, r);
	if (abs(fl + fr - ans) <= 15 * eps && step < 0)
		return fl + fr + (fl + fr - ans) / 15;
	return asr(l, mid, eps / 2, fl, step - 1) + asr(mid, r, eps / 2, fr, step - 1);
}
// 调用该函数，求解 [l,r] 上对 f(x) 的积分，误差 eps
double calc(double l, double r, double eps) {
  	return asr(l, r, eps, simpson(l, r), 12);
}
```


## 牛顿迭代

对于在 $[a,b]$ 上连续且单调的函数 $f(x)$，求方程 $f(x)=0$ 的近似解。

根据递推式：$x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}$，不断递推，递推速度在 $\log \log n$

## 拉格朗日乘数法

求在 $g(x_i)=0$ 的限制下，$f(x_i)$ 的极值

令 $F(x_i,\lambda)=f(x_i)+\lambda g(x_i)$

该函数对任意 $x_i,\lambda$ 求偏导为 $0$

通常 $\lambda$ 和上述偏导式具有一定的单调关系，可以二分


> Written with [StackEdit中文版](https://stackedit.cn/).
