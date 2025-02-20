
# 随机算法

基于随机数据或基于随机收敛局部最优的算法

## 珂朵莉树

通过维护值相同的连续段来保证效率，在特殊构造的数据下会退化为普通暴力算法。

```cpp
struct node {
    int l,r; mutable int v;
    node(int l,int r,int v):l(l),r(r),v(v) {}
    bool operator<(const node &o) const {return l<o.l;}
};
set<node> t;

// 将区间 [l,r] 分割成 [l,pos-1] 和 [pos,r]
// 返回的是后面 [pos,r] 区间的迭代器
auto split(int pos) {
    auto it=t.lower_bound(node(pos,0,0));
    if (it!=t.end()&&it->l==pos) return it;
    it--; int l=it->l,r=it->r,v=it->v;
    t.erase(it); t.insert(node(l,pos-1,v));
    return t.insert(node(pos,r,v)).first;
}
// 将区间 [l,r] 覆盖成 v
void assign(int l,int r,int v) {
    auto end=split(r+1),begin=split(l);
    t.erase(begin,end); t.insert(node(l,r,v));
}
void add(int l,int r,int v) {
    auto end=split(r+1);
    for (auto it=split(l);it!=end;it++) it->v+=v;
}
int kth(int l,int r,int k) {
    auto end=split(r+1);
    vector<pii> v;
    for (auto it=split(l);it!=end;it++)
        v.pb(mk(it->v,it->r-it->l+1));
    sort(v.begin(),v.end());
    for (auto [val,siz]:v) {
        k-=siz;
        if (k<=0) return val;
    }
}
```

## 模拟退火

可以把整个值域分成几段，每段跑一遍模拟退火，然后再取最优解。

```cpp
double Rand() {return (double)rand()/RAND_MAX; }
void solve() {
	double t=100000; // 初始温度
	double now; // 当前解，初状态
	while (t>0.001) {
		double nxt=now+t*(Rand()*2-1); // 在当前解附近 [-t,t] 产生新解
		double delta=calc(nxt)-calc(now); // 计算两个解的差距
		if (exp(-delta/t)>Rand()) now=nxt;
		t*=0.99;
	}
}
```

<div STYLE="page-break-after: always;"></div>

## 遗传算法

```cpp
struct node { // 一个个体
	double t[12],adp; // t: 基因；adp: 适应性函数
	void getadp() {
		// 计算适应性函数，并保存到 adp 中
	}
	void init() {
		// 随机生成初始个体，以便于生成初始种群
		getadp();
	}
	bool operator < (const node &y) {return adp<y.adp;}
	bool operator > (const node &y) {return adp>y.adp;}
}d[210];
node crossover(node &x,node &y) {
    // 对两个个体进行交配，产生新个体
    node nw;
    // ...
    nw.getadp();
}
void mutation(node &x) {
    // 考察个体是否需要变异
	double rd=double(rand())/RAND_MAX;
    if (rd>0.1) return ;
    // ...
    x.getadp();
}
void GA() {
    srand(time(NULL));
    const int Siz=50,T=100000; // 种群大小、迭代次数
    for (int i=1;i<=Siz;i++) d[i].init();
    sort(d+1,d+Siz+1,greater<node>());
    while (T--) {
        int cnt=Siz;
        for (int i=1,j=2;j<=Siz;i+=2,j+=2) {
            node d[++cnt]=crossover(d[i],d[j]);
            mutation(d[cnt]);
        }
        sort(d+1,d+cnt+1,greater<node>());
    }
}
```




> Written with [StackEdit中文版](https://stackedit.cn/).
