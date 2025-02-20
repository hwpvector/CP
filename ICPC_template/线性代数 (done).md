# 线性代数

## 矩阵乘法

注意枚举顺序，内存接近。

```cpp
for (int i=1;i<=n;++i)
    for (int k=1;k<=n;++k)
        for (int j=1;j<=n;++j)
            p[i][j]+=a[i][k]*b[k][j];
```

## 线性基

```cpp
int a[61],tmp[61],flag;
void ins(int x) {
    for (int i=60;~i;i--) if (x&(1ll<<i))
        if (!a[i]) {a[i]=x;return;}
        else x^=a[i];
    flag=true;
}
bool check(int x) {
    for (int i=60;~i;i--) if (x&(1ll<<i))
        if (!a[i]) return false;
        else x^=a[i];
    return true;
}
int qmax(int res=0) {
    for (int i=60;~i;i--) res=max(res,res^a[i]);
    return res;
}
int qmin(){
    if (flag) return 0;
    for (int i=0;i<=60;i++)
        if (a[i]) return a[i];
}
int query(int k){
    int res=0,cnt=0;
    k-=flag; if (!k) return 0;
    for (int i=0;i<=60;i++){
        for (int j=i-1;~j;j--)
            if (a[i]&(1ll<<j)) a[i]^=a[j];
        if (a[i]) tmp[cnt++]=a[i];
    }
    if (k>=(1<<cnt)) return -1;
    for (int i=0;i<cnt;i++)
        if (k&(1<<i)) res^=tmp[i];
    return res;
}
```


## 高斯消元

```cpp
const int N=1e5;
constexpr double eps=1e-6;
double a[N][N],x[N];

int solve(int n, int m) {
    int c=0,r;
    for(r=0;r<n&&c<m;r++,c++){
        int maxr=r;
        for(int i=r+1;i<n;i++)
            if(abs(a[i][c])>abs(a[maxr][c]))
                maxr=i;
        if(maxr!=r) swap(a[r], a[maxr]);
        if(fabs(a[r][c])<eps){
            r--; continue;
        }
        for(int i=r+1;i<n;i++){
            if(fabs(a[i][c])>eps){
                double k=a[i][c]/a[r][c];
                for(int j=c;j<m+1;j++) a[i][j]-=a[r][j]*k;
                a[i][c]=0;
            }
        }
    } 
    for(int i=r;i<m;i++){
        if(fabs(a[i][c])>eps) return -1; //无解
    }    
    if(r<m) return m-r; //返回自由元个数
    for(int i=m-1;i>=0;i--){
        for(int j=i+1;j<m;j++) a[i][m]-=a[i][j]*x[j];
        x[i]=a[i][m]/a[i][i];
    }
    return 0; //有唯一解
}
int main() {
    int n;
    cin>>n;
    for(int i=0;i<n;i++)
        for(int j=0;j<=n;j++) 
            cin>>a[i][j];
    int pan=solve(n, n);
    if(pan!=0) { cout<<"No Solution";return 0;}
    for(int i=0;i<n;i++)
        printf("%.2lf\n", x[i]);
}
```

> Written with [StackEdit中文版](https://stackedit.cn/).
