
# 对拍

## Windows 下对拍

对拍程序

```cpp
int main() {
	while (1) {
		system("data.exe");
		system("A.exe");
		system("std.exe");
		if (system("fc A.txt std.txt")) break;
	}
} 
```

造数据的程序（运行后生成 data.exe）

```cpp
int main() {
	srand(time(0));
	freopen("in.txt","w",stdout);
	// 正常造数据
}
```

错误的程序 $A$（运行后生成 $A$.exe）

```cpp
int main() {
	freopen("in.txt","r",stdin);
	freopen("A.txt","w",stdout);
	// 正常程序
}
```

正解或暴力程序（运行后生成 std.exe）

```cpp
int main() {
	freopen("in.txt","r",stdin);
	freopen("std.txt","w",stdout);
	// 正常程序
}
```


## Linux 下对拍

对拍程序（其余程序与 windows 下相同处理）

```cpp
int main() {
	for (int i=1;;i++) {
		printf("The result of No. %d Case is:");
		system("./data");
		system("./A");
		system("./std");
		if (system("diff A.txt std.txt")) {
			printf("Wrong Answer\n"); return 0;
		}
		else printf("Accepted\n");
	}
}
```




> Written with [StackEdit中文版](https://stackedit.cn/).
