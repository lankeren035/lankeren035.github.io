---
title: 1-1. 快速排序

date: 2023-10-02

tags: [算法，排序]

categories: [算法]

comment: false

toc: true
---
#

<!--more-->

# 1. 快速排序

## 1. 1 思想


1. 确定分界点
	- q[l] （用j）
	- q[r] （用i）
	- q[(l+r)/2]
	- q[random]
2. 调整区间（*）
	- 左边<=x x0 右边>=x
	- x不一定在x0位置
3. 递归两端

```
1. 暴力求解
	a[], b[], c[],对c进行排序，<=x的放到a，>=x的放到b。放完后先将a放到c，再将b放到c.
	
2. 优化空间
	定义左右指针i,j当c[i]>x时停，当c[j]<x时停。两个都停之后交换。
```
## 1.2 算法模板
```c
#include<iostream>
using namespace std;
int a[100000];
void qsort(int a[],int l, int r){
    if(l>=r) return;
    int i=l-1,j=r+1;
    int x=a[((r-l)>>1)+l]; //这里要用a[]不要只计算索引，因为后续该索引位置上的元素可能会改变。
    while(i<j){
        do i++; while(a[i]<x);
        do j--; while(a[j]>x);
        if(i<j) swap(a[i],a[j]);
    }
    qsort(a,l,j);
    qsort(a,j+1,r);
}

int main(){
    int n;
    scanf("%d",&n);
    for(int i=0;i<n;i++) scanf("%d",&a[i]);
    qsort(a,0,n-1);
    for(int i=0;i<n;i++) printf("%d ",a[i]);
}
```
```
//输入
5
5 4 3 2 1
```



- c++里的快排是快排与插排的组合。

- 快排一般做题不会遇到，在面试的时候要你手写。
- 当分界点选择为a[l]时，如果使用i分割左右，那么可能会死循环，例如[0,1]用i分割后左：[]，右：[0,1]死循环。所以`mid = a[l]`时，递归用`(a, l, j)`。反之同理。

- 可以考虑使用i或j
- 可以考虑使用while(a[i++])

- 快排不稳定，如果要变成稳定的，可以变成<a[i], i>
- 循环最终只会有两种情况：
    - i==j
    - i==j+1


## 1.3 练习

### 1.3.1 第k个数

- 给定一个长度为 n的整数数列，以及一个整数 k，请用快速选择算法求出数列从小到大排序后的第 k个数。

- 输入格式

    - 第一行包含两个整数 n 和 k。

    - 第二行包含 n个整数（所有整数均在1~$10^9$范围内），表示整数数列。

- 输出格式

    - 输出一个整数，表示数列的第 k小数。

- 数据范围

    - 1≤n≤100000

    - 1≤k≤n

- 输入样例：
  
    ```
    5 3
    2 4 1 5 3
    ```

- 输出样例：

    ```
    3
    ```

#### 解1

- 先快排，然后输出第k个数

```c++
#include<iostream>
using namespace std;

int a[100000];

void qsort(int l, int r){
	if (l>=r) return;
	int i=l-1,j=r+1;
	int mid = a[((j-l)>>1)+i];
	while(i < j){
		while(a[++i]<mid);
		while(a[--j]>mid);
		if (i<j) swap(a[i],a[j]);
	}
	qsort(l,j);
	qsort(j+1,r);
}

int main(){
	int n=0,k=0;
	cin>>n>>k;
	for(int i=0;i<n;i++) cin>>a[i];
	qsort(0,n-1);
	cout<<a[k-1];
	return 0;
}
```

#### 解2

- 快速选择：O(n)

    1. 确定分界点
        - q[l]
        - q[r]
        - q[(l+r)/2]
        - q[random]
    2. 调整区间（*）
        - 左边<=x x0 右边>=x
        - x不一定在x0位置
    3. if 左区间长度sl >= k, 递归左区间
    4. else 左区间长度sl < k, 递归右区间(更新k)

```c++
#include<iostream>
using namespace std;

int a[100000];

int qselect(int l, int r, int k){
	if (l>=r) return a[r]; //这里可以写==，快排里不能，因为快排里面这里有可能没有数 
	int i=l-1,j=r+1;
	int mid = a[((j-l)>>1)+i];
	while(i < j){
		while(a[++i]<mid);
		while(a[--j]>mid);
		if (i<j) swap(a[i],a[j]);
	}
	if (j < k) return qselect(j+1,r,k); //左区间长度：j-l+1 
	return qselect(l,j,k); 
}

int main(){
	int n=0,k=0;
	cin>>n>>k;
	for(int i=0;i<n;i++) cin>>a[i];
	cout<<qselect(0,n-1,k-1); //传入的是下标，所以要k-1 
	return 0;
}
```
