---
title: 1. 快速排序

date: 2023-10-02

tags: [算法，排序]

categories: [算法]

comment: false

toc: true
---
#
<!--more-->

# 1 快速排序

### 1. 1 思想

```
1. 确定分界点
	- q[l]
	- q[r]
	- q[(l+r)/2]
	- q[random]
2. 调整区间（*）
	- 左边<=x x0 右边>=x
	- x不一定在x0位置
3. 递归两端
```
```
1. 暴力求解
	a[], b[], c[],对c进行排序，<=x的放到a，>=x的放到b。放完后先将a放到c，再将b放到c.
	
2. 优化空间
	定义左右指针i,j当c[i]>x时停，当c[j]<x时停。两个都停之后交换。
```
### 1.2 算法模板
```c
#include<iostream>
using namespace std;
int a[100000];
void qsort(int a[],int l, int r){
    if(l>=r) return;
    int i=l-1,j=r+1;
    int x=a[((r-l)>>1)+l];
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



- 可以考虑使用i或j
- 可以考虑使用while(a[i++])




