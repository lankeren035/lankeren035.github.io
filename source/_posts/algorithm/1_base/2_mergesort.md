---
title: 1-2 归并排序

date: 2023-10-02

tags: [算法，排序]

categories: [算法]

comment: false

toc: true

---

#  

<!--more-->

# 2. 归并排序

## 2.1 思想


1. 确定分界点 a[(l+r)/2]
2. 递归排序左右
3. 归并，合二为一（*）


## 2.2 算法模板
```c
#include<iostream>

using namespace std;

int a[100010];
int temp[100010];
int n=0;

void msort(int a[],int l,int r){
    if(l>=r) return;
    int mid=((r-l)>>1)+l;
    msort(a,l,mid);
    msort(a,mid+1,r);
    
    //merge
    int i=l,j=mid+1,k=0;
    while(i<=mid && j<=r){
        temp[k++]=(a[i]<=a[j]?a[i++]:a[j++]);
    }
    while(i<=mid) temp[k++]=a[i++];
    while(j<=r) temp[k++]=a[j++];
    //
    for(i=l,j=0;i<=r;i++,j++) a[i]=temp[j];
}

int main(){
    scanf("%d",&n);
    for(int i=0;i<n;i++) scanf("%d",&a[i]);
    msort(a,0,n-1);
    for(int i=0;i<n;i++) printf("%d ",a[i]);
    return 0;
}
```

```c++
//测试
5
5 4 3 2 1
```

## 2.3 练习

### 2.3.1 逆序对的数量

- 给定一个长度为 n的整数数列，请你计算数列中的逆序对的数量。(24194中2在1前面，且2>1，所以2和1构成一个逆序对)

- 输入格式：第一行包含整数 n，表示数列的长度。第二行包含 n个整数，表示整个数列。

- 输出格式：输出一个整数，表示逆序对的数量。

- 数据范围：$1≤n≤100000$，$1≤数列中元素的值≤10^9$

- 输入样例：
    ```
    6
    2 3 4 5 6 1
    ```

- 输出样例：
    ```
    5
    ```

#### 解1

- 基于归并排序的解法，对于一个逆序对ab分为三种情况：

    - ab都在左边

    - ab都在右边

    - a在左边b在右边

1. 确定分界点 a[(l+r)/2]
2. 递归排序左右
3. 左+右+左右(当a[i]>a[j]时，a[i]后面的数都大于a[j])

```c++
#include<iostream>
using namespace std;

typedef long long ll; //如果全部逆序，最多有5*10^9个，会爆int 

int a[100000];
int temp[100000];

ll merge_sort(int l, int r){
	if (l>=r) return 0;
	int mid = ((r-l)>>1)+l;
	ll res = merge_sort(l,mid) + merge_sort(mid+1,r);
	
	//merge
	int i=l,j=mid+1,k=0;
	while(i<=mid && j<=r){
		if (a[i]>a[j]){
			temp[k++] = a[j++];
			res += (mid-i+1); //重点，左区间中a[i]后面的数都大于a[j] 
		}
		else{
			temp[k++] = a[i++];
		}
	}
	while(i<=mid) temp[k++] = a[i++];
	while(j<=r) temp[k++] = a[j++];
	for(int i=0;l<=r;i++) a[l++]=temp[i];
	return res;
}

int main(){
	int n;
	cin>>n;
	for(int i=0;i<n;i++) cin>>a[i];
	cout<<merge_sort(0,n-1);
	return 0;
}
```