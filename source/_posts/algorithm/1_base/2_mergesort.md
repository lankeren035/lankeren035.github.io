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

## 2 归并排序

### 2.1 思想

```
1. 确定分界点 a[(l+r)/2]
2. 递归排序左右
3. 归并，合二为一（*）
```

### 2.2 算法模板
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

