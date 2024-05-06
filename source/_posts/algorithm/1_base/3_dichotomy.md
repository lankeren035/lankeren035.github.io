---
title: 1-3 二分

date: 2024-4-18

tags: [算法，排序]

categories: [算法]

comment: false

toc: true


---

#
<!--more-->

# 3 整数二分

## 3.1思想

- 二分的本质并不是单调性：有单调性一定可以二分，可以二分不一定有单调性。

- 如果可以找到一个性质，将区间分为一边满足，一边不满足，那么二分就可以寻找这个性质的边界。

- 每次分一半，保证答案处在目标区间里。二分一定会有解。

## 3.2 模板

```c++
//目标在右区间。区间[l, r]被划分成[l, mid], [mid+1, r]
int bsearch_1(int l, int r){
    while(l < r){
        int mid = ((r-l)>>1)+l; //多一个+1
        if(check(mid)) r=mid;  //check()查看mid是否符合性质
        else l = mid + 1;
    }
    return l;
}

//目标在左区间。区间[l, r]被划分成[l, mid-1]，[mid, r]
int bsearch_2(int l, int r){
    while(l<r){
        int mid = ((r-l+1)>>1)+l;
        if (check(mid)) l=mid;
        else r=mid-1;
    }
    return l;
}
```



## 3.3 例题

- 给定一长度为n的升序数组，以及q个查询。对于每次查询，输入一个数k，返回元素k的起始位置和终止位置（从0开始）。不存在则返回-1 -1

```c++
//测试
6 3
1 2 2 3 3 4
3
4
5
    
//输出
3 4
5 5
-1 -1
```

```c++
#include<iostream>
using namespace std;
int a[100010];
int n=0,m=0;

int main(){
        scanf("%d%d",&n,&m);
        for(int i=0; i<n; i++) scanf("%d",&a[i]);
        while(m--){
                int x;
                scanf("%d", &x);

                int l=0, r=n-1;
                while(l<r){
                        int mid = ((r-l)>>1)+l;
                        if (a[mid] >= x) r=mid;
                        else l=mid+1;
                }
                if (a[l] != x) cout<<"-1 -1"<<endl;
                else{
                        cout <<l <<' ';

                        int l =0, r=n-1;
                        while(l<r){
                                int mid = ((r-l+1)>>1)+l;
                                if (a[mid] <=x) l=mid;
                                else r=mid-1;
                        }
                        cout<<l<<endl;
                }
        }
        return 0;
}
```

# 4 浮点数二分

- 更简单，不用考虑+1问题。

## 4.1 例题

- 计算一个数（>1）的平方根

```c++
#include<iostream>
using namespace std;

int main(){
        double x=0;
        cin >>x;
        double l=0, r=x;
        while(r-l>1e-8){
                double mid = ((r-l)/2)+l;
                if(mid*mid<x) l=mid;
                else r=mid;
        }
        cout<<l<<endl;
        return 0;
}
```

