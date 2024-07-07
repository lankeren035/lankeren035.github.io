---
title: 1-4. 高精度

date: 2024-5-31

tags: [算法，高精度]

categories: [算法]

comment: false

toc: true

---

#  

<!--more-->

# 4. 高精度

- 通常c++才需要高精度，java、python不需要。

- 高精度在面试中不常考，笔试偶尔出现。

- 通常又四种类型：假设A是大数，a是小数：

  - A + B
  - A - B
  - A * a
  - A / a
  - A * B偶尔会考

- c++中对于大整数，将每一位存到数组里：123456789

  - | a[   | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8]   |
    | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
    |      | 9    | 8    | 7    | 6    | 5    | 4    | 3    | 2    | 1    |




## 4.1 加法

- 给定两个正整数（不含前导 0），计算它们的和。

- 输入格式：共两行，每行包含一个整数。

- 输出格式：共一行，包含所求的和。

- 数据范围：1≤整数长度≤100000

### 4.1.1 思想



### 4.1.2 模板

```c++
#include<iostream>
#include<vector>


using namespace std;

vector<int> A,B;

vector<int> add(vector<int> &A, vector<int> &B){
	vector<int> C;
	int t=0; //进位
	for(int i=0;i<A.size() || i<B.size();i++) {
		if(i<A.size()) t+=A[i];
		if(i<B.size()) t+=B[i];
		C.push_back(t%10);
		t/=10;
	}
	if(t) C.push_back(1);
	return C;
}

int main(){
	string a,b;
	cin>>a>>b; //123

	for(int i=a.size()-1;i>=0;i--) A.push_back(a[i]-'0'); //[321]
	for(int i=b.size()-1;i>=0;i--) B.push_back(b[i]-'0');
	
	auto C = add(A,B); //auto让编译器自动识别类型 
	
	for(int i=C.size()-1;i>=0;i--) cout<<C[i]; 
	return 0;
} 
```





## 4.2 减法

- 给定两个正整数（不含前导 0），计算它们的差，计算结果可能为负数。

- 输入格式：共两行，每行包含一个整数。

- 输出格式：共一行，包含所求的差。

- 数据范围：1≤整数长度≤$10^5$

### 4.2.1 思想



### 4.2.2 模板

```c++
#include<iostream>
#include<vector> 
using namespace std;

int cmp(vector<int> &A, vector<int> &B){ // A >= B
	if (A.size() == B.size()){
		for(int i=A.size()-1; i>=0; i--)
			if (A[i] != B[i]) return A[i] > B[i];
	}
	return A.size() >= B.size();	
}

vector<int> sub(vector<int> &A, vector<int> &B){ //A >= B
	vector<int> C;
	for(int i=0 , t=0; i<A.size(); i++){ //t是借位 
		t = A[i] - t;
		if (i<B.size()) t-=B[i]; //B[i]存在
		C.push_back((t+10) % 10);
		if (t<0) t=1;
		else t=0; 
	} 
	//除去前导0
	while(C.size()>1 && C.back() ==0) C.pop_back();
	return C;
}

int main(){
	vector<int> A,B,C;
	string a,b;
	cin>>a>>b;
	for(int i=a.size()-1;i>=0;i--) A.push_back(a[i]-'0');
	for(int i=b.size()-1;i>=0;i--) B.push_back(b[i]-'0');
	
	if(cmp(A,B)) //A>=B 
		C=sub(A,B);
	else{
		C=sub(B,A);
		cout<<'-'; 
	}
	
	for(int i=C.size()-1 ;i>=0 ;i-- ) cout<<C[i]; 
	return 0;
}
```



## 4.3 乘法

-  给定两个非负整数（不含前导 0） A和 B，请你计算 A×B的值。 
- 输入格式： 共两行，第一行包含整数 A，第二行包含整数 B。 
- 输出格式： 共一行，包含 A×B的值。 
- 数据范围：
  -  1≤A的长度≤100000
  -  0≤B≤10000


### 4.3.1 思想



### 4.3.2 模板

```c++
#include<iostream>
#include<vector> 
using namespace std;

vector<int> mul(vector<int> &A, int b){
	vector<int> C;
	
	for(int i=0 , t=0; i<A.size() || t; i++){ //t是进位
		if(i<A.size()) t += A[i]*b; //A[i]存在 
		C.push_back(t % 10);
		t /= 10;
	} 

    while (C.size() > 1 && C.back() == 0) C.pop_back();//前导零
	return C;
}

int main(){
	vector<int> A,C;
	int b=0;
	string a;
	cin>>a>>b;
	for(int i=a.size()-1;i>=0;i--) A.push_back(a[i]-'0');
	C=mul(A,b);
	for(int i=C.size()-1 ;i>=0 ;i-- ) cout<<C[i]; 
	return 0;
}
```



## 4.4 除法

- 给定两个非负整数（不含前导 0） A，B请你计算 A/B的商和余数。

-  输入格式：共两行，第一行包含整数 A，第二行包含整数 B。

- 输出格式：共两行，第一行输出所求的商，第二行输出所求余数。

- 数据范围
  - 1≤A的长度≤100000
  - 1≤B≤10000
  - B 一定不为 0

### 4.4.1 思想



### 4.4.2 模板

```c++
#include<iostream>
#include<vector> 
#include<algorithm>
using namespace std;

vector<int> div(vector<int> &A, int b, int &r){
	vector<int> C;
	
	for(int i=A.size()-1 ; i>=0 ; i--){ //t是被除数
		r = r*10+A[i];
		C.push_back(r/b);
		r %= b;
	} 
	
	reverse(C.begin(),C.end()); //低位在前
    while (C.size() > 1 && C.back() == 0) C.pop_back();//前导零
	return C;
}

int main(){
	vector<int> A,C;
	int b=0,r=0; //r是余数
	string a;
	cin>>a>>b;
	for(int i=a.size()-1;i>=0;i--) A.push_back(a[i]-'0');
	C=div(A,b,r);
	for(int i=C.size()-1 ;i>=0 ;i-- ) cout<<C[i]; 
	cout<<endl<<r;
	return 0;
}
```

