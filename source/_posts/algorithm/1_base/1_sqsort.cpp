#include<iostream>
using namespace std;
int a[100000];
void qsort(int a[], int l, int r){
	if(l>=r) return;
	int i=l-1,j=r+1;
	int x=a[((r-l)>>1)+l];
	while(i<j){
		do i++; while(a[i]<x);
		do j--; while(a[j]>x);
		if(i<j) swap(a[i],a[j]);
	}
	qsort(a, l, j);
	qsort(a,j+1,r);
}
int main(){
	int n=0;
	scanf("%d",&n);
	for(int i=0; i<n; i++) scanf("%d", &a[i]);
	qsort(a,0,n-1);
	for(int i=0;i<n;i++) printf("%d",a[i]);
}
