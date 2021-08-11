#include<bits/stdc++.h>
using namespace std;
//在MobilenetV2中 max_free=4303 ,op数量5173 
const int N=6000;	
const int inf=2e9;
struct HeuristicAllocator{
	char id[20];
	int free,alloc,size;
}infos[N],optimal_infos[N];
struct Memory_address{
	int m0;
	int m1;
	bool operator<(const Memory_address&b)const{
	return m0<b.m0;
	}
};
int T=1000;
int Tmin=20;
int num;		//op个数 
int max_time;	//最后一个free的时刻 
int heuristic_address[N];
bool cmp(HeuristicAllocator a,HeuristicAllocator b)
{
	if (a.size!=b.size)
		return a.size>b.size;
	if (a.free-a.alloc!=b.free-b.alloc)
		return a.free-a.alloc>b.free-b.alloc;
	return a.alloc<b.alloc;
}
bool cmp2(Memory_address a,Memory_address b)//size更小更靠近0的好，最佳分配策略
{
	if (a.m1!=b.m1) 
	return a.m1<b.m1;
	return a.m0<b.m0;
}
void Backup()
{
	memcpy(optimal_infos,infos,sizeof(infos));
}
void Reset()
{
	memcpy(infos,optimal_infos,sizeof(infos));
}
int Random(int low,int high)	//返回[low,high)的整数 
{
	int Mod=1e4;
	int x=rand();
	x=x*rand()+rand();
	x%=Mod;
	return low+(high-low)*x/Mod;
}
void sample_change()
{
	int siz=T*num/10000;
	siz=min(siz*2,num);
	for (int i=0; i<siz; i+=2){
		int a=Random(0,num);
		int b=Random(0,num);
		swap(infos[a],infos[b]);
	}
}
bool overlap(int s1,int t1,int s2,int t2)
{
	return !(t1<=s2 || t2<=s1);
}
int heuristic_alloc()
{
	int max_address=0;
	vector <Memory_address> heuristic[N];
	//heuristic[i] 第i个时刻内存地址 [heuristic[i].m0,heuristic[i],m1)被分配 
	for (int i=0; i<max_time; ++i){
		heuristic[i].clear();
	}
	for (int j=0; j<num; ++j){
		vector <Memory_address> possible;
		possible.clear();
		//内存地址 [possible.m0,possible.m1)可能可用 
		for (int op=infos[j].alloc; op<infos[j].free; ++op){
			if (heuristic[op].size()==0){	
				possible.push_back((Memory_address){0,inf});
				continue;
			}
			if (heuristic[op][0].m0 >= infos[j].size)					//py版本中忽略了该情况 
				possible.push_back((Memory_address){0,heuristic[op][0].m0});
			for (int i=0; i<heuristic[op].size()-1; ++i)
				if (heuristic[op][i + 1].m0 - heuristic[op][i].m1 >= infos[j].size)
                	possible.push_back((Memory_address){heuristic[op][i].m1, heuristic[op][i + 1].m0 - heuristic[op][i].m1});

			possible.push_back ((Memory_address){heuristic[op][heuristic[op].size()-1].m1,inf});	
		}
		sort(possible.begin(),possible.end(),cmp2);

		bool ok=true;
		for (int i=0; i<possible.size(); ++i){

			Memory_address now=(Memory_address){possible[i].m0,possible[i].m0+infos[j].size};
			//尝试将第j个op分配在内存地址[now.m0,now.m1) 
			ok=true;
			for (int op=infos[j].alloc; op<infos[j].free; ++op){
				int pos=lower_bound(heuristic[op].begin(),heuristic[op].end(),now)-heuristic[op].begin();
				if (0<=pos-1 && pos-1<heuristic[op].size() && overlap(now.m0,now.m1,heuristic[op][pos - 1].m0, heuristic[op][pos - 1].m1)
				|| 0<=pos && pos< heuristic[op].size() && overlap(now.m0,now.m1,heuristic[op][pos].m0, heuristic[op][pos].m1)){
					ok=false;
					break;
				}
			}
			if (ok){
			//	printf("alloc  time [%d %d)  size:%d  add [%d %d)\n",infos[j].alloc,infos[j].free,infos[j].size,now.m0,now.m1); 
				for (int op=infos[j].alloc; op<infos[j].free; ++op){
					vector <Memory_address>::iterator pos=lower_bound(heuristic[op].begin(),heuristic[op].end(),now);
					heuristic[op].insert(pos,now);
				}
				heuristic_address[j]=now.m1;
				max_address=max(max_address,now.m1);
				break;
			}
			
		} 
		if (!ok) printf("error! op %d can't alloc",j);
		
	}
	printf("current: %d\n",max_address);
    return max_address;
}
void SimulatedAnnealing()
{
	char filename[100]="new_data/Googlenet_4.in"; 

	freopen(filename,"r",stdin);
//	freopen("new_data/Squeezenet_4.out","w",stdout);
	while (scanf("%s%d%d%d",infos[num].id,&infos[num].alloc,&infos[num].free,&infos[num].size)!=EOF){	
		max_time=max(max_time,infos[num].free);
		num++;
	}
	sort(infos,infos+num,cmp);
	int optimal=heuristic_alloc();
	int k=5;
	int t=0;
	while (T>=Tmin){
		printf("Temperature is %.2lf at %d times;  optimal is %d; {%d} heuristics moved\n",
		T,t,optimal,T*num/10000);
		for (int i=1; i<=k; ++i){
			Backup();
			sample_change();
			int cur = heuristic_alloc();
			if (cur<optimal)
				 optimal=cur;
			else Reset();	
		}
		t+=1;
		T=1000/(1+t);
	}
	printf ("optimal memory is %d\n",optimal);
}

int main()
{
	srand(time(0));
//	test();
	SimulatedAnnealing();
}
