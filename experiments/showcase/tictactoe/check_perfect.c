#define main recorded_native_main
#include "main.c"
#undef main
static unsigned games,losses;
static void explore(int turn){
 int win=winner();
 if(win){games++;losses+=win==1;return;}
 int empty=0;for(int i=0;i<9;i++)empty+=board[i]==0;
 if(!empty){games++;return;}
 if(turn==1){
  for(int i=0;i<9;i++)if(!board[i]){board[i]=1;explore(2);board[i]=0;}
 }else{
  int move=-1,best=-2;
  for(int i=0;i<9;i++)if(!board[i]){board[i]=2;int v=score(1,-2,2);board[i]=0;if(v>best){best=v;move=i;}}
  board[move]=2;explore(1);board[move]=0;
 }
}
int main(void){explore(1);printf("%u games, %u human wins\n",games,losses);return losses?1:0;}
