#include "platform.h"
#include "moves.h"
static unsigned char board[9];
static const unsigned char lines[8][3]={{0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};
static int winner(void){for(int i=0;i<8;i++){int a=board[lines[i][0]];if(a&&a==board[lines[i][1]]&&a==board[lines[i][2]])return a;}return 0;}
static int score(int turn,int alpha,int beta){
 int win=winner();if(win)return win==2?1:-1;
 int best=turn==2?-2:2,any=0;
 for(int i=0;i<9;i++)if(!board[i]){
  any=1;board[i]=(unsigned char)turn;int v=score(3-turn,alpha,beta);board[i]=0;
  if(turn==2){if(v>best)best=v;if(best>alpha)alpha=best;}
  else{if(v<best)best=v;if(best<beta)beta=best;}
  if(alpha>=beta)break;
 }
 return any?best:0;
}
static void show(void){for(int i=0;i<9;i++){byte(board[i]?".XO"[board[i]]:'.');if(i%3==2)byte('\n');}byte('\n');}
static int host_move(unsigned *cursor){
#ifdef LIVE_NATIVE
 int c;do{c=getchar();}while(c==' '||c=='\n'||c=='\r'||c=='\t');
 return c>='0'&&c<='8'?c-'0':-1;
#else
 return *cursor<MOVE_COUNT?moves[(*cursor)++]:-1;
#endif
}
int engine_main(void){
 unsigned cursor=0;text("Human X, CPU O; squares 0..8\n");show();
 for(int ply=0;ply<9;ply++){
  int move=-1;
  if(ply%2==0){do{move=host_move(&cursor);if(move<0){text("Input exhausted\n");return 1;}}while(board[move]);text("Human ");}
  else{
   int best=-2;for(int i=0;i<9;i++)if(!board[i]){board[i]=2;int v=score(1,-2,2);board[i]=0;if(v>best){best=v;move=i;}}
   text("CPU ");
  }
  number((unsigned)move);byte('\n');board[move]=(unsigned char)(ply%2?2:1);show();
  int win=winner();if(win){text(win==2?"CPU wins\n":"Human wins\n");return 0;}
 }
 text("Draw\n");return 0;
}
