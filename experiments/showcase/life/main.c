#include "platform.h"
#define W 24
#define H 16
static unsigned char board[H][W],next[H][W];
int engine_main(void){
 board[1][2]=board[2][3]=board[3][1]=board[3][2]=board[3][3]=1;
 board[9][14]=board[9][15]=board[9][16]=1;
 board[5][19]=board[5][20]=board[6][19]=board[6][20]=1;
 for(unsigned gen=0;gen<5;gen++){
  text("Generation ");number(gen);byte('\n');
  for(int y=0;y<H;y++){for(int x=0;x<W;x++)byte(board[y][x]?'#':'.');byte('\n');}
  for(int y=0;y<H;y++)for(int x=0;x<W;x++){
   unsigned n=0;
   for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++)if(dx||dy)n+=board[(y+dy+H)%H][(x+dx+W)%W];
   next[y][x]=(n==3)||(board[y][x]&&n==2);
  }
  for(int y=0;y<H;y++)for(int x=0;x<W;x++)board[y][x]=next[y][x];
 }
 return 0;
}
