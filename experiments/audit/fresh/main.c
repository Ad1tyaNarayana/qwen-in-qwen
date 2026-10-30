/* Audit guest, never run before: ndot address sweep over the whole DROM and RAM, merge sort through a
   comparator pointer, CRC-32, M-extension edge cases, and code at the very top of the ROM. */
#include <stdint.h>
#include "ndot.h"
static void byte(unsigned c){register unsigned a0 __asm__("a0")=c;register unsigned a7 __asm__("a7")=1;__asm__ volatile("ecall":"+r"(a0):"r"(a7):"memory");}
static void hex(uint32_t v){for(int i=28;i>=0;i-=4){unsigned d=(v>>i)&15;byte(d<10?'0'+d:'a'+d-10);}byte('\n');}
extern const int8_t blob[]; extern const char blob_end[];
static uint32_t s=0xC0FFEE11u;
static uint32_t rnd(void){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
#define RAM_TOP 0x01010000u
static int16_t scratch[8] __attribute__((aligned(16)));
static uint32_t scalar4(uint32_t acc,const int8_t*w,const volatile int16_t*a){for(int i=0;i<4;i++)acc+=(uint32_t)((int32_t)w[i]*(int32_t)a[i]);return acc;}
static int cmp_up(int32_t a,int32_t b){return a<=b;}
static int cmp_down(int32_t a,int32_t b){return a>=b;}
static int32_t data[160],tmp[160];
static int (*const order[2])(int32_t,int32_t)={cmp_up,cmp_down};
static void msort(int32_t*a,int n,int (*f)(int32_t,int32_t)){
 if(n<2)return;int m=n/2;msort(a,m,f);msort(a+m,n-m,f);
 int i=0,j=m,k=0;while(i<m&&j<n)tmp[k++]=f(a[i],a[j])?a[i++]:a[j++];
 while(i<m)tmp[k++]=a[i++];while(j<n)tmp[k++]=a[j++];
 for(k=0;k<n;k++)a[k]=tmp[k];}
static uint32_t crc(const uint8_t*p,int n){uint32_t c=~0u;while(n--){c^=*p++;for(int i=0;i<8;i++)c=(c>>1)^(0xEDB88320u&-(c&1));}return ~c;}
__attribute__((section(".text.far"),noinline)) uint32_t far_top(uint32_t v){return (v*2654435761u)^(v>>7);}
int engine_main(void){
 uint32_t acc=0x7fffff00u,chk=acc,bad=0;
 uint32_t blobn=(uint32_t)(blob_end-(const char*)blob);
 /* ndot sweep: DROM-wide offsets, RAM-wide vectors, the last valid spans of both */
 for(int t=0;t<170;t++){
  uint32_t off=(rnd()%blobn)&~3u; if(off+4>blobn)off=blobn-4; if(t==3)off=blobn-4; if(t==4)off=0;
  uint32_t r=rnd(); uint32_t ra=0x01002000u+((r%0xD000u)&~7u); if(t==5)ra=RAM_TOP-8; if(t==6)ra=0x01000000u+0x1000;
  volatile int16_t*v=(volatile int16_t*)ra;
  uint32_t mix=rnd();
  v[0]=(int16_t)mix;v[1]=(int16_t)(mix>>16);uint32_t m2=rnd();v[2]=(t&1)?(int16_t)m2:-32768;v[3]=(int16_t)(m2>>16);
  acc=ndot4(acc,blob+off,(const int16_t*)ra);
  chk=scalar4(chk,blob+off,v);
  if(acc!=chk)bad++;
  if(t%40==0)hex(acc);
 }
 hex(acc);hex(bad);
 for(int i=0;i<160;i++)data[i]=(int32_t)rnd()>>(i&15);
 msort(data,160,order[0]);
 for(int i=1;i<160;i++)if(data[i-1]>data[i])bad++;
 hex(crc((const uint8_t*)data,640));
 msort(data,160,order[1]);
 hex(data[0]);
 hex(crc((const uint8_t*)data+1,637));
 /* sub-word writes at every alignment */
 for(int i=0;i<16;i++)((volatile uint8_t*)scratch)[i]=(uint8_t)(0xF0+i);
 for(int i=0;i<8;i++)scratch[i]=(int16_t)(scratch[i]^0x1234);
 hex(((volatile uint32_t*)scratch)[1]);
 volatile int32_t mn=-2147483647-1,neg1=-1,z=0,p=7;
 hex(mn/neg1);hex(mn%neg1);hex(p/z);hex(p%z);hex((uint32_t)p/(uint32_t)z);hex(-7/p*3+(-7%p));
 hex((uint32_t)(((int64_t)mn*(int64_t)p)>>32));hex((uint32_t)(((uint64_t)(uint32_t)mn*(uint64_t)(uint32_t)neg1)>>32));
 hex((uint32_t)((int64_t)mn*(uint64_t)(uint32_t)p>>32));
 hex(far_top(acc));
 hex(bad);
 return 0;}
