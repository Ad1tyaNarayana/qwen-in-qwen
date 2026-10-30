#include "platform.h"
#include "model.h"
static _Alignas(8) int16_t input[784],hidden[HIDDEN];
static int32_t dot(const int8_t *w,const int16_t *a,int n){
 uint32_t acc=0;
#if !defined(NATIVE) && !defined(SCALAR)
 for(int j=0;j<n;j+=4)acc=ndot4(acc,w+j,a+j);
#else
 for(int j=0;j<n;j++)acc+=(uint32_t)((int32_t)w[j]*a[j]);
#endif
 return (int32_t)acc;
}
static int32_t scale(int32_t x,int32_t m){return (int32_t)(((int64_t)x*m+(1<<19))>>20);}
static unsigned classify(const uint8_t *image){
 for(int i=0;i<784;i++)input[i]=image[i];
 for(int i=0;i<HIDDEN;i++){
  int32_t v=scale(dot(w1+i*784,input,784)+b1[i],m1[i]);
  hidden[i]=(int16_t)(v<0?0:v>32767?32767:v);
 }
 int32_t best=(-2147483647-1);unsigned prediction=0;
 for(int i=0;i<10;i++){
  int32_t v=scale(dot(w2+i*HIDDEN,hidden,HIDDEN)+b2[i],m2[i]);
  if(v>best){best=v;prediction=(unsigned)i;}
 }
 return prediction;
}
#ifdef ALL_TEST
int engine_main(void){
 FILE *f=fopen(TEST_IMAGES,"rb");if(!f)return 1;
 uint8_t image[784];unsigned n=0;
 while(fread(image,1,784,f)==784){byte(classify(image));n++;}
 fclose(f);return n==10000?0:2;
}
#else
int engine_main(void){
 for(int n=0;n<SAMPLES;n++){
  text("MNIST test ");number((unsigned)n);byte('\n');
  for(int y=0;y<28;y+=2){for(int x=0;x<28;x++)byte(" .+#"[images[n*784+y*28+x]/64]);byte('\n');}
  text("Label ");number(labels[n]);text(" prediction ");number(classify(images+n*784));byte('\n');
 }
 return 0;
}
#endif
