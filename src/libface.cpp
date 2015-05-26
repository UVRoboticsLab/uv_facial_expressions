/*
** Copyright (C) 2006 Sujith K.R. All rights reserved.
** Email: sujith.k.raman@gmail.com,sujithkr@au-kbc.org
     
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions
** are met:
** 1. Redistributions of source code must retain the above copyright
**    notice, this list of conditions and the following disclaimer.
** 2. Redistributions in binary form must reproduce the above copyright
**    notice, this list of conditions and the following disclaimer in the
**    documentation and/or other materials provided with the distribution.
** 3. The name of the author may not be used to endorse or promote products
**    derived from this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
** ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
** ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
** OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
** HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
** LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
** OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
** SUCH DAMAGE.
** 
** Modifications by Veronica Vicencio-Del Angel
** Email: jgvitz@gmail.com
*/

#include "../include/uv_facial_expression/libface.h"

using namespace std;

string PATH = "/home/hakavitz/catkin_ws/src/uv_facial_expression/";
//#define PATH="/home/amarin/ros/hydro_ws/src/uv_facial_expression/" //para shaak-ti
//#define PATH="/home/vvicencio/catkin_ws/src/uv_facial_expression/" //para shaak-ti

/*
 ** Function Illumination normalization. Process Self Quotient image (SQI) 
 ** Input:
 **         IplImage* inp	Input facial image to be pre-processed
 ** Output:
 **         IplImage* 		Output normalized image
*/
IplImage* SQI(IplImage* inp){
  int    size_filter,i,j,k;
  double tmp1,tmp2,tmp3;
  IplImage *res_img;
  CvMat *filtered_image;
  CvMat *qi;
  CvMat *inp_mat;
  CvMat *res;
  CvMat *g_ker; 

  size_filter = 3;
  inp_mat = IplImage2Mat(inp);   //libface
  g_ker = Gaussian(size_filter); //libface
  filtered_image = cvCreateMat(inp_mat->rows,inp_mat->cols,CV_64FC1); 

  filtered_image = Conv_Weighted_Gaussian(inp,g_ker);  //libface  
  
  qi = cvCreateMat(inp_mat->rows,inp_mat->cols,CV_64FC1);

  //aplica transformación logaritmica  
  for(j=0; j<inp_mat->rows; j++){
     for(k=0; k<inp_mat->cols; k++){
        tmp1 = cvmGet(inp_mat,j,k);
        tmp2 = cvmGet(filtered_image,j,k);
        tmp3 = log10((tmp1+1.0)/(tmp2+1.0));
	cvmSet(qi,j,k,tmp3);
     }
  }

  Scale_Mat(qi,255);               //libface

  res = cvCreateMat(inp_mat->rows,inp_mat->cols,CV_64FC1);
  cvSetZero(res);
  
  for(j=0;j<inp_mat->rows;j++){
     for(k=0;k<inp_mat->cols;k++){
    	tmp1 = cvmGet(qi,j,k);
	tmp2 = cvmGet(res,j,k);
	cvmSet(res,j,k,tmp1+tmp2);
     }
  }

  //Scale_Mat(res,255);           //libface
  res_img = Mat2IplImage(res,0);  //libface
  cvReleaseMat(&g_ker);           //free mem
  cvReleaseMat(&filtered_image);  //free mem
  cvReleaseMat(&qi);              //free mem
  cvReleaseMat(&inp_mat);         //free mem  
  cvReleaseMat(&res);             //free mem  
  return res_img;
}

/** Funcion para convertir imagen de IplImage a Mat. **/
CvMat* IplImage2Mat(IplImage *inp_img){
  CvMat *result;
  IplImage *temp;
  int i,j;
  unsigned char tmp_char;
  temp=Rgb2Gray(inp_img);
  result=cvCreateMat(temp->height,temp->width,CV_64FC1);
   for(i=0;i<temp->height;i++){
      for(j=0;j<temp->width;j++){
         tmp_char=(unsigned char)temp->imageData[(i*temp->widthStep)+j];
         cvmSet(result,i,j,(double)tmp_char);
       }
   }
//   cvReleaseImage(&inp_img); //free mem
//   cvReleaseImage(&temp);    //free mem
   return result;
}

IplImage* Mat2IplImage(CvMat *inp_mat,int type){  
  IplImage *result;
  int i,j;
  double tmp_val;
  if(type==0){
      result=cvCreateImage(cvSize(inp_mat->cols,inp_mat->rows),IPL_DEPTH_8U,1);
  }
  else if(type==1){
      result=cvCreateImage(cvSize(inp_mat->cols,inp_mat->rows),IPL_DEPTH_32F,1);
  }
  else{
      return 0;
  }
 
  for(i=0;i<result->height;i++){
      for(j=0;j<result->width;j++){
          tmp_val=cvmGet(inp_mat,i,j);
          result->imageData[(i*result->widthStep)+j]=(unsigned char) tmp_val;
      }
  }
  cvReleaseMat(&inp_mat);  //free mem
  return result;
}

/*
 ** Funcion para crear kernel 3x3 e inicializar valores.
 ** Entrada:
 **         int size	imagen de entrada a procesamiento
 ** Salida:
 **         CvMat* 		imagen normalizada
*/
CvMat* Gaussian(int size){
   CvMat *res;
   int i,j,x,y,halfsize;
   double tmp,sigma;   
   sigma = (double)size/6;  // x5?      // 3/5=0.6
   res = cvCreateMat(size,size,CV_64FC1);
   halfsize = size/2;         // 3/2=1.5
   for(i=0; i<res->rows; i++){
      for(j=0; j<res->cols; j++){
         x = j-halfsize;
         y = i-halfsize;
         tmp = exp(-(double)(x*x+y*y)/sigma);
         cvmSet(res,i,j,tmp);
      }
   }
   return res;
}
/*
 ** Funcion convolucion de imagen con filtro gausiano con pesos.
 ** Entrada:
 **         IplImage *inp_img	Imagen para ser procesada
 **         CvMat *kernel       Kernel tamaño 3x3
 ** Salida:
 **         CvMat*	   Imagen con convolucion
*/
CvMat* Conv_Weighted_Gaussian(IplImage *inp_img,CvMat *kernel){
   int i,j;
   double val;
   CvPoint start;
   CvMat *tmp;
   CvMat *ddd;
   CvMat *w_gauss;
   ddd = cvCreateMat(inp_img->height,inp_img->width,CV_64FC1);
   for(i=0; i<inp_img->height; i++){
       for(j=0; j<inp_img->width; j++){
            start.x = abs(j-(kernel->cols/2));
            start.y = abs(i-(kernel->rows/2));
            tmp = Get_Mat(start,kernel->cols,kernel->rows,inp_img); //obtiene region
            w_gauss = Weighted_Gaussian(tmp,kernel); //aplica kernel a region	    
            val = cvDotProduct(w_gauss,w_gauss); //producto de matrices 
            cvmSet(ddd,i,j,val);
       }
   }
   cvReleaseMat(&tmp);
   cvReleaseMat(&w_gauss);
   cvReleaseImage(&inp_img); //free mem
   cvReleaseMat(&kernel);    //free mem
   //cvReleaseMat(&tmp);       //free mem
   //cvReleaseMat(&w_gauss);   //free mem
   return ddd;
}

/*
 ** Funcion Weighted_Gaussian Aplica kernel gaussiano y normaliza pesos.
 ** Entrada:
 **         CvMat *inp	
 **         CvMat *gaussian   
 ** Salida:
 **         CvMat*	  
*/
CvMat* Weighted_Gaussian(CvMat *inp,CvMat *gaussian){
   double sum,tmp1,tmp2,scl_factor,lt_cnt,gt_cnt,threshold;
   int i,j;
   bool mr_t_thr;   
   CvMat* w_gauss = cvCreateMat(gaussian->rows,gaussian->cols,CV_64FC1);
   cvSetZero(w_gauss);
   sum=0.0;
   for(i=0; i<inp->rows; i++){
      for(j=0; j<inp->cols; j++){
            sum=sum+cvmGet(inp,i,j);     //sumatoria
      }
   }
   threshold=sum/(inp->cols*inp->rows);  //promedio
   lt_cnt=0;
   gt_cnt=0;
   for(i=0; i<inp->rows; i++){
      for(j=0; j<inp->cols; j++){
          tmp1=cvmGet(inp,i,j);
          if(tmp1>threshold){
             gt_cnt=gt_cnt+1;           //contador mayor que umbral
          }
          else{
               lt_cnt=lt_cnt+1;         //contador menor que umbral
          }
      }
   }
   if(gt_cnt>lt_cnt){
      mr_t_thr=true;
   }
   else{
       mr_t_thr=false;
   }        
   scl_factor=0.0;
   for(i=0; i<inp->rows; i++){
      for(j=0; j<inp->cols; j++){
          tmp1=cvmGet(inp,i,j);
          if(((tmp1>threshold)&& mr_t_thr==false) || ((tmp1<threshold)&& mr_t_thr==true)){
             cvmSet(w_gauss,i,j,0.0);
          }
          else{
               tmp2=cvmGet(gaussian,i,j);
               scl_factor=scl_factor+tmp2;  //sumatoria
               cvmSet(w_gauss,i,j,tmp2);
          }
      }
   }      
   /*Normalizing the weighted gaussian matrix*/
   for(i=0; i<inp->rows; i++){
    for(j=0; j<inp->cols; j++){
     tmp1 = cvmGet(w_gauss,i,j);
     if(tmp1!=0){
       tmp2 = tmp1/scl_factor;
       cvmSet(w_gauss,i,j,tmp2);
     }
    }
   }
   cvReleaseMat(&inp);      //free mem
   cvReleaseMat(&gaussian); //free mem
   return w_gauss;
}

/*
 ** Funcion Obtiene region de la imagen con pesos.
 ** Entrada:
 **         CvPoint a	
 **         int width     
 **         int height
 **         IplImage *image
 ** Salida:
 **         CvMat*	   
*/
CvMat* Get_Mat(CvPoint a,int width,int height,IplImage *image){
  CvMat *fea_ar;
  unsigned char t_val;
  int h_i,w_i,i,j;
  fea_ar=cvCreateMat(height,width,CV_64FC1);
  cvSetZero(fea_ar);

   for(i=a.y; i<(a.y+height); i++){
     for(j=a.x; j<(a.x+width); j++){
        if((i>=0)&&(0>=j)&&(i<(image->height))&&(j<(image->width))){
           t_val=(unsigned char)image->imageData[(i*image->widthStep)+j]; //**
           cvmSet(fea_ar,i-a.y,j-a.x,(double)t_val);
        }
        else{
            if(j<0){
                    w_i=image->width+j;
            }
            else if(j>=image->width){

                    w_i=j-image->width;
                 }
                 else{
                     w_i=j;
                 }
            if(i<0){
                    h_i=-i;
            }
            else if(i>=image->height){
                    h_i=image->height-(i-image->height);
                 }
                 else{
                     h_i=i;
                 }
            t_val=(unsigned char)image->imageData[(h_i*image->widthStep)+w_i]; //**
            cvmSet(fea_ar,h_i-a.y,w_i-a.x,(double)t_val);
        }
     } 
   }
   cvReleaseImage(&image); //free mem
   return (fea_ar);
}

/*
 ** Funcion .
 ** Entrada:
 **         CvMat *input
 **         double scale  
 ** Salida:
 **         int	   
*/
int Scale_Mat(CvMat *input,double scale){
  double tmp,val,min,max;
  int i,j;
  min = 20000.0;
  max = -20000.0;  
  for(i=0; i<input->rows; i++){
    for(j=0; j<input->cols; j++){
      tmp=cvmGet(input,i,j);
      if(tmp<min)
         min=tmp;
      if(tmp>max)
         max=tmp;
    }
  }
  //printf("%g - %g\n",min,max); 
  for(i=0; i<input->rows; i++){
    for(j=0; j<input->cols; j++){
       tmp=cvmGet(input,i,j);
       val=scale*((tmp-min)/(max-min));
       cvmSet(input,i,j,val);
    }
  }
  cvReleaseMat(&input); //free mem
  return 1;
}

IplImage* Rgb2Gray(IplImage *src){
 IplImage *result;
 int i,j;
 int step_src,step_res;
 result=cvCreateImage(cvSize(src->width,src->height),src->depth,1);
 unsigned char *src_data;
 unsigned char *res_data;
 src_data=(unsigned char*)src->imageData;
 res_data=(unsigned char*)result->imageData;
 step_src=src->widthStep;
 step_res=result->widthStep;
 for(i=0;i<src->height;i=i+1){
    for(j=0;j<(src->width*src->nChannels);j=j+src->nChannels){
        res_data[j/src->nChannels]=(unsigned char)((0.3*(double)src_data[j+2])+(0.59*(double)src_data[j+1])+(0.11*(double)src_data[j]));
        //res_data[j+2]=src_data[j+2]; BGR format gray
    }
       src_data+=step_src;
       res_data+=step_res;
 }
 cvReleaseImage(&src); //free mem
 return result;
}

int createRectFeatures(){
 //int r3x3[250][9];//rows cols
 //0-black   1-white 
 ifstream fin;
 fin.open ("/home/hakavitz/catkin_ws/src/uv_facial_expression/src/typeRectanglesFeatures.dat");
 if (fin){
   for (int y=0; y<250; y++){
      for (int x=0; x<9; x++ ){
         fin >> r3x3[y][x];
      }
   } 
   //Printf matrix type rectangle features
   printf( "\n" );
   for (int a=0; a<250; a++){
     for (int b=0; b<9; b++){
       printf( "%d ", r3x3[a][b]);
     }
     printf( "\n" );
   }
   fin.close();
   return 0;
 }
 else{
    printf( "Cannot open file: typeRectanglesFeatures.dat\n" );
    return -1;
 }
}

int deleteRectFeatures(){
 //CvMat *res;
 int x,y;
 return 0;
}

/*
 ** Function .
 ** Input:
 **         string dir             Path directory
 **         vector<string> &files  Vector files's names
 ** Output:
 **         int	                   Number of files in directory
*/
int getDir (string dir, vector<string> &files){
    int numFiles;
    DIR *dp;
    struct dirent *dirp;
    if((dp=opendir(dir.c_str()))==NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }
    numFiles=0;
    while ((dirp = readdir(dp))!=NULL) {
        if((string(dirp->d_name)!=".") && !(string(dirp->d_name)=="..")){
          files.push_back(string(dirp->d_name));
          numFiles=numFiles+1;
        }
        
    }
    closedir(dp);
    return numFiles;
}


int ViolaJonesBoostingTraining(){
 int m=10;  //num img negative -hap
 int l=21;  //num img positive +hap
 int tot;   //total imgs
 tot=l+m;
 int e;
 int refBestRF;
 double W1[4][tot]; //weight hap //now 1 
 int tRecFeaHap[tot];
 double sum;
 double threshold=0.5; //?
 double val_act;
 double val_ant;
 double val_sup;
 double error[tot];
 double alphas[4][tot]; //0-hap   neutral surprise disgust
 double highAlpha[4][250];
 double bestalphaRF;
 int refq[250];
 unsigned char t_val;
 CvMat *matSrc;
 IplImage* Isrc;
 string dir = string(PATH+"imgs/preprocesamiento/posHA/");
 string pathfile;
 vector<string> files = vector<string>();
 int numf;

 //load names imgs db 
 numf=getDir(dir,files);
 cout<<numf<<endl;
 for (unsigned int i=0;i<files.size();i++){
   if(!(files[i]==".")&&!(files[i]=="..")){
     cout<<files[i]<<endl; 
   }
 }
 
 //for each facial expression classes
 for (int p=0;p<1;p++){  //now happiness    //after neutral,surprise,disgust

   //all rectangular features  type structure 3x3
   for (int q=0;q<250;q++){  //total 0-249 permutations

     //inicialize weights
     for(int a=0;a<l;a++){
       W1[p][a]=(double)1/(2*l);   
     }
      
     for(int b=l;b<tot;b++){
      W1[p][b]=(double)1/(2*m);
     }

     for (int t=0; t<numf/*tot*/; t++){ //each image example Weak Classifier
       //first positive 
       //second negative

       pathfile=dir+files[t];
       printf("%s \n",pathfile.c_str());
       Isrc=cvLoadImage(pathfile.c_str());
       matSrc=IplImage2Mat(Isrc); // 1 canal

       //normalize weights   
       for(int c=0;c<numf/*tot*/;c++){
         sum=W1[p][c];
       }
       for(int d=0;d<numf/*tot*/;d++){
         W1[p][d]=W1[p][d]/sum;
       }
  
       //classification
       error[t]=weakClassifier(matSrc, q, W1[p][t]);

       //update weights
       if (t<l) e=0;
       else e=1;
       for(int f=0;f<numf/*tot*/;f++){
         W1[p][f]=W1[p][f]*(error[t]/(1-error[t]));
       }
    
       //*calculate alpha
       alphas[p][t]=0.0;
       if(error[t]!=0){
        alphas[p][t]=log(error[t]/(1-error[t]));
       }
       tRecFeaHap[t]=q;
       printf("alpha %f \n",alphas[p][t]);

     }

     //find most high alpha value by all images
     highAlpha[p][q]=0.0;
     refq[q]=0;
     for(int g=0;g<numf/*tot*/;g++){
      if (!isnan(alphas[p][g]) && !isinf(alphas[p][g]) && alphas[p][g]>highAlpha[p][q]){
       highAlpha[p][q]=alphas[p][g];
       refq[q]=g;
       //printf("alpha %f \n",alphas[p][g]);
       printf("highAlpha[%d][%d] %f ,",p,q,highAlpha[p][q]);
      }
     }
       printf("\n");
     
   }
   
   //find most high alpha value by type rectangle
   bestalphaRF=0.0;
   refBestRF=0;
   for(int h=0;h<250;h++){
     if(highAlpha[p][h]>bestalphaRF){
	bestalphaRF=highAlpha[p][h];
	refBestRF=refq[h];
     }
   }
   //save best alpha by facial expression
   int saved=savetofile(refBestRF, bestalphaRF);
   printf("bestalphaRF %f \n",bestalphaRF);
   printf("refBestRF %d \n",refBestRF);
 }
 
 //cvReleaseImage(&Isrc); //free mem
 //cvReleaseMat(&matSrc); //free mem
 return 0;
}

double weakClassifier(CvMat *input, int typeStruct, double weight){ //return lowest error 
 CvMat *t_rect;
 double lowesterror;
 double sumBlackWhite[2];
 int count;
 double listerror[input->rows*input->cols];

 //load 1 type structure
 t_rect=getRectFeatures(typeStruct);
 count=0;

 //3x3 normal scale
 for(int i=0; i<input->rows-2; i++){
  for(int j=0; j<input->cols-2; j++){
   sumBlackWhite[0]=filterFeatures(input, t_rect, 0, 1); //color 0black // 1 // 3 // 6 scale
   sumBlackWhite[1]=filterFeatures(input, t_rect, 1, 1); //color 1white // 1 // 3 // 6 scale
   listerror[count]=weight*abs((sumBlackWhite[1]-sumBlackWhite[0]));
   count++;
  }
 }

 //find lowest error
 lowesterror=listerror[0];
 for(int k=0; k<count; k++){
  //printf("%f ",listerror[k]);
  if (listerror[k]<lowesterror){
   lowesterror=listerror[k];
  }
 }
 
 //cvReleaseMat(&t_rect);  //free mem
 //cvReleaseMat(&input);   //free mem
 printf("lowest error %f\n",lowesterror);
 return lowesterror;
}

//load each type structure
CvMat *getRectFeatures(int typeStruct){
 CvMat *res;
 int x,y,z;
 res=cvCreateMat(3,3,CV_64FC1);
 z=0;
 for(int y=0; y<res->rows; y++){
  for(int x=0; x<res->cols; x++){
   cvmSet(res,y,x,r3x3[typeStruct][z]);
   z++;
  }
 }
 return res;
}

//sumarizing individuel filter
double filterFeatures(CvMat *imginput, CvMat *imgrect, int color, int scale){ //return sum values, first(0) black, second(1) white 
  double sumWhite;
  double sumBlack;
  int tmp1;
  sumWhite=0;
  sumBlack=0;

  if (scale==1){ //normal scale
   for(int y=0;y<imgrect->rows;y++){
    for(int x=0;x<imgrect->cols;x++){
       tmp1 = cvmGet(imgrect,y,x);
       if (tmp1==1) sumWhite=cvmGet(imginput,y,x);
       if (tmp1==0) sumBlack=cvmGet(imginput,y,x);   
    }
   }
  }
  else
   if (scale==2){ //9x9
     ;//escalada
   }

  //cvReleaseMat(&imginput);  //free mem
  //cvReleaseMat(&imgrect);  //free mem

  if (color==0) //black
   return sumBlack;
  else
   if (color==1) //white
   return sumWhite;
   else
    return -999.999;
}

double integralImg(CvMat *imginp, int posv, int posu,int npix){
  double sumatoria;
  sumatoria=0.0;
  for(int k=0; k<npix; k++){ //rows
   for(int l=0; l<npix; l++){ //cols
     sumatoria=sumatoria+cvmGet(imginp,posv+k,posu+l);
   }
  }
  return sumatoria;
}

int savetofile(int qq, double alf){
/** 
 save coordinate point width heght
 save feature exact
 to file
 5 best
**/
 ofstream fs("/home/hakavitz/catkin_ws/src/uv_facial_expression/src/highAlpha.txt");
 stringstream ss;
 string str;
 ss <<"hap"<<" "<<alf<<" "
    <<r3x3[qq][0]<<" "<<r3x3[qq][1]<<" "<<r3x3[qq][2]<<" "
    <<r3x3[qq][3]<<" "<<r3x3[qq][4]<<" "<<r3x3[qq][5]<<" "
    <<r3x3[qq][6]<<" "<<r3x3[qq][7]<<" "<<r3x3[qq][8]<<endl;
 str = ss.str();
 ss.clear();
 fs<<str;
 fs.close();
 return 0;
}
