#include <ros/ros.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cvDia.h>
#include <opencv/cv.h>
#include <iostream>
#include <stdio.h>
#include <GL/freeglut.h>
#include <uv_face_detect/bBox.h>
#include <uv_face_detect/FaceDetected.h>
#include <glip.h>
// -- -- -- jgvitz
#include <string>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <sstream>
#include "dirent.h"
#include "std_msgs/String.h"
#include <opencv2/core/core.hpp>
#include <sensor_msgs/PointCloud2.h>
#include "libface.cpp"

// -- -- --
#define yMin -1.20
#define yMax 8.0

namespace enc = sensor_msgs::image_encodings;
using namespace cv;
using namespace std;
int peopleFound=0,nFaces,notValidtd;
int window, window2, window3;
int fbbData[64],noFacesData[64];
int imgRedFactor = 2;
glipImageSt *imageGlip,*imageGlip2, *imageGlip3;
glipDataSt *faceBBox,*noFaces;
CvRect *bBfaces,*rectNoFaces;  
IplImage* cvImage=NULL;	
Mat image,tempImg2;
char fileName[]="/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";

uv_face_detect::FaceDetected noFaceData;
uv_face_detect::FaceDetected faceData;
uv_face_detect::bBox msgtemp;
uv_face_detect::bBox msgtemp2;

// -- -- -- jgvitz
//string PATH = "/home/hakavitz/catkin_ws/src/uv_facial_expression/";
//#define PATH="/home/amarin/ros/hydro_ws/src/uv_facial_expression/" //para shaak-ti
//#define PATH="/home/vvicencio/catkin_ws/src/uv_facial_expression/" //para shaak-ti
cv_bridge::CvImagePtr cv_ptr;  
cv::Vec3b const VBLACK = Vec3b(0, 0, 0); ///< convenience color definition
int rf3x3[1][9]; //cambiar a 4

//guardar boundingbox
int cvRPtr(CvRect rect,int *data){
  int i;
  int j;

  data[0]=rect.x;     
  data[1]=rect.y;  
  data[2]=rect.x+rect.width; 
  data[3]=rect.y;  
  data[4]=rect.x+rect.width; 
  data[5]=rect.y;  
  data[6]=rect.x+rect.width; 
  data[7]=rect.y+rect.height;
  data[8]=rect.x+rect.width; 
  data[9]=rect.y+rect.height;  
  data[10]=rect.x;   
  data[11]=rect.y+rect.height;
  data[12]=rect.x;   
  data[13]=rect.y+rect.height;
  data[14]=rect.x;     
  data[15]=rect.y;  
 
  //multiplicar los valores por factor de reducción
  for(i=0;i<16;i++){
  data[i]*=imgRedFactor;
  }

  if (peopleFound>0){
   for(j=0;j<peopleFound;j++){
    msgtemp.boundingBox[0].x=rect.x;
    msgtemp.boundingBox[0].y=rect.y;
    msgtemp.boundingBox[0].z=0;
    msgtemp.boundingBox[1].x=rect.width;
    msgtemp.boundingBox[1].y=rect.height;
    msgtemp.boundingBox[1].z=0;
   }
  if (peopleFound>0){
   for(j=0;j<notValidtd;j++){
    msgtemp2.boundingBox[0].x=rect.x;
    msgtemp2.boundingBox[0].y=rect.y;
    msgtemp2.boundingBox[0].z=0;
    msgtemp2.boundingBox[1].x=rect.width;
    msgtemp2.boundingBox[1].y=rect.height;
    msgtemp2.boundingBox[1].z=0;
   }
  }
 }
 return 0;
}

int getGlipData(int peopleFound){
  int i;
  if (peopleFound>0){
   for(i=0;i<peopleFound;i++)
    cvRPtr(bBfaces[i],&fbbData[i*16]);}
  if (notValidtd>0){
   for(i=0;i<notValidtd;i++) 
    cvRPtr(rectNoFaces[i],&noFacesData[i*16]);}
  return 0;
}

void callback(const sensor_msgs::ImageConstPtr& msg){
  IplImage* iplImg;
  Mat tempImg;
  ros::NodeHandle nh1_;
  ros::NodeHandle nh2_;
  ros::Publisher pub;
  ros::Publisher pub2;
  pub = nh1_.advertise<uv_face_detect::FaceDetected>("faceData", 1000); 
  pub2 = nh2_.advertise<uv_face_detect::FaceDetected>("noFaceData",1000);
  //  sensor_msgs::CvBridge bridge;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

   imageGlip->data=cv_ptr->image.data;
   iplImg=cvCreateImage(cvSize(imageGlip->width/2,imageGlip->height/2),IPL_DEPTH_8U,3);
   resize(cv_ptr->image,tempImg,cvSize(imageGlip->width/2,imageGlip->height/2));//,0,0,CV_INTER_LINEAR);
  //  iplImg = cvCloneImage(&(IplImage)tempImg);
   iplImg->imageData = (char *) tempImg.data;
   peopleFound=cvDiaFindFaces(iplImg,&nFaces,&bBfaces,&rectNoFaces);

   getGlipData(peopleFound);
   notValidtd=nFaces-peopleFound;
   faceBBox->NoPts=peopleFound*8;
   noFaces->NoPts=notValidtd*8;
   
   faceData.noFaces=nFaces;
   if (faceData.noFaces>0){
     faceData.faceDetected.push_back(msgtemp);
     pub.publish(faceData);
     faceData.faceDetected.clear(); 
     //i1=(msgtemp.boundingBox[0].x)+abs(msgtemp.boundingBox[1].x/2);  //centro x
     //j1=(msgtemp.boundingBox[0].y)+abs(msgtemp.boundingBox[1].y/2);  //centro y
     int sizefacewidth=msgtemp.boundingBox[1].x; 
     int sizefaceheight=msgtemp.boundingBox[1].y;		
     int facex=msgtemp.boundingBox[0].x;
     int facey=msgtemp.boundingBox[0].y;
	
     cv::Rect roi(facex*imgRedFactor, facey*imgRedFactor, sizefacewidth*imgRedFactor, sizefaceheight*imgRedFactor);
     cv_ptr->image(roi).copyTo(tempImg2);
     //printf("%d %d %d %d %d\n",tempImg2.cols,tempImg2.rows, static_cast<int>(tempImg2.step[0]), static_cast<int>(tempImg2.step[1]), static_cast<int>(tempImg2.total()));//tempImg2.step[1]);
     //imageGlip2=(glipImageSt*)glipCreateImage(150,150,1,GLIP_RGB,GL_UNSIGNED_BYTE);
     //tempImg2.step1();
     imageGlip2->width=tempImg2.cols;
     imageGlip2->height=tempImg2.rows;
     imageGlip2->data=tempImg2.data;
   }

   noFaceData.noFaces=notValidtd;
   if (noFaceData.noFaces>0){
     noFaceData.faceDetected.push_back(msgtemp2);
     pub2.publish(noFaceData); 
     noFaceData.faceDetected.clear();
   }

   glipRedisplayImage(window);
   glipRedisplayImage(window2);
   glutMainLoopEvent();
}

//Function:
//            Save complete image of kinect 640x480  .jpg
//            Save face's boundingbox .jpg
int saveImage(int dist, int facex, int facey, int sizefacewidth,int sizefaceheight){
 //printf("sizefacewidth: %d \n",sizefacewidth);	
 //printf("sizefaceheight: %d \n",sizefaceheight);
 std::string result,directorio,directorio2,ext,nom,nom2;
 char distan[10];
 char facewidth[10];
 char faceheight[10];
 directorio=PATH+"imgs/sujeto1/";
 nom="sujeto1_";
 ext=".jpg";
 sprintf(distan, "%d", dist);
 sprintf(facewidth, "%d", sizefacewidth);
 sprintf(faceheight, "%d", sizefaceheight);
 result = directorio + nom + string(distan) + ext; 
 //cvtColor(cv_ptr->image, cv_ptr->image, CV_RGB2GRAY);
 cvtColor(cv_ptr->image,cv_ptr->image,CV_BGR2RGB);
 cv::imwrite(result, (cv_ptr->image));  //Guardar a disco imagen completa

 //resize(cv_ptr->image,tempImg2,cvSize(sizefacewidth,sizefaceheight));
 directorio=PATH+"imgs/sujeto1/rostro/";
 nom2="sujeto1rostro_";
 result = directorio + nom2 + string(distan) + "_" + string(facewidth) + "x" + string(faceheight) + ext;
 //cambiar color  tempImg2  cvtColor(cv_ptr->image,cv_ptr->image,CV_BGR2RGB);
 cv::imwrite(result,tempImg2);   //Guarda a disco solo rostro
 return 0;
}

//Function: Load rectangle features of training for each expression
int loadRectFeatures(){
  string line;
  ifstream myfile ("/home/hakavitz/catkin_ws/src/uv_facial_expression/src/highAlpha.txt");
  char split_char=' ';
  int cont1=0,cont2=0;
  vector<string> tokens;
  tokens.clear();
  if (myfile.is_open()){
    while ( getline(myfile,line) ){
     istringstream split(line);
     for (string each; std::getline(split, each, split_char); tokens.push_back(each));
     for (vector<string>::iterator it = tokens.begin()+2; it != tokens.end(); ++it){
      //cout << ' ' << *it;
      rf3x3[cont1][cont2]=atoi((*it).c_str());
      cont2++;
     }
    }
    myfile.close();
  }
  else cout << "Unable to open file"; 
  
  printf("Type RectFeat load... \n");
  for(int a=0;a<9;a++)
   printf("%d ",rf3x3[0][a]);
  printf("\n");
  
  return 0;
}

//Drawing text 2D screen.
void drawText(const char *text, int length, int x, int y){
 glMatrixMode(GL_PROJECTION); // change the current matrix to PROJECTION
 double matrix[16]; // 16 doubles in stack memory
 glGetDoublev(GL_PROJECTION_MATRIX, matrix); // get the values from PROJECTION matrix to local variable
 glLoadIdentity(); // reset PROJECTION matrix to identity matrix
 glOrtho(0, 800, 0, 600, -5, 5); // orthographic perspective
 glMatrixMode(GL_MODELVIEW); // change current matrix to MODELVIEW matrix again
 glLoadIdentity(); // reset it to identity matrix
 glPushMatrix(); // push current state of MODELVIEW matrix to stack
 glLoadIdentity(); // reset it again. (may not be required, but it my convention)
 glRasterPos2i(x, y); // raster position in 2D
 for(int i=0; i<length; i++){
  glutBitmapCharacter(GLUT_BITMAP_9_BY_15, (int)text[i]); // generation of characters in our text with 9 by 15 GLU font
 }
 glPopMatrix(); // get MODELVIEW matrix value from stack
 glMatrixMode(GL_PROJECTION); // change current matrix mode to PROJECTION
 glLoadMatrixd(matrix); // reset
 glMatrixMode(GL_MODELVIEW); // change current matrix mode to MODELVIEW
}

void displayRectFeat(){
  float center_u=50;//+sizefacewidth/2;
  float center_v=50;//+sizefaceheight/2;
  int incr=9;
  int pos=0;
  glClear(GL_COLOR_BUFFER_BIT);
  
  for(int v=0;v<3;v++){
   center_u=50;
   for(int u=0;u<3;u++){
    if (rf3x3[0][pos]==0){
      glColor3f(1.0, 1.0, 0.0); //yellow
      glRectf(center_u, center_v, center_u+incr, center_v+incr);
    }
    else{
      glBegin(GL_LINE_LOOP);
      glVertex2f(center_u, center_v);
      glVertex2f(center_u+incr, center_v);
      glVertex2f(center_u+incr, center_v+incr);
      glVertex2f(center_u, center_v+incr);
      glVertex2f(center_u, center_v);
      glEnd();
    }
    center_u=center_u+incr;
    pos=pos+1;
   }
   center_v=center_v+incr;
  }

  glFlush();
  glutSwapBuffers(); 
}

int expressionRecognition(Mat mface,int coordu,int coordv,int incrpix){
 int expr,pos;
 int i,j,x,y,halfsize,size,value;
 double tmp,sigma,sum,threshold,lt_cnt,gt_cnt,scl_factor;
 double sumBlack,sumWhite;
 double error1,alpha1;
 bool mr_t_thr;
 CvMat *cvmatface;
 CvMat *kernel;

printf("mface rows %d \n",mface.rows);
printf("mface cols %d \n",mface.cols);

 //create filter
 kernel = cvCreateMat(3,3,CV_64FC1);
 cvSetZero(kernel);
 halfsize = abs(3/2);
 sigma = (double)3/6;

 for(j=0; j<3; j++){
  for(i=0; i<3; i++){
   tmp=0.0;
   x = abs(j-halfsize);
   y = abs(i-halfsize);
   tmp = exp(-(double)(x*x+y*y)/sigma);
   cvmSet(kernel,j,i,tmp);
  }
 }
////////////////////
printf(" /////////  kernel  ////////\n");
 for(j=0; j<kernel->rows; j++){
  for(i=0; i<kernel->cols; i++){
   sigma=cvmGet(kernel,j,i);
   printf("%f ",sigma);
  }
   printf("\n");
 }
 printf("\n");
//////////////////
 //sum
 sum=0.0;
 for(j=0; j<mface.rows; j++){
  for(i=0; i<mface.cols; i++){
   value=mface.at<unsigned>(j,i);
   sum=sum+value;
  }
 }

 //average
 threshold=0.0;
 threshold=sum/(mface.cols*mface.rows);

 //Filter SQI illumination normalization simple

 //count >threshold <threshold
 lt_cnt=0;
 gt_cnt=0;
 for(j=0; j<mface.rows; j++){
  for(i=0; i<mface.cols; i++){
   value=mface.at<unsigned>(j,i);
   tmp=value;
   if(tmp>threshold){
    gt_cnt=gt_cnt+1;           //contador mayor que umbral
    }
   else{
    lt_cnt=lt_cnt+1;         //contador menor que umbral
   }
  }
 }
  
 if(gt_cnt>lt_cnt){ mr_t_thr=true; }
 else{ mr_t_thr=false; }

 //gaussian
 cvmatface = cvCreateMat(mface.rows,mface.cols,CV_64FC1);
 cvSetZero(cvmatface);

 scl_factor=0.0;
 y=0;
 for(j=1; j<mface.rows-1; j++){
  x=0;
  for(i=1; i<mface.cols-1; i++){
     value=mface.at<unsigned>(j,i);
     tmp=value;
     if(((tmp>threshold) && mr_t_thr==false) || ((tmp<threshold) && mr_t_thr==true)){
       cvmSet(cvmatface,j,i,0.0);
       //sigma=cvmGet(cvmatface,j,i);
       //printf("%f ",sigma);
     }
     else{
      if(x==3||x>3) x=0;
       tmp=cvmGet(kernel,y,x);
       scl_factor=scl_factor+tmp;
       cvmSet(cvmatface,j,i,tmp);
       x++;
     }
  }
  y++;
  if(y==3||y>3) y=0;
 }

   //normalization  
   for(j=0; j<mface.rows-1; j++){
    for(i=0; i<mface.cols-1; i++){
     tmp=cvmGet(cvmatface,j,i);
     if (tmp!=0){
      //printf("%f ",scl_factor);
      sigma=tmp/scl_factor;
      cvmSet(cvmatface,j,i,sigma);
      //sigma=cvmGet(cvmatface,j,i);
      //printf("sig %f ",sigma);
     }
    }
   }

////////////////////
//revision valores 
  Mat gray_image=cvmatface;
  imwrite("/home/hakavitz/catkin_ws/src/uv_facial_expression/imgs/preprocesamiento/conv_Image.jpg",gray_image);
/*
 for(j=0; j<cvmatface->rows; j++){
  for(i=0; i<cvmatface->cols; i++){
   sigma=cvmGet(cvmatface,j,i);
   printf("%f ",sigma);
  }
   printf("\n");
 }
 printf("\n");
*/
//////////////////// 

 //comparing

 //sum pixels black & white
 pos=0;
 sumBlack=0.0;
 sumWhite=0.0;
 scl_factor=0.0;

 for(int v=coordv;v<coordv+(3*incrpix);v=v+incrpix){
  for(int u=coordu;u<coordu+(3*incrpix);u=u+incrpix){
   if (rf3x3[0][pos]==0){
    scl_factor=integralImg(cvmatface,v,u,incrpix);
    sumBlack=sumBlack+scl_factor;
   }
   else{
    scl_factor=integralImg(cvmatface,v,u,incrpix);
    sumWhite=sumWhite+scl_factor;
   }
   pos=pos+1;
  }
 }

 printf("sumBlack %f \n",sumBlack);
 printf("sumWhite %f \n",sumWhite);

 error1=0.0;
 error1=sumWhite-sumBlack;
 printf("error1 %f \n",error1);

 //alpha1=0.0;
 //if(error1!=0.0){
  alpha1=log(error1/(1-error1));
  printf(" %f \n",1-error1);
  printf(" %f \n",error1/(1-error1));
  printf("alpha1 %f \n",alpha1);   ////son negativos
 //}
  printf("\n");
 return 0;
}

void profundidadCb(const sensor_msgs::PointCloud2::ConstPtr& point_msg){	
 int rSize2,i1,j1,cont,dist,facex,facey,sizefacewidth,sizefaceheight,ax,ay,bx,by;
 float x1,y1,z1,nearest;
 float *tmpPtr;	
 rSize2=point_msg->row_step;

 //para tomar solo el rostro más cercano
 if (nFaces==1){

 //printf("rSize: %d\n",rSize2);
 i1=(msgtemp.boundingBox[0].x)+abs(msgtemp.boundingBox[1].x/2);  //centro x
 j1=(msgtemp.boundingBox[0].y)+abs(msgtemp.boundingBox[1].y/2);  //centro y
 sizefacewidth=msgtemp.boundingBox[1].x;
 sizefaceheight=msgtemp.boundingBox[1].y;	
 facex=msgtemp.boundingBox[0].x;
 facey=msgtemp.boundingBox[0].y;
 //medidas reales
 i1=i1*imgRedFactor;
 j1=j1*imgRedFactor;
 sizefacewidth=sizefacewidth*imgRedFactor;
 sizefaceheight=sizefaceheight*imgRedFactor;
 facex=facex*imgRedFactor;
 facey=facey*imgRedFactor;
 //accceder a dato de profundidad en centro de boundingbox		
 tmpPtr=(float*)&point_msg->data[(j1*rSize2+i1*16)];//z	   y*size+x+bits
 x1=tmpPtr[0]; y1=tmpPtr[1]; z1=tmpPtr[2];
 dist=0;
 
 //rangos de casos de pruebas
 if (z1>=0.8 && z1<0.9) dist=8;
 if (z1>=0.9 && z1<1.0) dist=9;
 if (z1>=1.0 && z1<1.10) dist=10;
 if (z1>=1.10 && z1<1.20) dist=11;
 if (z1>=1.20 && z1<1.30) dist=12;
 if (z1>=1.30 && z1<1.40) dist=13;
 if (z1>=1.40 && z1<1.50) dist=14;
 if (z1>=1.50 && z1<1.60) dist=15;
 if (z1>=1.60 && z1<1.70) dist=16;
 if (z1>=1.70 && z1<1.80) dist=17;
 if (z1>=1.80 && z1<1.90) dist=18;
 if (z1>=1.90 && z1<2.0) dist=19;
 if (z1>=2.0 && z1<2.10) dist=20;
 if (z1>=2.10 && z1<2.20) dist=21;
 if (z1>=2.20 && z1<2.30) dist=22;
					
 saveImage(dist,facex,facey,sizefacewidth,sizefaceheight);

 //Búsqueda del pixel más cercano en profundidad
 nearest=1000.0;
 for(ax=facex;ax<facex+sizefacewidth;ax++){ //ax+=2
  for(ay=facey;ay<facey+sizefaceheight;ay++){ //y+=2
    tmpPtr=(float*)&point_msg->data[(ay*rSize2+ax*16)];//z	
    z1=tmpPtr[2];		      
    if(z1<nearest){ nearest=z1; }
  }
 }

 //printf("nearest (m): %f\n",nearest);
 //imagen pixeles a negros
 Mat tempImg3(cv_ptr->image, cv::Rect(facex, facey, sizefacewidth, sizefaceheight));
 ay=facey;
 for(int by=0;by<tempImg3.rows;by++){
  ax=facex;
  for(int bx=0;bx<tempImg3.cols;bx++){
   tmpPtr=(float*)&point_msg->data[(ay*rSize2+ax*16)];//z	 
   // y*size+x+bits ¡¡¡¡¡¡¡¡posibles problema en sincronia
   z1=tmpPtr[2];
   //printf("promd: %f , z1: %f \n",promd,z1);
   if(z1 > nearest+.20){ //  + 30 centimetros 
    tempImg3.at<Vec3b>(by,bx)=VBLACK;
   }
   ax++;
  }
  ay++;	   
 }

 //guardar imagen
 //cout << "tempImg3 = " << endl << " " << tempImg3 << endl << endl;
 std::string result,directorio,nom2,ext;
 char distan[10];
 char facewidth[10];
 char faceheight[10];
 sprintf(facewidth, "%d", sizefacewidth);
 sprintf(faceheight, "%d", sizefaceheight);
 directorio=PATH+"imgs/sujeto1/filtro/";
 nom2="sujeto1rostrofiltro_";
 ext=".jpg";
 sprintf(distan, "%d", dist);
 result = directorio + nom2 + string(distan) + "_" + string(facewidth) + "x" + string(faceheight) + ext;
 //cambiar color tempImg3  cvtColor(cv_ptr->image,cv_ptr->image,CV_BGR2RGB);
 cv::imwrite(result,tempImg3);


 ////////////////////////// RECOGNITION FACIAL EXPRESSION
 //IplImage* face3;
 //face3=cvCreateImage(cvSize(tempImg3.cols,tempImg3.rows),IPL_DEPTH_8U,1);
 cv::Mat greyMat;
 cv::cvtColor(tempImg3, greyMat, cv::COLOR_BGR2GRAY);

 printf("expressionRecognition\n");
 expressionRecognition(greyMat,50,50,9); //cambiar valores fijos
 //delete tempImg3;

 }

  displayRectFeat();

 //glipRedisplayImage(window2);
 //glipRedisplayImage(window3);
 //glutMainLoopEvent();
}

static void keyboard ( unsigned char key, int x, int y )
{
  switch ( key ) {
  case 27 :
  case 'q' :
  case 'Q' :
    exit(0);
    break;
  }
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "uv_facial_expression");  //nombre nodo cambió
  glutInitWindowSize(640,480); //ventana principal
  glutInit(&argc,argv);
  glutInitDisplayMode (GLUT_RGB);

  imageGlip=(glipImageSt*)glipCreateImage(640,480,1,GLIP_RGB,GL_UNSIGNED_BYTE);
  imageGlip2=(glipImageSt*)glipCreateImage(150,150,1,GLIP_RGB,GL_UNSIGNED_BYTE);
	
  imageGlip->signY=-1;
  imageGlip2->signY=-1;

  cvImage=cvCreateImageHeader(cvSize(640,480),IPL_DEPTH_8U,3);
  faceBBox=(glipDataSt*)glipCreateDataSt(0,2,2,GLIP_COLOR_GREEN,GLIP_LINES,GLIP_INT,fbbData);
  noFaces=(glipDataSt*)glipCreateDataSt(0,2,2,GLIP_COLOR_RED,GLIP_LINES,GLIP_INT,noFacesData);
  //cvDiaInitPeopleDet(40/imgRedFactor,100/imgRedFactor,fileName);
  cvDiaInitPeopleDet(50,60,fileName);

  ros::NodeHandle nh_;
  ros::Subscriber sub;
  sub = nh_.subscribe("/camera/rgb/image_rect_color", 1, callback);
  window=glipDisplayImage(imageGlip,"Kinect Image",0);
  window2=glipDisplayImage(imageGlip2,"Face Image",0);

// -- -- -- jgvitz
     int reload=loadRectFeatures();
     ros::NodeHandle node_; //probar con el mismo nodo nh_
     ros::Subscriber sub_;
     sub_ = node_.subscribe("/camera/depth/points", 1, profundidadCb);
// -- -- --

  glipDrawInImage(window,faceBBox);
  glipDrawInImage(window,noFaces);

  glutKeyboardFunc(keyboard);

  while (ros::ok())
   { 
     ros::spinOnce();
   }

return 0;
}
