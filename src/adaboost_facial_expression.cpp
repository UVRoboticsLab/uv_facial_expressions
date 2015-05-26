#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cvDia.h>
#include <opencv/cv.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include "libface.cpp"

using namespace std;

int main(int argc, char **argv)
{
/*
        IplImage* src= cvLoadImage("1.tif");
        Mat m= src;//no copy
        Mat m1(src);//no copy
        m.release();//don't free mem
        m1.release();//don't free mem
        cvReleaseImage(&src);//free mem
*/
   /******* Cargar imagenes de base de datos para entrenamiento ************/
   //!!!automatizar
   //IplImage* Isrc= cvLoadImage("/home/hakavitz/catkin_ws/src/uv_facial_expression/imgs/preprocesamiento/MK.HA1.116.tiff");
   //IplImage* Iresult;

   //Preprocessing
   //normalizacion iluminacion 
   //!!!automatizar
   //Iresult=SQI(Isrc); 
   //cvSaveImage("/home/hakavitz/catkin_ws/src/uv_facial_expression/imgs/preprocesamiento/SQIMK.HA1.116.jpg",Iresult);

   //Training
   //ADABOOST Learning ALgorithm
   //Haar-Like rectangle features
   int s=createRectFeatures();
   if (s==0)
     int r=ViolaJonesBoostingTraining();
   else
    return -1;
   //cvReleaseImage(&Isrc);
   //cvReleaseImage(&Iresult);

   return 0;
}
