//includes
#include <vector>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <unistd.h>

//namespaces
using namespace std;
using namespace cv;


//fuction declerations
vector<Mat> autoCorrelate (vector<Mat> input,int offset);
void haveALook(int lengthOfBuffers, vector<Mat> corrBuffer, vector<Mat> imageBuffer);
Mat findAvOfVid(string fileName,float decimation);

//cheeky global variables



int main(){
    //parameters worth changing.
    int lengthOfCode = 8;
    int derating = 2;           //this is the factor slow down that Oscar suggested.
    float decimation = 0.1;      //the amount the image is resized, makes performance better.
    float secondsToProcess = 2;
    float FPSCamera = 118.4;
    string fileName = "slowerFlash.mp4";

    //derived less interesting variables;
    int numberToDo = int(FPSCamera * secondsToProcess);
    int lengthOfBuffers = lengthOfCode * derating;

    Mat av = findAvOfVid(fileName,decimation);
    //open file
    VideoCapture cap(fileName.c_str());
    if(!cap.isOpened()) {
        cout << "we did not open the file/camera correctly"<<endl;
        return -1; // check if we succeeded
    }
    int NoOfFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    float FPS = cap.get(CV_CAP_PROP_FPS);
    cout<<"The loaded file has "<< NoOfFrames << " frames. These were recorded at "<< FPS<<" FPS."<< endl;
    if(NoOfFrames < numberToDo){
        cout<< "this video is not of the required length"<< endl;
        return -1;
    }
    cout <<"set up video stream"<<endl;
    //declare variables
    Mat frameIn;                //The newest frame from the stream
    Mat frameSmall;             // A compressed version of the image
    Mat frameGrey;              // gray scale image.
    Mat frameFloat;             // float image that is the 'base' of the operation
    vector<Mat> imageBuffer;    //buffer with grey images
    vector<Mat> corrBuffer;     //buffer with the corelation accumilation.

    //initialise our buffers.
    for(int i = 0; i <lengthOfBuffers; i++){
        cap >> frameIn;
        resize(frameIn,frameSmall,Size(0,0),decimation,decimation);
        cv::cvtColor(frameSmall, frameGrey, CV_BGR2GRAY);
        frameGrey.convertTo(frameFloat,CV_32FC1);
        imageBuffer.push_back(Mat(frameFloat.size(),CV_32FC1, 0.0));
        imageBuffer.at(i) = frameFloat-av;
        //frameFloat.copyTo(imageBuffer.at(i));
        corrBuffer.push_back(Mat(frameFloat.size(),CV_32FC1, 0.0));
    }


    cout<< "finished initialising buffers"<<endl;
    //process the whole video here.
    cout<< lengthOfBuffers << " "<< numberToDo << endl;

    for(int i = lengthOfBuffers; i < numberToDo; i++){          //for every frame to be processed.
        vector<Mat> thisCorrelation = autoCorrelate(imageBuffer,i%lengthOfBuffers);
        for(int j = 0; j< lengthOfBuffers; j++){                    // accumilate the correlation in all bins at each time step forwards.
            corrBuffer.at(j) = corrBuffer.at(j) +thisCorrelation.at(j);
        }
        cap >> frameIn;
        resize(frameIn,frameSmall,Size(0,0),decimation,decimation);
        cv::cvtColor(frameSmall, frameGrey, CV_BGR2GRAY);
        frameGrey.convertTo(frameFloat,CV_32FC1);
        imageBuffer.at(i%lengthOfBuffers) = frameFloat-av;
        //frameFloat.copyTo(imageBuffer.at(i%lengthOfBuffers));

        cout<<i<<" "<<flush;
        if (i%20 == 0) cout<<endl;
    }
    Mat averageCorr(corrBuffer.at(0).size(),CV_32FC1);
    for(int i = 0; i< lengthOfBuffers; i++){
        averageCorr += corrBuffer.at(i);
    }
    averageCorr /= lengthOfBuffers;
    for(int i = 0; i< lengthOfBuffers; i++){
        corrBuffer.at(i) -= averageCorr;
    }
    double min, max;
    for(int i = 0; i < lengthOfBuffers; i++){
        minMaxLoc(corrBuffer.at(i), &min, &max);
    }
    cout << "min corr val:"<< min<< " max corr val:"<< max<<endl;
    for(int i = 0; i < lengthOfBuffers; i++){
        corrBuffer.at(i) -= min;
        corrBuffer.at(i) *= 1/max;
    }
    for(int i = 0; i < lengthOfBuffers; i++){
        corrBuffer.at(i) += 1;
        corrBuffer.at(i) *= 0.5;
    }

    haveALook(lengthOfBuffers,corrBuffer,imageBuffer);


}

//function implementations

vector<Mat> autoCorrelate (vector<Mat> input, int offset){      //the ofset aims to allow a circular buffer use of the vector.

    vector<Mat> corrResult;
    int sz = input.size();
    Mat intermediateCorr(input.at(0).size(),CV_32FC1,0.0);
    for(int i = 0; i < input.size(); i++){                                      //for all offsets of the thing to correlate
        for(int j = 0; j < input.size(); j++){                                  //for each pair of images that align this time
            Mat a = input.at((j+offset)%sz);
            Mat b = input.at((i+j+offset)%sz);
            Mat c = a.mul(b);
            intermediateCorr = c+intermediateCorr;
            cout << "|"<<(j+offset)%sz << ","<<(i+j+offset)%sz<<"| ";
        }
        cout<<endl;
        corrResult.push_back(Mat(input.at(0).size(),CV_32FC1,0.0));
        intermediateCorr.copyTo(corrResult.at(i));

    }
    cout<<"------------------------------------------------------"<<endl;
    return corrResult;
}
void haveALook(int lengthOfBuffers, vector<Mat> corrBuffer, vector<Mat> imageBuffer){
    namedWindow("corrBuffer",WINDOW_NORMAL );
    namedWindow("imageBuffer",WINDOW_NORMAL );
    while(1){
        int k =waitKey(1);
        usleep(100000);
        static int myWin =0;
        if(k == 'j'){
            myWin++;
            if( myWin == lengthOfBuffers) myWin = lengthOfBuffers-1;
            cout<<myWin<<endl;
        }
        if(k == 'k'){
            myWin--;
            if( myWin == -1) myWin = 0;
            cout<<myWin<<endl;
            cout<< "value"<<corrBuffer.at(myWin).at<float>(20 ,20)<<endl;
        }
        if(k == 'q') break;
        imshow("corrBuffer",corrBuffer.at(myWin));
        imshow("imageBuffer",imageBuffer.at(myWin)/255);
    }
    return;
}

Mat findAvOfVid(string fileName,float decimation){
    Mat frameIn;                //The newest frame from the stream
    Mat frameSmall;             // A compressed version of the image
    Mat frameGrey;              // gray scale image.
    Mat frameFloat;             // float image that is the 'base' of the operation
    VideoCapture cap(fileName.c_str());

    int NoOfFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    float FPS = cap.get(CV_CAP_PROP_FPS);
    cout<<"The loaded file has "<< NoOfFrames << " frames. These were recorded at "<< FPS<<" FPS."<< endl;

    cap >> frameIn;
    resize(frameIn,frameSmall,Size(0,0),decimation,decimation);
    cv::cvtColor(frameSmall, frameGrey, CV_BGR2GRAY);
    frameGrey.convertTo(frameFloat,CV_32FC1);
    Mat averageFloatIn(frameFloat.size(),CV_32FC1,0.0);
    frameFloat.copyTo(averageFloatIn);
    for(int i = 1; i < NoOfFrames; i++){
        cap >> frameIn;
        resize(frameIn,frameSmall,Size(0,0),decimation,decimation);
        cv::cvtColor(frameSmall, frameGrey, CV_BGR2GRAY);
        frameGrey.convertTo(frameFloat,CV_32FC1);
        averageFloatIn += frameFloat;
    }
    averageFloatIn = averageFloatIn/NoOfFrames;
    return averageFloatIn;
    cap.release();
}
