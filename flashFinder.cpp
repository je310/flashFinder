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

void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void drawGraph(vector<float> corrSeries, string WinName);

struct getFrameFunctor{
	float _decimation;
	
	getFrameFunctor(float decimation):_decimation(decimation){}

	Mat operator() (VideoCapture &cap){
		Mat frameIn;                //The newest frame from the stream
		Mat frameSmall;             // A compressed version of the image
		Mat frameGrey;              // gray scale image.
		Mat frameFloat;             // float image that is the 'base' of the operation

		cap >> frameIn;
		resize(frameIn,frameSmall,Size(0,0),_decimation,_decimation);
		cv::cvtColor(frameSmall, frameGrey, CV_BGR2GRAY);
		frameGrey.convertTo(frameFloat,CV_32FC1);
		
		return frameFloat;
	}
};


int main(){
    //parameters worth changing.
    int lengthOfCode = 8;
    int derating = 2;           //this is the factor slow down that Oscar suggested.
    float decimation = 0.1;      //the amount the image is resized, makes performance better.
    float secondsToProcess = 2;
    float FPSCamera = 118.4;
    string fileName = "Videos/slowerFlash.mp4";

    //derived less interesting variables;
    int numberToDo = int(FPSCamera * secondsToProcess);
    int lengthOfBuffers = lengthOfCode * derating;
    cout << "finding frame average"<<endl;
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
    vector<Mat> imageBuffer;    //buffer with grey images
    vector<Mat> corrBuffer;     //buffer with the corelation accumilation.

	getFrameFunctor getFrame(decimation);

    //initialise our buffers.
    for(int i = 0; i <lengthOfBuffers; i++){
		Mat frame = getFrame(cap);
        imageBuffer.push_back(Mat(frame.size(),CV_32FC1, 0.0));
        imageBuffer.at(i) = frame-av;
        //frame.copyTo(imageBuffer.at(i));
        corrBuffer.push_back(Mat(frame.size(),CV_32FC1, 0.0));
    }


    cout<< "finished initialising buffers"<<endl;
    //process the whole video here.

    for(int i = lengthOfBuffers; i < numberToDo; i++){          //for every frame to be processed.
        vector<Mat> thisCorrelation = autoCorrelate(imageBuffer,i%lengthOfBuffers);
        for(int j = 0; j< lengthOfBuffers; j++){                    // accumilate the correlation in all bins at each time step forwards.
            corrBuffer.at(j) = corrBuffer.at(j) +thisCorrelation.at(j);
        }
		Mat frame = getFrame(cap);
        imageBuffer.at(i%lengthOfBuffers) = frame-av;
        //frame.copyTo(imageBuffer.at(i%lengthOfBuffers));

        cout<<i<<" "<<flush;
        if (i%20 == 0) cout<<endl;
    }

    Mat averageCorr(corrBuffer.at(0).size(),CV_32FC1, 0.0);
    for(int i = 0; i< lengthOfBuffers; i++){
        averageCorr += corrBuffer.at(i);
    }
    averageCorr /= lengthOfBuffers;
    for(int i = 0; i< lengthOfBuffers; i++){
        corrBuffer.at(i) -= averageCorr;
    }

    double min, max, minSaved, maxSaved;
    minSaved = 1e60;
    maxSaved = -1e60;
    for(int i = 0; i < lengthOfBuffers; i++){
        minMaxLoc(corrBuffer.at(i), &min, &max);
        if(min<minSaved) minSaved = min;
        if(max>maxSaved) maxSaved = max;
    }
    cout<< minSaved << "   "<< maxSaved << endl;
    for(int i = 0; i < lengthOfBuffers; i++){
        corrBuffer.at(i) -= minSaved;
        corrBuffer.at(i) *= 1/(maxSaved-minSaved);
    }
//    for(int i = 0; i < lengthOfBuffers; i++){
//        //corrBuffer.at(i) += 1;
//        corrBuffer.at(i) *= 0.5;
//    }
    minSaved = 1e60;
    maxSaved = -1e60;
    for(int i = 0; i < lengthOfBuffers; i++){
        minMaxLoc(corrBuffer.at(i), &min, &max);
        if(min<minSaved) minSaved = min;
        if(max>maxSaved) maxSaved = max;
    }
    cout<< minSaved << "   "<< maxSaved << endl;

    haveALook(lengthOfBuffers,corrBuffer,imageBuffer);

	return 0;
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
        }
        corrResult.push_back(Mat(input.at(0).size(),CV_32FC1,0.0));
        intermediateCorr.copyTo(corrResult.at(i));

    }
    return corrResult;
}
void haveALook(int lengthOfBuffers, vector<Mat> corrBuffer, vector<Mat> imageBuffer){
    namedWindow("corrBuffer",WINDOW_NORMAL );
    namedWindow("imageBuffer",WINDOW_NORMAL );
    Point clickLocation;
    Point clickLocationOld;
    clickLocationOld.x = 0;
    clickLocationOld.y = 0;
    vector<float> corrSeries;
    for(int i =0; i < corrBuffer.size(); i++){
        corrSeries.push_back(0.0);
    }
    setMouseCallback("corrBuffer", CallBackFunc, &clickLocation);
    setMouseCallback("imageBuffer", CallBackFunc, &clickLocation);
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
        }
        if(clickLocation.x != clickLocationOld.x || clickLocation.y != clickLocationOld.y){
            for(int i =0; i < corrBuffer.size(); i++){
                corrSeries.at(i) = corrBuffer.at(i).at<float>(clickLocation.x,clickLocation.y);
            }
            drawGraph(corrSeries, "inspectSeries");
            clickLocationOld = clickLocation;
        }
        if(k == 'q') break;
        imshow("corrBuffer",corrBuffer.at(myWin));
        imshow("imageBuffer",imageBuffer.at(myWin)/255);
    }
    return;
}

Mat findAvOfVid(string fileName,float decimation){
    VideoCapture cap(fileName.c_str());
	getFrameFunctor getFrame(decimation);

    int NoOfFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    float FPS = cap.get(CV_CAP_PROP_FPS);
    cout<<"The loaded file has "<< NoOfFrames << " frames. These were recorded at "<< FPS<<" FPS."<< endl;

    Mat frame = getFrame(cap);
    Mat averageFloatIn(frame.size(),CV_32FC1,0.0);
    frame.copyTo(averageFloatIn);
    for(int i = 1; i < NoOfFrames; i++){
		frame = getFrame(cap);
        averageFloatIn += frame;
    }
    averageFloatIn = averageFloatIn/NoOfFrames;
    return averageFloatIn;
    cap.release();
}

void CallBackFunc(int event, int x, int y, int flags, void* clickLocation){
    Point* thisLocation = (Point*) clickLocation;
     if  ( event == EVENT_LBUTTONDOWN ){
          cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
          thisLocation->x = x;
          thisLocation->y = y;
     }
     else if  ( event == EVENT_RBUTTONDOWN ){
          cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
     }
     else if  ( event == EVENT_MBUTTONDOWN ){
          cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
     }
     else if ( event == EVENT_MOUSEMOVE ){


     }
}

void drawGraph(vector<float> corrSeries, string WinName){
    namedWindow(WinName,WINDOW_NORMAL );
    int sheetSize = 256;
    Mat graph(sheetSize,sheetSize,CV_32FC1,0.0);
    int numX = corrSeries.size();
    int secSize  = (int)(sheetSize/numX);
    Rect rio;
    for(int i = 0; i < numX; i++){
        int height = (int)((1-(corrSeries.at(i)))*sheetSize)-1;
        int start = i*secSize;
        std::cout.precision(2);
        cout << corrSeries.at(i)<<" ";
        rio =Rect(start, height, secSize, sheetSize-height);
        //cout << rio.x << " " << rio.y << " " << rio.width<< " "  << rio.height <<" "<< graph.size() << endl;
        //graph(rio) = 1.0;
    }
    cout <<endl;
    imshow(WinName, graph);
}
