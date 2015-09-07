//includes
#include <bitset>
#include <iostream>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include <unistd.h>

//namespaces
using namespace std;
using namespace cv;


//fuction declerations
vector<Mat> autoCorrelate (vector<Mat> input, size_t firstframe);
void haveALook(int lengthOfBuffers, vector<Mat> corrBuffer, vector<Mat> imageBuffer,Mat av, const int derating);
Mat findAvOfVid(string fileName,double decimation,int periodsToAverage,int lengthOfBuffers);
vector<Point> findCodedness(vector<Mat> corrBuffer,string winName, double threshold, vector<double> codeSeries, bool veryCorr,double factorHigh, double factorLow, vector<Mat> &maxedCorr, Point clickedLocationOld);

void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void drawGraph(vector<double> corrSeries, string WinName);

template<unsigned int N> vector<double> corrCode(string bitstring, int derating);

struct getFrameFunctor{
    double _decimation;

    getFrameFunctor(double decimation):_decimation(decimation){}

    Mat operator() (VideoCapture &cap){
        Mat frameIn;                //The newest frame from the stream
        Mat frameSmall;             // A compressed version of the image
        Mat frameGrey;              // gray scale image.
        Mat framedouble;             // double image that is the 'base' of the operation

        cap >> frameIn;
        resize(frameIn,frameSmall,Size(0,0),_decimation,_decimation);
        cv::cvtColor(frameSmall, frameGrey, CV_BGR2GRAY);
        frameGrey.convertTo(framedouble,CV_64FC1);

        return framedouble;
    }
};


int main(){
    //parameters worth changing.
    const int lengthOfCode = 8;
    const int derating = 2;           //this is the factor slow down that Oscar suggested.
    const double decimation = 0.1;      //the amount the image is resized, makes performance better.
    const double secondsToProcess = 2.5;
    const double FPSCamera = 118.4;
    const int periodsToAverage = 8;
    //string fileName = "Videos/glowFlash.mp4";
    //string fileName = "fakeVideos/video.mp4";
    //string fileName = "Videos/fasterFlash.mp4";
    string fileName = "Videos/slowerFlash.mp4";
    //string fileName = "Videos/longglowFlash.mp4";




    //derived less interesting variables;
    int numberToDo = int(FPSCamera * secondsToProcess);
    int lengthOfBuffers = lengthOfCode * derating;
    cout << "finding frame average"<<endl;
    Mat av = findAvOfVid(fileName,decimation,periodsToAverage,lengthOfBuffers);
    //Mat av = findAvOfVid(fileName,decimation,0,lengthOfBuffers);
    //open file
    VideoCapture cap(fileName.c_str());
    if(!cap.isOpened()) {
        cout << "we did not open the file/camera correctly"<<endl;
        return -1; // check if we succeeded
    }
    int NoOfFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    double FPS = cap.get(CV_CAP_PROP_FPS);
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
        imageBuffer.push_back(Mat(frame.size(),CV_64FC1, 0.0));
        imageBuffer.at(i) = frame-av;
        //frame.copyTo(imageBuffer.at(i));
        corrBuffer.push_back(Mat(frame.size(),CV_64FC1, 0.0));
    }


    cout<< "finished initialising buffers"<<endl;
    //process the whole video here.

    for(int i = lengthOfBuffers; i < numberToDo; i++){          //for every frame to be processed.
        vector<Mat> thisCorrelation = autoCorrelate(imageBuffer, i);
        for(int j = 0; j< lengthOfBuffers; j++){                    // accumilate the correlation in all bins at each time step forwards.
            corrBuffer.at(j) = corrBuffer.at(j) +thisCorrelation.at(j);
        }
        Mat frame = getFrame(cap);
        imageBuffer.at(i%lengthOfBuffers) = frame-av;
        //frame.copyTo(imageBuffer.at(i%lengthOfBuffers));

        if (i%20 == 0) cout<<i<<" "<<flush;
    }
    cout <<endl;

    Mat averageCorr(corrBuffer.at(0).size(),CV_64FC1, 0.0);
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
    cout<< minSaved << "   "<< maxSaved << endl;

    haveALook(lengthOfBuffers,corrBuffer,imageBuffer,av, derating);

    return 0;
}

//function implementations

vector<Mat> autoCorrelate (vector<Mat> input, size_t firstframe){      //the ofset aims to allow a circular buffer use of the vector.
    vector<Mat> corrResult(input.size());
    int sz = input.size();

#pragma omp parallel for
    for(int j = 0; j < input.size(); j++){                                  //for each pair of images that align this time
        Mat a = input.at((firstframe  )%sz);
        Mat b = input.at((firstframe+j)%sz);
        Mat c = a.mul(b);
        corrResult.at(j)=c;
    }

    return corrResult;
}
void haveALook(int lengthOfBuffers, vector<Mat> corrBuffer, vector<Mat> imageBuffer,Mat av, const int derating){
    namedWindow("corrBuffer",WINDOW_NORMAL );
    namedWindow("averageImage",WINDOW_NORMAL );
    namedWindow("heatMap",WINDOW_NORMAL );
    namedWindow("heatMap2",WINDOW_NORMAL );
    static bool veryCor = 0;
    static double factorHigh = 0.05;
    static double factorLow = 0.025;
    Point clickLocation;
    Point clickLocationOld;
    clickLocationOld.x = 0;
    clickLocationOld.y = 0;
    vector<double> corrSeries;
    vector<double> corrSeries2;
    vector<Mat> maxedCorr;
    for(int i =0; i < corrBuffer.size(); i++){
            maxedCorr.push_back(Mat(corrBuffer.at(0).size(),CV_64FC1,0.0));
    }
    for(int i =0; i < corrBuffer.size(); i++){
        corrSeries.push_back(0.0);
        corrSeries2.push_back(0.0);
    }
    setMouseCallback("corrBuffer", CallBackFunc, &clickLocation);
    setMouseCallback("imageBuffer", CallBackFunc, &clickLocation);
    setMouseCallback("heatMap", CallBackFunc, &clickLocation);
    setMouseCallback("heatMap2", CallBackFunc, &clickLocation);
    setMouseCallback("averageImage", CallBackFunc, &clickLocation);
    while(1){
        char k =waitKey(1);
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
        if(k == 'h'){
            vector<double> code = corrCode<8>("11110000", derating);
            drawGraph(code, "the code");
            findCodedness(corrBuffer, "heatMap", 0.1,code ,veryCor,factorHigh,factorLow,maxedCorr,clickLocationOld);
        }
        if(k == '='){
            factorHigh +=0.005;
            cout << "displaying only the best:"<< factorHigh<<endl;
        }
        if(k == '-'){
            factorHigh -=0.005;
            cout << "displaying only the best:"<< factorHigh<<endl;
        }
        if(k == '0'){
            factorLow +=0.005;
            cout << "displaying only the worst:"<< factorLow<<endl;
        }
        if(k == '9'){
            factorLow -=0.005;
            cout << "displaying only the worst:"<< factorLow<<endl;
        }
        if(clickLocation.x != clickLocationOld.x || clickLocation.y != clickLocationOld.y){
            for(int i =0; i < corrBuffer.size(); i++){
                corrSeries.at(i) = corrBuffer.at(i).at<double>(clickLocation.y,clickLocation.x);
            }
            for(int i =0; i < corrBuffer.size(); i++){
                corrSeries2.at(i) = maxedCorr.at(i).at<double>(clickLocation.y,clickLocation.x);
            }
            drawGraph(corrSeries, "inspectSeries");
            drawGraph(corrSeries2, "maxed");
            clickLocationOld = clickLocation;
        }
        if(k == 'q') break;
        if(k == 'd'){
            veryCor = ! veryCor;
        }
        imshow("corrBuffer",corrBuffer.at(myWin));
        imshow("averageImage",av/255);
    }
    return;
}

Mat findAvOfVid(string fileName,double decimation,int periodsToAverage,int lengthOfBuffers){
    VideoCapture cap(fileName.c_str());
    getFrameFunctor getFrame(decimation);

    int NoOfFrames = periodsToAverage*lengthOfBuffers;
    if(periodsToAverage == 0){
        NoOfFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    }

    Mat frame = getFrame(cap);
    Mat averagedoubleIn(frame.size(),CV_64FC1,0.0);
    frame.copyTo(averagedoubleIn);
    for(int i = 1; i < NoOfFrames; i++){
        frame = getFrame(cap);
        averagedoubleIn += frame;
    }
    averagedoubleIn = averagedoubleIn/NoOfFrames;
    return averagedoubleIn;
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

void drawGraph(vector<double> corrSeries, string WinName){
    namedWindow(WinName,WINDOW_NORMAL );
    int sheetSize = 256;
    Mat graph(sheetSize,sheetSize,CV_64FC1,0.0);
    int numX = corrSeries.size();
    int secSize  = (int)(sheetSize/numX);
    Rect rio;
    for(int i = 0; i < numX; i++){
        int height = (int)((1-(corrSeries.at(i)))*sheetSize)-1;
        int start = i*secSize;
        std::cout.precision(6);
        cout << corrSeries.at(i)<<" ";
        if (height < 0){
            height = 0;
        }
        rio =Rect(start, height, secSize, sheetSize-height);
        //cout << rio.x << " " << rio.y << " " << rio.width<< " "  << rio.height <<" "<< graph.size() << endl;
            //graph(rio) = 1.0;
            rectangle(graph, rio, Scalar(1));
    }
    cout <<endl;
    imshow(WinName, graph);
}

vector<Point> findCodedness(vector<Mat> corrBuffer,string winName, double threshold, vector<double> codeSeries, bool veryCorr,double factorHigh, double factorLow, vector<Mat>& maxedCorr, Point clickedLocationOld){
    namedWindow(winName,WINDOW_NORMAL );
    namedWindow(winName+"2",WINDOW_NORMAL );
    Mat heatMap(corrBuffer.at(0).size(),CV_64FC1,0.0);
    Mat heatMapNorm(corrBuffer.at(0).size(),CV_64FC1,0.0);
    vector<Point> hotSpots;
    int rows = heatMap.rows;
    int cols = heatMap.cols;
    for(int i = 0; i < cols; i++){
        for(int j=0; j< rows; j++){
            double sum  = 0;
            double sum2  = 0;
            vector<double> corrSeries;
            for(int k = 0;  k < corrBuffer.size(); k++){
                corrSeries.push_back((corrBuffer.at(k).at<double>(j,i)));
            }

            double minCorr = 10e8;
            double maxCorr = -10e9;
             for(int p = 0; p <corrBuffer.size(); p++){
                 if(corrSeries.at(p)>maxCorr){
                     maxCorr = corrSeries.at(p);
                 }
                 if(corrSeries.at(p)<minCorr){
                     minCorr = corrSeries.at(p);
                 }
             }
            for(int k = 0;  k < corrBuffer.size(); k++){

                corrSeries.at(k) = (corrSeries.at(k) - minCorr)/(maxCorr - minCorr);

            }
            for(int k = 0;  k < corrBuffer.size(); k++){
                maxedCorr.at(k).at<double>(j,i) = corrSeries.at(k) ;
            }


            for(int k = 0;  k < corrBuffer.size(); k++){
                double val = corrSeries.at(k) - codeSeries.at(k);
                double val2 = corrBuffer.at(k).at<double>(j,i) - codeSeries.at(k);
                sum += val * val;
                sum2 += val2 * val2;
            }
            heatMap.at<double>(j,i) = (sum);
            heatMapNorm.at<double>(j,i) =sum2;
            if(sum > threshold){
                Point thisHotspot;
                thisHotspot.x = i;
                thisHotspot.y = j;
                hotSpots.push_back(thisHotspot);
            }
        }
    }


    double min,max;
    minMaxLoc(heatMap, &min, &max);
    double diff = max - min;
    if(veryCorr){
        max =  min + (factorHigh*diff);
        min =  max - (factorLow*diff);
        for(int i = 0; i < cols; i++){
            for(int j=0; j< rows; j++){
                if (heatMap.at<double>(j,i) > max){
                    heatMap.at<double>(j,i) =  max;
                }
                if (heatMap.at<double>(j,i) < min){
                    heatMap.at<double>(j,i) =  min;
                }
            }
        }

    }

    heatMap = (heatMap - min)/(max - min);
    min = 10e10;
    max = -10e10;
    minMaxLoc(heatMapNorm, &min, &max);
    cout << min << "        "<< max << endl;
    if(veryCorr){
        max =  min + (factorHigh*diff);
        min =  max - (factorLow*diff);
        for(int i = 0; i < cols; i++){
            for(int j=0; j< rows; j++){
                if (heatMapNorm.at<double>(j,i) > max){
                    heatMapNorm.at<double>(j,i) =  max;
                }
                if (heatMapNorm.at<double>(j,i) < min){
                    heatMapNorm.at<double>(j,i) =  min;
                }
            }
        }

    }
    heatMapNorm = (heatMapNorm - min)/(max - min);
    Mat heatInverse(heatMap.size(),CV_64FC1,1.0);
    Mat heatInverse2(heatMapNorm.size(),CV_64FC1,1.0);
    heatInverse -= heatMap;
    heatInverse2 -= heatMapNorm;
    cout<<"heatmap value is:"<< heatMap.at<double>(clickedLocationOld)<< endl;
    cout<<"heatmapNorm value is:"<< heatMapNorm.at<double>(clickedLocationOld)<< endl;
    imshow(winName, heatInverse);
    imshow(winName+"2", heatInverse2);
    return hotSpots;
}


template<unsigned int N>
vector<double> corrCode(string bitstring, int derating){
    bitset<N> bits(bitstring);
    vector<double> input(bits.size()*derating);
    vector<double> output(input.size(), 0);

    for(size_t i = 0; i != input.size(); i++)
        input[i] = bits[i/derating];

    for(size_t i = 0; i != input.size(); i++){
        for(size_t j = 0; j!= input.size(); j++){
            output[i] += input[j]*input[(j+i)%input.size()];
        }
    }

    for(size_t i = input.size(); i !=0; i--)
        output[i-1] /= output[0];

    return output;
}
