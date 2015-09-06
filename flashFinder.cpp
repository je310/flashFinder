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
Mat findAvOfVid(string fileName,float decimation);
vector<Point> findCodedness(vector<Mat> corrBuffer,string winName, float threshold, vector<float> codeSeries);

void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void drawGraph(vector<float> corrSeries, string WinName);

template<unsigned int N> vector<float> corrCode(string bitstring, int derating);

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
    const int lengthOfCode = 8;
    const int derating = 2;           //this is the factor slow down that Oscar suggested.
    const float decimation = 1;      //the amount the image is resized, makes performance better.
    const float secondsToProcess = 2;
    const float FPSCamera = 118.4;
    string fileName = "Videos/glowFlash.mp4";
    //string fileName = "fakeVideos/video.mp4";



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
        vector<Mat> thisCorrelation = autoCorrelate(imageBuffer, i);
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
    cout<< minSaved << "   "<< maxSaved << endl;

    //haveALook(lengthOfBuffers,corrBuffer,imageBuffer,av, derating);

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
    setMouseCallback("heatMap", CallBackFunc, &clickLocation);
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
            vector<float> code = corrCode<8>("11010100", derating);
            drawGraph(code, "the code");
            findCodedness(corrBuffer, "heatMap", 0.1,code );
        }
        if(clickLocation.x != clickLocationOld.x || clickLocation.y != clickLocationOld.y){
            for(int i =0; i < corrBuffer.size(); i++){
                corrSeries.at(i) = corrBuffer.at(i).at<float>(clickLocation.y,clickLocation.x);
            }
            drawGraph(corrSeries, "inspectSeries");
            clickLocationOld = clickLocation;
        }
        if(k == 'q') break;

        imshow("corrBuffer",corrBuffer.at(myWin));
        imshow("averageImage",av/255);
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
        if (height < 0){
            height = 0;
        }
        rio =Rect(start, height, secSize, sheetSize-height);
        //cout << rio.x << " " << rio.y << " " << rio.width<< " "  << rio.height <<" "<< graph.size() << endl;
        graph(rio) = 1.0;
    }
    cout <<endl;
    imshow(WinName, graph);
}

vector<Point> findCodedness(vector<Mat> corrBuffer,string winName, float threshold, vector<float> codeSeries){
    namedWindow(winName,WINDOW_NORMAL );
    Mat heatMap(corrBuffer.at(0).size(),CV_32FC1,0.0);
    vector<Point> hotSpots;
    int rows = heatMap.rows;
    int cols = heatMap.cols;
    for(int i = 0; i < cols; i++){
        for(int j=0; j< rows; j++){
            float sum  = 0;
            for(int k = 0;  k < corrBuffer.size(); k++){
                float val =  (corrBuffer.at(k).at<float>(j,i)) - codeSeries.at(k);
                sum += val * val;
            }
            heatMap.at<float>(j,i) = sum;
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
    heatMap = (heatMap - min)/(max - min);
    Mat heatInverse(heatMap.size(),CV_32FC1,1.0);
    heatInverse -= heatMap;
    imshow(winName, heatInverse);
    return hotSpots;
}


template<unsigned int N>
vector<float> corrCode(string bitstring, int derating){
	bitset<N> bits(bitstring);
	vector<float> input(bits.size()*derating);
	vector<float> output(input.size(), 0);
	
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
