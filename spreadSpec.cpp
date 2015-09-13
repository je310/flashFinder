//includes
#include <bitset>
#include <complex>
#include <iostream>
#include <vector>

#include <cv.h>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <highgui.h>
#include <unistd.h>

#include <fftw3.h>

//namespaces
using namespace std;
using namespace cv;

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
        cvtColor(frameSmall, frameGrey, CV_BGR2GRAY);
        frameGrey.convertTo(framedouble,CV_64FC1);

        return framedouble;
    }
};


struct crosscorrstruct{
	Mat Amplitude;
	Mat Phase;
};

template <unsigned int N> vector<double> spreadGenerator(vector <size_t> XorFrom, int derating);
crosscorrstruct crossCorr(vector <Mat> imageBuffer, vector<double> spreadcode);
void CallBackFunc(int event, int x, int y, int flags, void* clickLocation);


int main(){
    //parameters worth changing.
    const int bits = 7;
    const vector<size_t> XorFrom = {7,6};
    const int lengthOfCode = pow(2,bits)-1;
    const int derating = 2;           //this is the factor slow down that Oscar suggested.

    const double decimation = 1;      //the amount the image is resized, makes performance better.
    const double secondsToProcess = 21;

    const double FPSCamera = 118.4;
    bool useAll = 1;
    //string fileName = "Videos/glowFlash.mp4";
    //string fileName = "fakeVideos/video.mp4";
    //string fileName = "Videos/fasterFlash.mp4";
    //string fileName = "Videos/slowerFlash.mp4";
    //string fileName = "Videos/longglowFlash.mp4";
    //string fileName = "Videos/Spreadcode.mp4";
    //string fileName = "Videos/longOutsideFlash.mp4";
    //string fileName = "Videos/3OutsideFlash.mp4";
    //string fileName = "Videos/1OutsideFlash.mp4";
    //string fileName = "Videos/insideFlash.mp4";
    //string fileName = "Videos/fajitaFlash.mp4";
    //string fileName = "Videos/glassFlash.mp4";
    //string fileName = "Videos/middleFlash.mp4";
    string fileName = "Videos/farLongerFlash.mp4";

    //derived less interesting variables;
    int numberToDo = int(FPSCamera * secondsToProcess);

    int lengthOfBuffers = lengthOfCode * derating;


    //open file
    VideoCapture cap(fileName.c_str());
    if(!cap.isOpened()) {
        cout << "we did not open the file/camera correctly"<<endl;
        return -1; // check if we succeeded
    }
    int NoOfFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);
    if (useAll) numberToDo = NoOfFrames;
    numberToDo = int(numberToDo / lengthOfBuffers)* lengthOfBuffers;

    double FPS = cap.get(CV_CAP_PROP_FPS);
    cout<<"The loaded file has "<< NoOfFrames << " frames. These were recorded at "<< FPS<<" FPS."<< endl;
    if(NoOfFrames < numberToDo){
        cout<< "this video is not of the required length"<< endl;
        return -1;
    }
    cout <<"set up video stream"<<endl;
    //declare variables


    cv::vector <double> spreadcode = spreadGenerator<bits>(XorFrom, derating);
    for(int i = 0; i < spreadcode.size(); i= i+2){
        cout << spreadcode.at(i);
    }
    cout<<endl;

    getFrameFunctor getFrame(decimation);
    vector<Mat> imageBuffer(lengthOfBuffers);    //buffer with grey images
    cout << numberToDo<< endl;
    for(int i = 0; i < lengthOfBuffers; i++){
        imageBuffer.at(i) = getFrame(cap);
    }
    namedWindow("Image", WINDOW_NORMAL);
    imshow("Image", imageBuffer.at(0)/255);
    namedWindow("Maxed Phase", WINDOW_NORMAL);
    cout<< "here"<< endl;
    for(int i =lengthOfBuffers; i < numberToDo; i++){
        imageBuffer.at(i%lengthOfBuffers) += getFrame(cap);
    }

	crosscorrstruct results = crossCorr(imageBuffer, spreadcode);
	
	double Amin,Amax;
    double AminAPhase,AmaxAPhase;
	int x,y;
    int xAPhase,yAPhase;
//    for(int i = 0; i< results.Amplitude.cols; i++){
//        for(int j = 0; j < results.Amplitude.rows; j++){
//            results.Amplitude.at<double>(j,i) = log(results.Amplitude.at<double>(j,i)+1);
//        }
//    }
    minMaxIdx(results.Amplitude, &Amin, &Amax, &x, &y);

	cout << " max: " << Amax << " maxframe: " << results.Phase.at<double>(y,x) << endl;

    minMaxIdx(imageBuffer.at(results.Phase.at<double>(y,x)), &AminAPhase, &AmaxAPhase, &xAPhase, &yAPhase);
    imshow("Maxed Phase", (imageBuffer.at(results.Phase.at<double>(y,x))-AminAPhase)/(AmaxAPhase - AminAPhase));
    Mat total = Mat(imageBuffer.at(0).size(), CV_64FC1,0.0);
    Mat peak = Mat(imageBuffer.at(0).size(), CV_64FC1,0.0);
//#pragma omp parallel for
    for(int i = 0; i < lengthOfBuffers; i++){
        Mat square= Mat(imageBuffer.at(0).size(), CV_64FC1,0.0);
        multiply(imageBuffer.at(i),imageBuffer.at(i),square);
        total = total + square;
    }

    for(int i = 0; i < lengthOfBuffers; i++){
        for(int j = 0; j < imageBuffer.at(0).cols;  j ++){
            for(int k = 0; k < imageBuffer.at(0).rows; k++){
                if(imageBuffer.at(i).at<double>(k,j)>peak.at<double>(k,j)){
                    peak.at<double>(k,j) = imageBuffer.at(i).at<double>(k,j);
                }
            }
        }
    }
    Mat ratio;
    multiply(peak,peak,peak);
    divide(peak, total, ratio);
    normalize(ratio,ratio,0,1,NORM_MINMAX);
    namedWindow("added buffers", WINDOW_NORMAL);
    imshow("added buffers", ratio);
	
	Mat phase;
	results.Phase.convertTo(phase, CV_8UC1);
	applyColorMap(phase/lengthOfBuffers*255.0, phase, COLORMAP_HSV);
	phase.convertTo(phase,CV_64FC3);
	phase /= 255;

	Mat amplitude = (results.Amplitude-Amin)/(Amax-Amin);
    multiply(amplitude, ratio,amplitude);
    normalize(amplitude,amplitude,0,1,NORM_MINMAX);
	amplitude.convertTo(amplitude, CV_32FC1);
	cvtColor(amplitude, amplitude, CV_GRAY2BGR);
	amplitude.convertTo(amplitude, CV_64FC3);


	namedWindow("Map", WINDOW_NORMAL);	
    Point clickLocation;
    setMouseCallback("Map", CallBackFunc, &clickLocation);
	imshow("Map", phase.mul(amplitude));//(results.Amplitude-Amin)/(Amax-Amin));
    Mat MaxedPhase(imageBuffer.at(0).size(), CV_64FC1,0.0);
    for (char k = '\0'; k != 'q'; k = waitKey(1)){
        double thisMin = 10e10;
        double thisMax = -thisMin;
        int minIdx = 0;
        int maxIdx = 0;
        for(int i = 0; i < lengthOfBuffers; i++){
            if(imageBuffer.at(i).at<double>(clickLocation)> thisMax){
                thisMax = imageBuffer.at(i).at<double>(clickLocation);
                maxIdx = i;
            }
            if(imageBuffer.at(i).at<double>(clickLocation)< thisMin){
                thisMin = imageBuffer.at(i).at<double>(clickLocation);
                minIdx =i;
            }
        }
        cout<< "max bin value here:"<< thisMax<< endl;
        cout<< "after normalisation:"<< amplitude.at<double>(clickLocation);
        minMaxIdx(imageBuffer.at(maxIdx), &AminAPhase, &AmaxAPhase, &xAPhase, &yAPhase);
        MaxedPhase = (imageBuffer.at(maxIdx)-AminAPhase)/(AmaxAPhase - AminAPhase);
//        for(int i = 0; i < imageBuffer.at(maxIdx).rows; i++){
//            for(int j= 0; j < imageBuffer.at(maxIdx).cols; j++){
//                MaxedPhase.at<double>(i,j) = log(MaxedPhase.at<double>(i,j)+1);
//            }
//        }
//        multiply(MaxedPhase,ratio,MaxedPhase);
//        double LogMin = 10e10;
//        double LogMax = -LogMin;
//        minMaxIdx(MaxedPhase, &LogMin, &LogMax, &xAPhase, &yAPhase);
        minMaxIdx(imageBuffer.at(maxIdx), &AminAPhase, &AmaxAPhase, &xAPhase, &yAPhase);
        imshow("Maxed Phase", (imageBuffer.at(maxIdx)-AminAPhase)/(AmaxAPhase - AminAPhase));
//        imshow("Loged Maxed Phase", ((MaxedPhase-LogMin)/(LogMax - LogMin)));
    }

	return 0;
}

//function implementations

crosscorrstruct crossCorr(vector <Mat> imageBuffer, vector<double> spreadcode){
	// Fourier Transform Spreadcode
	fftw_complex *spreadcodeftin, *spreadcodeftout;
	spreadcodeftin  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * spreadcode.size());
	spreadcodeftout = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * spreadcode.size());

	fftw_plan p;
#pragma omp parallel for
    for(size_t i = 0; i < spreadcode.size(); i++){
        spreadcodeftin[i][0] = spreadcode.at(i);
        spreadcodeftin[i][1] = 0;
    }
	p = fftw_plan_dft_1d( spreadcode.size()
						, spreadcodeftin, spreadcodeftout
						, FFTW_FORWARD, FFTW_ESTIMATE
						);
	fftw_execute(p); 
	spreadcodeftout[0][0] = 0; // set DC offset to 0
	spreadcodeftout[0][1] = 0;
	fftw_destroy_plan(p);
	fftw_free(spreadcodeftin); 
	
	crosscorrstruct out;
	out.Amplitude = Mat::zeros(imageBuffer.at(0).size(), imageBuffer.at(0).type());
	out.Phase     = Mat::zeros(imageBuffer.at(0).size(), imageBuffer.at(0).type());

	// Cross correlate

#pragma omp parallel for
    for(int x = 0; x < imageBuffer.at(0).rows; x++){
        fftw_complex *pixelsftin, *pixelsftout;
        pixelsftin  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * imageBuffer.size());
        pixelsftout = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * imageBuffer.size());
        fftw_plan(forward);
        forward = fftw_plan_dft_1d( imageBuffer.size()
                                  , pixelsftin, pixelsftout
                                  , FFTW_FORWARD, FFTW_ESTIMATE
                                  );
        fftw_plan(backward);
        backward= fftw_plan_dft_1d( imageBuffer.size()
                                  , pixelsftin, pixelsftout
                                  , FFTW_BACKWARD, FFTW_ESTIMATE
                                  );
        for(int y = 0; y < imageBuffer.at(0).cols; y++){
//#pragma omp parallel for
            for(int i = 0; i < imageBuffer.size(); ++i){
                pixelsftin[i][0] = imageBuffer.at(i).at<double>(x,y)/255.0;
                pixelsftin[i][1] = 0;
            }

			fftw_execute(forward);
//#pragma omp parallel for
            for (int j = 0; j < imageBuffer.size(); ++j ){
                complex<double> p (    pixelsftout[j][0], -pixelsftout    [j][1]);
                complex<double> c (spreadcodeftout[j][0],  spreadcodeftout[j][1]);
                complex<double> o = p*c;
                pixelsftin[j][0] = o.real();
                pixelsftin[j][1] = o.imag();
            }

			fftw_execute(backward);

            for(int i = 0; i < imageBuffer.size(); ++i){
                imageBuffer.at(i).at<double>(x,y) = pixelsftout[i][0];
                if (pixelsftout[i][0] > out.Amplitude.at<double>(x,y)){
                    out.Amplitude.at<double>(x,y) = pixelsftout[i][0];
                    out.Phase.at<double>(x,y) = i;
                }
            }
        }
//        fftw_destroy_plan(forward);
//        fftw_destroy_plan(backward);
        fftw_free(pixelsftin);
        fftw_free(pixelsftout);
    }


	fftw_free(spreadcodeftout);

	
	return out;
}

template <unsigned int N>
vector<double> spreadGenerator(vector <size_t> XorFrom, int derating) {
     vector<double> out;
     bitset<N> initialState = 1;
     bitset<N> lsfr = initialState;

     for(size_t x = 0; x!= XorFrom.size(); x++)
         XorFrom[x] -= 1;

     do {
         bool bit = lsfr[XorFrom[0]];
         for ( size_t x = 1
             ; x!= XorFrom.size()
             ; x++
             )
            bit ^= lsfr[XorFrom[x]];
         lsfr <<= 1;
         lsfr[0] = bit;

         for (int d = 0; d != derating; d++)
			 out.push_back(bit);
     } while(initialState != lsfr);

     return out;
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

