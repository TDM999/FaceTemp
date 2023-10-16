//============================================================================
// Name        : FaceTemp.cpp
// Author      : Adrián Cordones Martínez
// Version     : 1.0
// Description : Program created for my 'FDP' on my Physics grade in the
//               University of Seville (US). Type "./FaceTemp --help" for more
//               information about this program.
//============================================================================

#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <string>
#include <sstream>

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/freetype.hpp"

// OpenCV external UI (CVUI):
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#include "wiringPi.h"
#include "Calib.hpp"

using namespace std;
using namespace cv;

// Custom namespace (from "Calib.hpp") for camera calibration:
using namespace cd;
// Custom class (from "Calib.hpp") for assigning Cameras IDs:
CameraDetector CD;

// Error values:
#define       CASCADE_ERROR   -1 // Could not load classifier cascade.
#define     NOIRIMAGE_ERROR   -2 // Could not read IR image.
#define   NOIRCAPTURE_ERROR   -3 // Capture from IR camera didn't work.
#define    NOVISIMAGE_ERROR   -4 // Could not read Visible image.
#define  NOVISCAPTURE_ERROR   -5 // Capture from Visible camera didn't work.
#define        ORIGIN_ERROR   -6 // Images must have the same origin (capturing from cameras or external images).
#define      EMPTYVIS_ERROR   -7 // Empty image in the visible range when capturing from camera.
#define       EMPTYIR_ERROR   -8 // Empty image in the IR range when capturing from camera.

// Temperature-IR calibration values:
// This calibration may change due to ambient temperature.
#define T1  36.6
#define T2  36.9
#define R1  8418.8
#define R2  11348.92
#define IRSensorTo8Bit  16383.0/255

// Temperature bounds:
#define LOWER_TEMP_THRES      37.1 //Slight fever
#define HIGHER_TEMP_THRES     37.5 //Fever

// Remapping factors:
#define scaleFactor  3.27
#define     xFactor  112.2
#define     yFactor  34.1

// Use "./FaceTemp --help" for more information about this program.
static void help(const char** argv)
{
    cout << "\nThe main purpose of this program is detecting faces and"
        " measuring their mean temperature using OpenCV library.\n"
        "This program uses the cv::CascadeClassifier class to"
        " detect objects (faces and eyes) using Haar features.\n"
            "\n"
        "There are two forms of operating:\n"
        "a) Using only one image on the IR light range and detecting"
        " the faces and measuring their mean temperature.\n"
        "b) Using two images, one on the visible light range to detect"
        " faces and another on the IR light range for measuring\n  "
        " the mean temperature by remapping the detected objects."
        " By using this mode, visible camera is used as a preview.\n"
            "\n"
        "Usage:\n"
        <<  argv[0]
        <<  "\n"
        "   @IR_input          <IR_file_or_cam_ID>    "
        "   -- IR range: Camera index or Image path.\n"

        "   @Vis_input         <Vis_file_or_cam_ID>   "
        "   -- Visible range: Camera index or Image path.\n"

        "   --cascade          = <cascade_path>       "
        "   -- Primary trained classifier (frontal face).\n"

        "   --nested-cascade   = <nested_cascade_path>"
        "   -- Optional secondary classifier (eyes)"
        " [only for visible range].\n"

        "   --pathToSave       = <save_path>          "
        "   -- Folder where the exported images are saved."
        " Default: data/Images\n"

        "   --IR_scale         = <IR_img_scale>       "
        "   -- Final IR Image scale factor. Default: 1.\n"

        "   --Vis_scale        = <Vis_img_scale>      "
        "   -- Final Vis. Image scale factor. Default: 1.\n"
            
        "   --try-flip                                "
        "   -- Use this when you want to search for inverted faces.\n"

        "   --remapping                               "
        "   -- Use this if you want to use the remapping method.\n"
        "\n"
        "When remapping, images must have the same origin"
        " (capturing from cameras or external images).\n"
        "\n"
        "Example:\n"
        <<  argv[0]
        <<  " --cascade=/usr/local/share/opencv4/haarcascades/"
        "haarcascade_frontalface_alt.xml --nested-cascade=/usr/local/"
        "share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
        " --pathToSave=/home/pi/Documents/exportedImagesFromOpenCV/ "
        "--IR_scale=4 --try-flip --remapping\n"
        "\n"
        "During execution:\n"
        "\t- Press 'S' to save all images.\n"
        "\t- Press any other key to exit.\n"
        "\n"
        "Using OpenCV version " << CV_VERSION << "\n" << endl;
}

// Functions declaration:
void DisplayHelp(const char** argv);
const string currentDateTime();
void remapper(vector<Rect> faces, vector<Rect>& newFaces, Mat img, double scF, double xF, double yF);
void nestedRemapper(vector<vector<Rect>> nestedObjects, vector<vector<Rect>>& newNestedObjects, Mat img, double scF);
void detectObjects(Mat& gray, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, bool tryflip,
                    vector<Rect>& faces, vector<vector<Rect>>& nestedObjects);
void drawingTool(Mat& img, vector<Rect> faces, vector<vector<Rect>> nestedObjects, Scalar color, double temperature, double scale);
void drawText(Mat& img, vector<Rect> faces, Scalar color, double temperature, double scale);
double IRtoTempComversion(const double A1, const double A2, const double B1, const double B2, const double IR);
double GetTempFromIRImage(const Mat& Image, const Rect& ROI, double& otsuThreshold, Mat& ImgROI, Mat& ImgROISegmented, Mat& Mask);
void Thermometer(Mat img, Mat& gray, vector<Rect> faces, double& meanTemp, double scale);
void getColor(const double& highThreshold, const double& lowerThreshold, const double& temperature, Scalar& color);
void saveData(Mat img1, Mat img2, Mat IR_image, Mat Vis_image, string savedImagesPath, bool remapping_mode);

int main(int argc, const char** argv)
{   
    // Input arguments declaration: 
    string cascadeName, nestedCascadeName;
    string IR_inputName, Vis_inputName, savedImagesPath;
    double IR_scale, Vis_scale;
    bool tryflip, remapping_mode;

    // Video capture, frames and images:
    VideoCapture IR_capture, Vis_capture;
    Mat IR_image, Vis_image;

    // Cascades Classifiers (Haar features):
    CascadeClassifier cascade, nestedCascade;

    char c; // 'Save' and 'Exit' key

    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{@IR_input||}"
        "{@Vis_input||}"
        "{cascade||}"
        "{nested-cascade||}"
        "{pathToSave|data/Images/|}"
        "{IR_scale|1|}"
        "{Vis_scale|1|}"
        "{try-flip||}"
        "{remapping||}"
    );
    
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    
    // Call for input arguments:
    cascadeName = parser.get<string>("cascade");
    nestedCascadeName = parser.get<string>("nested-cascade");
    IR_scale = parser.get<double>("IR_scale");
    Vis_scale = parser.get<double>("Vis_scale");
    tryflip = parser.has("try-flip");
    remapping_mode = parser.has("remapping");
    IR_inputName = parser.get<string>("@IR_input");
    Vis_inputName = parser.get<string>("@Vis_input");
    savedImagesPath = parser.get<string>("pathToSave");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    // Checking for ERRORS and WARNINGS:
    if(nestedCascadeName.empty() || !nestedCascade.load(samples::findFileOrKeep(nestedCascadeName,1)))
        cout << "[WARN]: Could not load classifier cascade for nested objects." << endl;
    if(cascadeName.empty() || !cascade.load(samples::findFileOrKeep(cascadeName,1)))
    {
        cerr << "ERROR (" << CASCADE_ERROR << "): Could not load classifier cascade." << endl;
        DisplayHelp(argv);

        return CASCADE_ERROR;
    }

    // We must check if IR and Visible images have the same origin
    // (capturing from cameras or external images, but not alternating both when remapping):
    bool IR_type, Vis_type;

    if(IR_inputName.empty() || (isdigit(IR_inputName[0]) && IR_inputName.size() == 1))
    {
        IR_type = 1; // IR camera

        // If @IR_filename is empty, search for IR camera:
        int IR_index = IR_inputName.empty() ? CD.DetectCameraID(CameraType::IR) : IR_inputName[0] - '0';
        
        if(!IR_capture.open(IR_index))
        {
            cerr << "ERROR (" << NOIRCAPTURE_ERROR << "): Capture from IR camera (#" <<  IR_index << ") didn't work." << endl;
            DisplayHelp(argv);
            
            return NOIRCAPTURE_ERROR;
        }
    }
    else
    {   
        IR_type = 0; // IR image
        IR_image = imread(samples::findFileOrKeep(IR_inputName,1), IMREAD_COLOR);
        if(IR_image.empty())
        {
            if(!IR_capture.open(samples::findFileOrKeep(IR_inputName,1)))
            {
                cerr << "ERROR (" << NOIRIMAGE_ERROR << "): Could not read IR image: " << IR_inputName << endl;
                DisplayHelp(argv);

                return NOIRIMAGE_ERROR;
            }
        }
    }
    if(remapping_mode)
    {
        if(Vis_inputName.empty() || (isdigit(Vis_inputName[0]) && Vis_inputName.size() == 1))
        {
            Vis_type = 1; // Visible camera
        
            // If @Vis_filename is empty, search for Visible camera:
            int Vis_index = Vis_inputName.empty() ? CD.DetectCameraID(CameraType::Visible) : Vis_inputName[0] - '0';
        
            if(!Vis_capture.open(Vis_index  ))
            {
                cerr << "ERROR (" << NOVISCAPTURE_ERROR << "): Capture from Visible camera (#" <<  Vis_index << ") didn't work." << endl;
                DisplayHelp(argv);

                return NOVISCAPTURE_ERROR;
            }
        }
        else
        {
            Vis_type = 0; // Visible image
            Vis_image = imread(samples::findFileOrKeep(Vis_inputName,1), IMREAD_COLOR);
            if(Vis_image.empty())
            {
                if(!Vis_capture.open(samples::findFileOrKeep(Vis_inputName,1)))
                {
                    cerr << "ERROR (" << NOVISIMAGE_ERROR << "): Could not read Visible image: " << Vis_inputName << endl;
                    DisplayHelp(argv);

                    return NOVISIMAGE_ERROR;
                }
            }
        }

        if(IR_type != Vis_type)
        {
            cerr << "ERROR (" << ORIGIN_ERROR << "): Images must have the same origin (capturing from cameras or external images)." << endl;
            cout << "\t    Alternating both have no sense because images will not be correlated!" << endl;
            DisplayHelp(argv);

            return ORIGIN_ERROR;
        }
    }

    // PREVIEW mode:
    if(remapping_mode && Vis_type)
    {
        cout << "\nStarting preview, press 'SAVE DATA' to start.\nPress 'ESC' or 'Q' to exit preview mode." << endl;

        // Loading picture from camera:
        Mat image;
        Vis_capture.read(image);

        // Create a frame where UI components will be rendered to:
        Mat frame;

        // CONTROL VALUES ADJUSTED MANUALLY...
        // Initialize arguments for the UI window:
        int top = 0.025*image.rows;
        int left = 0.025*image.cols;
        int right = left;
        int bottom = 0.15*image.rows; // Botton margin
        int borderType = BORDER_CONSTANT;
        Scalar value(237, 237, 237); // Window color
        // Button parameters:
        int s_x = 0.25*image.cols;
        int s_y = 0.075*image.rows;
        int x = 0.5*(image.cols + left + right) - s_x/2;
        int y = 1.1*image.rows - s_y/2;
        double button_font_size = 0.5;
        unsigned int inside_color = 0xFF87919B;
        string button_name = "SAVE DATA";

        // Init CVUI and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME):
        string win_name = "FaceTemp (Preview)";
        cvui::init(win_name);

        for(;;)
        {
            Vis_capture.read(image);

            // Create frame border:
            copyMakeBorder(image, frame, top, bottom, left, right, borderType, value);

            if (cvui::button(frame, x, y, s_x, s_y, button_name, button_font_size, inside_color) || c == 's' || c == 'S')
            {
                break;
            }

            cvui::imshow(win_name, frame);
            
            c = waitKey(20);
            if(c == 27 || c == 'q' || c == 'Q')
            {
                exit(0);
            }
        }

        destroyAllWindows();

        // Release memory:
        image.release();
        frame.release();
    }

    // Needed variables:
    vector<Rect> faces;
    vector<vector<Rect>> nestedObjects;
    Mat img1, img2, gray;
    double meanTemp;
    Scalar color;

    // Program initialization:
    if(IR_capture.isOpened())
    {   
        cout << "\nTaking facial picture, stay still and look at camera(s)..." << endl;

        // Take picture(s):
        IR_capture.read(IR_image);
        if(remapping_mode)
        {
            Vis_capture.read(Vis_image);

            if(Vis_image.empty())
            {
                cout << "ERROR (" << EMPTYVIS_ERROR << "): Empty image in the visible range when capturing from camera." << endl;
                return EMPTYVIS_ERROR;
            }
        }
        if(IR_image.empty())
        {
            cout << "ERROR (" << EMPTYIR_ERROR << "): Empty image in the IR range when capturing from camera." << endl;
            return EMPTYIR_ERROR;
        }
    }
    else
    {
        if(remapping_mode)
            cout << "\nDetecting face(s) from " << Vis_inputName << endl;
        else
            cout << "\nDetecting face(s) from " << IR_inputName << endl;
    }

    img1 = IR_image.clone();
    
    if(remapping_mode)
    {
        img2 = Vis_image.clone();

        // Convert Vis. space color to Gray Scale:
        cvtColor(img2, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        // Detecting faces in the Visible light range:
        detectObjects(gray, cascade, nestedCascade, tryflip, faces, nestedObjects);

        vector<Rect> newFaces;
        vector<vector<Rect>> newNestedObjects;
        
        // Remapping faces and eyes:
        remapper(faces, newFaces, img1, scaleFactor, xFactor, yFactor);
        nestedRemapper(nestedObjects, newNestedObjects, img1, scaleFactor);
        
        // Convert IR space color to Gray Scale:
        cvtColor(img1, gray, COLOR_BGR2GRAY);

        // Measure mean temperature from faces:
        Thermometer(img1, gray, newFaces, meanTemp, IR_scale);

        // Color for Drawing Tool:
        getColor(HIGHER_TEMP_THRES, LOWER_TEMP_THRES, meanTemp, color);

        // Resize and draw objects for Vis. image:
        resize(img2, img2, Size(), Vis_scale, Vis_scale, INTER_LINEAR);
        drawingTool(img2, faces, nestedObjects, color, meanTemp, Vis_scale);
        drawText(img2, faces, color, meanTemp, Vis_scale);

        // Resize and draw objects for IR image:      
        resize(img1, img1, Size(), IR_scale, IR_scale, INTER_LINEAR);
        drawingTool(img1, newFaces, newNestedObjects, color, meanTemp, IR_scale);
        drawText(img1, newFaces, color, meanTemp, IR_scale);

        // Release memory:
        newFaces.clear();
        newNestedObjects.clear();
    }
    else
    {
        // Convert IR space color to Gray Scale:
        cvtColor(img1, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        // Detect faces in the IR light range:
        detectObjects(gray, cascade, nestedCascade, tryflip, faces, nestedObjects);

        // Measure mean temperature from faces:
        Thermometer(img1, gray, faces, meanTemp, IR_scale);

        // Color for Drawing Tool:
        getColor(HIGHER_TEMP_THRES, LOWER_TEMP_THRES, meanTemp, color);

        // Resize and draw objects for IR image:
        resize(img1, img1, Size(), IR_scale, IR_scale, INTER_LINEAR);
        drawingTool(img1, faces, nestedObjects, color, meanTemp, IR_scale);
        drawText(img1, faces, color, meanTemp, IR_scale);
    }

    // Release memory:
    gray.release();
    faces.clear();
    nestedObjects.clear();
        
    cout << "\t- Press 'S' to save all images.\n\t- Press any other key to exit.\n" << endl;

    imshow("IR Image", img1);
    if(remapping_mode)
        imshow("Visible Image", img2);

    c = waitKey(0);
    
    if(c == 's' || c == 'S')
        saveData(img1, img2, IR_image, Vis_image, savedImagesPath, remapping_mode);

    // Close all windows:
    destroyAllWindows();

    // Release images from memory:
    IR_image.release();
    Vis_image.release();
    img1.release();
    img2.release();

    // Release Video capture: 
    IR_capture.release();
    Vis_capture.release();

    return 0;
}

// Functions definition:
void DisplayHelp(const char** argv)
{   
    cout << "\nPlease, make sure you have introduced the correct arguments. This is how this program works..." << endl;
    help(argv);
}

const string currentDateTime()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%d-%m-%Y.%H-%M-%S", &tstruct);

    return buf;
}

void remapper(vector<Rect> faces, vector<Rect>& newFaces, Mat img, double scF, double xF, double yF)
{   
    Rect face, newFace;

    for(size_t i = 0; i < faces.size(); i++)
    {   
        face = faces[i];

        cout << "\nFace #" << i+1 << " detected at: x = " << face.x << "; y = " << face.y << "; height = " << face.height << "; width = " << face.width << endl;

        // Translation movement:
        newFace.x = cvRound((face.x - xF + 1) / scF);
        newFace.y = cvRound((face.y - yF + 1) / scF);

        // Remapping scale factor:
        newFace.width = cvRound(face.width / scF);
        newFace.height = cvRound(face.height / scF);

        // Check if part of the face is outside the image and correct it:
        if(newFace.x < 0)
            newFace.x = 0;
        if(newFace.y < 0)
            newFace.y = 0;
        if((newFace.x + newFace.width-1) > img.cols)
            newFace.width = img.cols - newFace.x;
        if((newFace.y + newFace.height-1) > img.rows)
            newFace.height = img.rows - newFace.y;

        cout << "Remapping to: x' = " << newFace.x << "; y' = " << newFace.y << "; height' = " << newFace.height << "; width' = " << newFace.width << endl;

        newFaces.push_back(newFace);
    }    
}

void nestedRemapper(vector<vector<Rect>> nestedObjects, vector<vector<Rect>>& newNestedObjects, Mat img, double scF)
{
    vector<Rect> eyes, newEyes;
    Rect eye, newEye;
    
    if(!nestedObjects.empty())
    {
        for(size_t i = 0; i < nestedObjects.size(); i++)
        {
            eyes = nestedObjects[i];

            if(eyes.empty())
            {
                newNestedObjects.push_back(eyes);
                continue;
            }

            for(size_t j = 0; j < eyes.size(); j++)
            {
                eye = eyes[j];

                // Remapping scale factor:
                newEye.x = cvRound(eye.x / scF);
                newEye.y = cvRound(eye.y / scF);
                newEye.width = cvRound(eye.width / scF);
                newEye.height = cvRound(eye.height / scF);

                // Check if part of the eye is outside of the image and correct it:
                if(newEye.x < 0)
                    newEye.x = 0;
                if(newEye.y < 0)
                    newEye.y = 0;
                if((newEye.x + newEye.width-1) > img.cols)
                    newEye.width = img.cols - newEye.x;
                if((newEye.y + newEye.height-1) > img.rows)
                    newEye.height = img.rows - newEye.y;

                newEyes.push_back(newEye);
            }

            newNestedObjects.push_back(newEyes);
        }
    }

    // Release memory for objects that will not be used anymore:
    eyes.clear();
    newEyes.clear();
}

void detectObjects(Mat& gray, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, bool tryflip,
                    vector<Rect>& faces, vector<vector<Rect>>& nestedObjects)
{   
    // Init time (in ticks):
    double t = (double)getTickCount();

    // Detect faces of different sizes using cascade classifier:
    cascade.detectMultiScale(gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

    if(tryflip)
    {   
        vector<Rect> faces2;

        flip(gray, gray, 0); // Vertical flip (across x-axis)
        cascade.detectMultiScale(gray, faces2, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

        for(vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); r++)
        {
            faces.push_back(Rect(r->x, gray.rows - r->y - r->height, r->width, r->height));
        }

        // Release memory:
        faces2.clear();

        // Turn back flip (across x-axis)
        flip(gray, gray, 0);
    }

    Mat grayROI;
    vector<Rect> nfaces;
    
    // Delete false detections:
    for(size_t i = 0; i < faces.size(); i++)
    {
        Rect r = faces[i];
        double correctionFactor = 0.25 * gray.rows;

        if(r.width < correctionFactor || r.height < correctionFactor)
            continue;

        nfaces.push_back(r);
    }

    faces = nfaces;
    
    // Release memory:
    nfaces.clear();

    if(!nestedCascade.empty())
    {   

        // Detect eyes:
        for(size_t i = 0; i < faces.size(); i++)
        {
            Rect r = faces[i];
            vector<Rect> eyes;
            grayROI = gray(r);
        
            // Detection of eyes for every face:
            nestedCascade.detectMultiScale(grayROI, eyes, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));

            nestedObjects.push_back(eyes);
        }
    }
    // Release memory for objects that will not be used anymore:
    grayROI.release();

    // Detection time (in seconds):
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Elapsed time in detection = " << t << " s" << endl;
}

void drawingTool(Mat& img, vector<Rect> faces, vector<vector<Rect>> nestedObjects, Scalar color, double temperature, double scale)
{
    double linewidth = 0.009 * img.rows;
    double correctionFactor = 10 * linewidth;

    for(size_t i = 0; i < faces.size(); i++)
    {   
        Rect r = faces[i];     
        
        // Draw rectangles around faces:
        rectangle(img, Point(cvRound(r.x*scale), cvRound(r.y*scale)), Point(cvRound((r.x + r.width-1)*scale), 
                    cvRound((r.y + r.height-1)*scale)), color, linewidth, 8, 0);

        vector<Rect> eyes = nestedObjects[i];
        if(eyes.empty() || nestedObjects.empty())
            continue;

        Point eyeCenter, x1, x2, y1, y2;
        int radius;
        
        // Draw "+" on eyes for every face:
        for(size_t j = 0; j < eyes.size(); j++)
        {
            Rect nr = eyes[j];

            eyeCenter.x = cvRound((r.x + nr.x-1 + (nr.width-1)*0.5)*scale);
            eyeCenter.y = cvRound((r.y + nr.y-1 + (nr.height-1)*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.0625*scale);

            // CORRECTION: when the light is not optimal, sometimes an eye is detected two times.
            if(abs(eyeCenter.x - eyes[j-1].x) < correctionFactor && abs(eyeCenter.y - eyes[j-1].y) < correctionFactor)
                continue;
            
            // Point X1:
            x1.x = eyeCenter.x - radius;
            x1.y = eyeCenter.y;
            // Point X2:
            x2.x = eyeCenter.x + radius;
            x2.y = eyeCenter.y;
            // Point Y1:
            y1.x = eyeCenter.x;
            y1.y = eyeCenter.y - radius;
            // Point Y2:
            y2.x = eyeCenter.x;
            y2.y = eyeCenter.y + radius;

            // Draw lines:
            line(img, x1, x2, color, linewidth, 8, 0);
            line(img, y1, y2, color, linewidth, 8, 0);
        }
    }
}

void drawText(Mat& img, vector<Rect> faces, Scalar color, double temperature, double scale)
{
    double linewidth = 0.009 * img.rows;

    for(size_t i = 0; i < faces.size(); i++)
    {   
        if(faces.size() > 1)
            break;
        
        Rect r = faces[i];
        r.x = r.x * scale;
        r.y = r.y * scale;
        r.width = r.width * scale;
        r.height = r.height * scale;

        // Draw temperature:
        stringstream ssTemp;
        ssTemp << std::fixed << std::setprecision(2) << temperature << "  C";
        Point textPoint;
        textPoint.x = cvRound(r.x + (r.width - 1)*0.5 - r.width*0.23);
        textPoint.y = cvRound(r.y + (r.height - 1) + r.height/7);
        putText(img, ssTemp.str(), textPoint, FONT_HERSHEY_DUPLEX, linewidth/6, color, 0.5*linewidth, 8, 0);

        // Draw degree symbol:
        Point degree;
        degree.x = cvRound(textPoint.x + img.rows*41/240);
        degree.y = cvRound(textPoint.y - img.cols/50);
        circle(img, degree, linewidth, color, 0.5*linewidth, 8, 0);
        
    }
}

double IRtoTempComversion(const double A1, const double A2, const double B1, const double B2, const double IR)
{
    
    double a, b, result;

    a = (A2-A1)/(B2-B1);
    b = A2 - a*B2;

    result = IR*IRSensorTo8Bit*a + b;

    return result;
}

double GetTempFromIRImage(const Mat& Image, const Rect& ROI, double& otsuThreshold, Mat& ImgROI, Mat& ImgROISegmented, Mat& Mask)
{
    double IR, temperature, threshold;
    ImgROI = Image(ROI);

    // Obtain the Otsu's threshold:
    otsuThreshold = cv::threshold(ImgROI, ImgROISegmented, 0, 255, THRESH_TOZERO|THRESH_OTSU);
    cout << "\nOTSU's Threshold = " << otsuThreshold << endl;
    // Binary segmentation using Otsu's threshold:
    cv::threshold(ImgROI, Mask, otsuThreshold, 255, THRESH_BINARY);
        
    // Compute ROI mean:
    IR = cv::mean(ImgROI, Mask)[0];
    cout << "IR = " << IR << "\n" << endl;

    // IR to temperature conversion:
    temperature = IRtoTempComversion(T1, T2, R1, R2, IR);

    return temperature;
}

void Thermometer(Mat img, Mat& gray, vector<Rect> faces, double& meanTemp, double scale)
{
    Mat ImgROI, ImgROISegmented, Mask;
    double thresholdSeg;
    
    for(size_t i = 0; i < faces.size(); i++)
    {   
        Rect r = faces[i];
            
        // Measuring Temperature from face [i]:
        meanTemp = GetTempFromIRImage(gray, r, thresholdSeg, ImgROI, ImgROISegmented, Mask);
        cout << "The temperature for the face #" << i+1 << " is: T(°C) = " << meanTemp << endl;

        // Resize all processed IR images:
        resize(ImgROI, ImgROI, Size(), scale, scale, INTER_LINEAR);
        resize(ImgROISegmented, ImgROISegmented, Size(), scale, scale, INTER_LINEAR);
        resize(Mask, Mask, Size(), scale, scale, INTER_LINEAR);

        stringstream ssNum;
        ssNum << i+1;
            
        string ROI_name = "ROI (face #" + ssNum.str() + ")";
        string OTSU_name = "OTSU (face #" + ssNum.str() + ")";
        string Mask_name = "Mask (face #" + ssNum.str() + ")";
        
        imshow(ROI_name, ImgROI);
        imshow(OTSU_name, ImgROISegmented);
        imshow(Mask_name, Mask);
    }
    cout << endl;

    // Release memory:
    ImgROI.release();
    ImgROISegmented.release();
    Mask.release();
}

void getColor(const double& highThreshold, const double& lowerThreshold, const double& temperature, Scalar& color)
{
    if(temperature >= highThreshold)
        // Red when fever:
        color = CV_RGB(255,0,0);    
    else if(temperature >= lowerThreshold)
        // Orange when slight fever:
        color = CV_RGB(255,153,51);
    else
        // Green when normal:
        color = CV_RGB(0,255,0);
    
}

void saveData(Mat img1, Mat img2, Mat IR_image, Mat Vis_image, string savedImagesPath, bool remapping_mode)
{
    string timeStamp = currentDateTime();
    string filename = savedImagesPath + "IR_raw_image_" + timeStamp + ".jpg";
    imwrite(filename, IR_image);

    if(remapping_mode)
    {   
        filename = savedImagesPath + "Vis_raw_image_" + timeStamp + ".jpg";
        imwrite(filename, Vis_image);

        filename = savedImagesPath + "Vis_FaceDetect_image_" + timeStamp + ".jpg";
        imwrite(filename, img2);
    }

    filename = savedImagesPath + "IR_FaceDetect_image_" + timeStamp + ".jpg";
    imwrite(filename, img1);
}
