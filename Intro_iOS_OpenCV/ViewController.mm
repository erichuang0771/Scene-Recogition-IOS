//
//  ViewController.m
//  Intro_iOS_OpenCV
//
//  Created by Simon Lucey on 9/7/15.
//  Copyright (c) 2015 CMU_16432. All rights reserved.
//

#import "ViewController.h"

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
//#import <opencv2/highgui/ios.h>
#import "armadillo" // Includes the armadillo library
#import "SceneRec.h"
#endif

// Include iostream and std namespace so we can mix C++ code in here
#include <iostream>

using namespace std;
using namespace cv;
@interface ViewController () {
    // Setup the view
    UIImageView *imageView_;
    SceneRec *RecSystem;
    int image_name;
    UIButton *aButton ;
    int test;
    //UIButton *gaussButton;// Button to initiate OpenCV processing of image

}
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    //Do any additional setup after loading the view, typically from a nib.
    
    imageView_ = [[UIImageView alloc] initWithFrame:CGRectMake(0.0, 0.0, self.view.frame.size.width, self.view.frame.size.height)];
    
    // 2. Important: add OpenCV_View as a subview
    [self.view addSubview:imageView_];
    imageView_.contentMode = UIViewContentModeScaleAspectFit;
    
    //[gaussButton setHidden:false];
    // Important part that connects the action to the member function buttonWasPressed
    aButton = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    aButton.frame = CGRectMake(10,20,200,30);
    [aButton setTitle:@"Pick Up One!" forState:UIControlStateNormal];
    [aButton addTarget:self action:@selector(nextG) forControlEvents:UIControlEventTouchUpInside];
//    [gaussButton setTitle:@"Next Gauss" forState:UIControlStateNormal];
//    gaussButton.frame = CGRectMake(80.0, 210.0, 160.0, 40.0);
//    [imageView_ addSubview:gaussButton];
//    [gaussButton addTarget:self action:@selector(nextG) forControlEvents:UIControlEventTouchUpInside];

    // 3.Read in the image (of the famous Lena)
    RecSystem = [[SceneRec alloc] init];
    
    RecSystem.dictionary = [SceneRec loadMat:@"dict" rowSize: 114 colSize: 200];
    RecSystem.Theta1 = [SceneRec loadMat:@"Theta1" rowSize: 100 colSize: 1001];
    RecSystem.Theta2 = [SceneRec loadMat:@"Theta2" rowSize: 8 colSize: 101];
    RecSystem.filterBank  = [SceneRec loadFilterBank:@"filter" rowSize:49 colSize:49 filterBankSize:38];
    
    //UIImage* test = [self UIImageFromCVMat:(*RecSystem.filterBank)[37]*255];
    // imageView_.image = test;
    image_name = 1;
//    while (image_name < 5) {
        NSString* cname = @"start.png";
//
        UIImage *image = [UIImage imageNamed:cname];
//        cv::Mat I = [self cvMatFromUIImage:image];
//        
       if(image != nil) imageView_.image = image; // Display the image if it is there....
//        else cout << "Cannot read in the file" << endl;
//        
//        RecSystem.wordMap = [RecSystem getVisualWord:I filterBank:RecSystem.filterBank dictionary:RecSystem.dictionary];
//        RecSystem.h = [RecSystem getImageFeatureSPM:2 wordMap:RecSystem.wordMap sizeOfDict:200];
//        int result = [RecSystem NNpredict:RecSystem.h Theta1:RecSystem.Theta1 Theta2:RecSystem.Theta2];
//        std::vector<std::string> ans = {"airport","auditorium","bedroom","campus","desert","football_stadium","landscape","rainforest"};
//        std::cout<<"done!"<<" result:"<<ans[result]<<std::endl;
//        
//        cv::Size sz = getTextSize(ans[result], FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0 );
//        cv::putText(I, ans[result], Point2d(I.cols/2 - sz.width/2,I.rows-10), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(255,20,255));
//        
//        imageView_.image = [self UIImageFromCVMat:I];
        image_name ++;

    [self.view addSubview:aButton];

    
}
- (void)nextG {
    if(image_name %2 == 1){
    NSString* cname = [@"c" stringByAppendingString:[NSString stringWithFormat:@"%d.jpg",test]];
    
    UIImage *image = [UIImage imageNamed:cname];
    cv::Mat I = [self cvMatFromUIImage:image];
    
    if(image != nil) imageView_.image = image; // Display the image if it is there....
    else cout << "Cannot read in the file" << endl;
    
    RecSystem.wordMap = [RecSystem getVisualWord:I filterBank:RecSystem.filterBank dictionary:RecSystem.dictionary];
    RecSystem.h = [RecSystem getImageFeatureSPM:2 wordMap:RecSystem.wordMap sizeOfDict:200];
    int result = [RecSystem NNpredict:RecSystem.h Theta1:RecSystem.Theta1 Theta2:RecSystem.Theta2];
    std::vector<std::string> ans = {"airport","auditorium","bedroom","campus","desert","football_stadium","landscape","rainforest"};
    std::cout<<"done!"<<" result:"<<ans[result]<<std::endl;
    
    cv::Size sz = getTextSize(ans[result], FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0 );
    cv::putText(I, ans[result], Point2d(I.cols/2 - sz.width/2,I.rows-10), FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar(255,20,255));
    
    imageView_.image = [self UIImageFromCVMat:I];
    [aButton setTitle:@"Pick Next One!" forState:UIControlStateNormal];

    }else{
        test = std::rand() % 6 + 1;
        NSString* cname = [@"c" stringByAppendingString:[NSString stringWithFormat:@"%d.jpg",test]];
        
        UIImage *image = [UIImage imageNamed:cname];
        imageView_.image = image;
        [aButton setTitle:@"Guess Now!" forState:UIControlStateNormal];
    }
    image_name++;

    
}
// Member functions for converting from cvMat to UIImage
- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
    CGFloat cols = image.size.width;
    CGFloat rows = image.size.height;
    
    cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}
// Member functions for converting from UIImage to cvMat
-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat
{
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

@end