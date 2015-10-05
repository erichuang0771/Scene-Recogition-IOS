//
//  readMat.h
//  Intro_iOS_OpenCV
//
//  Created by HYJ on 9/22/15.
//  Copyright © 2015 CMU_16432. All rights reserved.
//

#import <Foundation/Foundation.h>

#ifdef __cplusplus
#include "armadillo" // Includes the armadillo library
#include <opencv2/opencv.hpp>
#include <stdlib.h> // Include the standard library
#include <iostream>
#endif

@interface SceneRec:NSObject
/* method declaration */
@property arma::fmat dictionary;
@property arma::fmat Theta1;
@property arma::fmat Theta2;
@property std::vector<cv::Mat>* filterBank;
@property arma::fmat wordMap;
@property arma::vec h;

- (int)NNpredict:(arma::vec)h Theta1:(arma::fmat)Theta1  Theta2:(arma::fmat)Theta2;

- (arma::vec) getImageFeatureSPM:(unsigned)layerNum wordMap:(arma::fmat)wordMap sizeOfDict:(int)size;

- (arma::fmat)getVisualWord:(cv::Mat) I filterBank:(std::vector<cv::Mat>*)filterBank dictionary:(arma::fmat)dict;

+ (arma::fmat)loadMat:(NSString*)name rowSize:(int)rows colSize:(int)cols;

+ (std::vector<cv::Mat>*) loadFilterBank:(NSString*)name rowSize:(int)rows colSize:(int)cols filterBankSize:(int) slides;


//Need to be done: getImageFeatureSPM & NNpredict
@end

