//
//  SceneRec.m
//  Intro_iOS_OpenCV
//
//  Created by HYJ on 9/22/15.
//  Copyright © 2015 CMU_16432. All rights reserved.
//

#import "SceneRec.h"

@implementation SceneRec


//==============================================================================
arma::fmat Cv2Arma(cv::Mat &cvX)
{
    arma::fmat X(cvX.ptr<float>(0), cvX.cols, cvX.rows, false); // This is the transpose of the OpenCV X_
    return X; // Return the new matrix (no new memory allocated)
}
//==============================================================================
cv::Mat Arma2Cv(arma::fmat &X)
{
    cv::Mat cvX = cv::Mat(X.n_cols, X.n_rows,CV_32F, X.memptr()).clone();
    return cvX; // Return the new matrix (new memory allocated)
}


+ (arma::fmat)loadMat:( NSString* )name rowSize:(int)rows colSize:(int)cols{

    ///////Read dictionary
    
    NSString* path = [[NSBundle mainBundle] pathForResource:name
                                                     ofType:@"txt"];
    
    const char *SphereName = [path UTF8String]; // Convert to const char *
    arma::fmat dict; dict.load(SphereName); // Load the Sphere into memory should be 3xN
    return dict;
}

+ (std::vector<cv::Mat>*) loadFilterBank:( NSString*)name rowSize:(int)rows colSize:(int)cols filterBankSize:(int) slides {
    std::vector<cv::Mat>* filterBank = new std::vector<cv::Mat>();

    
    for (int ind = 0; ind<slides; ind++) {
        NSString* fname = [name stringByAppendingString:[NSString stringWithFormat:@"%d",ind+1]];
        
     //   NSLog(fname);
        
        NSString* path = [[NSBundle mainBundle] pathForResource:fname
                                                         ofType:@"txt"];
        
         const char *SphereName = [path UTF8String]; // Convert to const char *
         arma::fmat sphere; sphere.load(SphereName); // Load the Sphere into memory should be 3xN
         filterBank->push_back(Arma2Cv(sphere));
        
  //sample code !!!!!! modeified!!!

  //      NSString *str = [[NSBundle mainBundle] pathForResource:@"filter1" ofType:@"txt"];

        //        const char *SphereName = [str UTF8String]; // Convert to const char *
//        arma::fmat sphere; sphere.load(SphereName); // Load the Sphere into memory should be 3xN
//        std::cout<<arma::max(arma::max(sphere))<<std::endl;
//        
        
//        NSString* content = [NSString stringWithContentsOfFile:path
//                                                      encoding:NSUTF8StringEncoding
//                                                         error:NULL];
//        
//        NSArray *components = [content componentsSeparatedByString: @"  "];
//        
//        NSMutableArray* nums = [[NSMutableArray alloc] init];
//        int i = 0;
//        while (i < [components count]) {
//            if ([[components objectAtIndex:i] floatValue] != 0) {
//                [nums addObject:[components objectAtIndex:i]];
//            }
//            i++;
//        }
//        filterBank->push_back(cv::Mat(49,49,CV_64FC1));
//        for(unsigned i = 0; i < rows; i++){
//            for(unsigned j = 0; j < cols; j++){
//                (*filterBank)[ind].at<double>(i, j) = [[nums objectAtIndex:i*rows + j] floatValue];
//            }
//        }
    }
    return filterBank;
}

- (arma::fmat)getVisualWord:(cv::Mat) I filterBank:(std::vector<cv::Mat>*)filterBank dictionary:(arma::fmat)dict{
    arma::fmat filterResponses = extractFilterResponses(I, filterBank);
    std::cout<<"filterResponses "<<mean(mean(filterResponses))<<std::endl;

    arma::fmat D = pdist2(dict, filterResponses);

    arma::fmat wordMap = arma::zeros<arma::fmat>(D.n_cols,1);

    for (int i = 0; i < D.n_cols; i++) {
        arma::uword tmp;
        D.col(i).min(tmp);wordMap.at(i) = tmp;
    }
    wordMap.reshape(I.rows, I.cols);
  // std::cout<<"wordMap "<<wordMap<<std::endl;

    return wordMap;
}


arma::fmat  pdist2(arma::fmat dict, arma::fmat responses){
    arma::fmat D = arma::zeros<arma::fmat>(dict.n_cols,responses.n_rows);
    responses = arma::trans(responses);
    for (unsigned j = 0; j<D.n_cols; j++) {
        for (unsigned i = 0; i<D.n_rows; i++) {
            //std::cout<<a.size()<<"  "<<b.size()<<std::endl;
            arma::fmat tmp =  dict.col(i) - responses.col(j);
            D(i,j) = arma::sum(arma::dot(tmp,tmp));
            //std::cout<<"D(i,j) "<<D(i,j)<<"  responses.col(j)  "<<responses.col(j)<<std::endl;

        }
    }
    return D;
    
}

arma::fmat extractFilterResponses(cv::Mat I_org, std::vector<cv::Mat>* filterBank){
    cv::Mat I(I_org.size(), CV_32FC3);
    I_org.convertTo(I, CV_32FC3);
    I = I*1.0/255;
    cv::cvtColor(I, I, CV_RGB2Lab);
    std::vector<cv::Mat> channels(3);
    cv::split(I, channels);
    
   unsigned pixelCount = I.cols*I.rows;
    arma::fmat filterResponse = arma::zeros<arma::fmat>(pixelCount,filterBank->size()*3); //arzeros(pixelCount,filterBank.size()*3,CV_64F);
    for (unsigned i = 0; i<filterBank->size(); i++) {
        cv::Mat tmp;
        cv::filter2D(channels[0],tmp,-1,(*filterBank)[i]);
        arma::fmat test =Cv2Arma(channels[0]);
       // std::cout<<"max "<<test(0,0)<<std::endl;

       // std::cout<<"max "<<arma::max(Cv2Arma((*filterBank)[i]))<<std::endl;

       // std::cout<<"max "<<arma::max(Cv2Arma(tmp))<<std::endl;

        filterResponse.col(i) = arma::reshape(Cv2Arma(tmp), pixelCount, 1);
        cv::filter2D(channels[1],tmp,-1,(*filterBank)[i]);
        filterResponse.col(i + filterBank->size()) = arma::reshape(Cv2Arma(tmp), pixelCount, 1);
        cv::filter2D(channels[2],tmp,I.depth(),(*filterBank)[i]);
        filterResponse.col(i + 2*filterBank->size()) = arma::reshape(Cv2Arma(tmp), pixelCount, 1);
    }
    return filterResponse;
}

- (int) NNpredict:(arma::vec)h Theta1:(arma::fmat)Theta1  Theta2:(arma::fmat)Theta2{
   // std::cout<<"h "<<h<<std::endl;
    //std::cout<<"Theta1 "<<Theta1<<std::endl; Good
    //std::cout<<"Theta2 "<<Theta2<<std::endl; Good

    h.insert_rows(0, 1);
    h.at(0) = 1;
    arma::mat a(h);
    arma::mat h1 = sigmoid(a.t()*Theta1.t());
    h1.insert_cols(0, 1);
    h1.at(0) = 1;
   // std::cout<<"h1 "<<h1<<std::endl;
    arma::mat h2 = sigmoid(h1*Theta2.t());
    arma::uword ans; h2.max(ans);
  //  std::cout<<"h2 "<<h2<<std::endl;
    return ans;
}

arma::mat sigmoid(arma::mat z){
    arma::mat one = arma::ones<arma::mat>(arma::size(z));
    return one/(one + exp(-z));
}

- (arma::vec)getImageFeatureSPM:(unsigned)layerNum wordMap:(arma::fmat)wordMap sizeOfDict:(int)size{
    std::vector<arma::Col<arma::uword>> h_subs;
    divide(wordMap, h_subs, 1, layerNum, size);
    //rebuild h
    arma::Col<arma::uword> h = h_subs[0];
    for (int i = 1; i<5; i++) {
        h = arma::join_cols(h, h_subs[i]);
    }
    //norm
  //  std::cout<<"final h size"<<h.size()<<std::endl;
    long long S = arma::sum(h);
   // std::cout<<"h: "<<h<<std::endl;

    arma::vec ret = arma::zeros(h.size());
    for(int i = 0; i < h.size();i++){
        ret[i] = h(i)*1.0/S;
    }
  //  std::cout<<"S: "<<S<<std::endl;

    //modified NNpredict
    return ret;
}

arma::Col<arma::uword> divide(arma::fmat wordMap, std::vector<arma::Col<arma::uword>>& h, int now_cnt, int layerNum, int dictSize){
    arma::Mat<arma::uword> sub_h(200,4);
    unsigned M = wordMap.n_rows-1;
    unsigned N = wordMap.n_cols-1;
    unsigned m = ceil(M*1.0/2);
    unsigned n = ceil(N*1.0/2);
    arma::Col<arma::uword> part;
    int weight = 1/4;
    if (now_cnt != layerNum) {
        for (int i = 0; i<2; i++) {
            for (int j = 0; j<2; j++) {
                arma::fmat tmp = wordMap(arma::span(i*m,std::min((i+1)*m, M)),arma::span(j*n,std::min((j+1)*n,N)));
                sub_h.col(i*2+j) = divide(tmp, h, now_cnt+1,layerNum,dictSize);
            }
        }
        part = sub_h.col(1) + sub_h.col(2) + sub_h.col(3) + sub_h.col(0);
        h.push_back(part);
    }else{
        wordMap.reshape(wordMap.n_cols*wordMap.n_rows, 1);
        arma::Col<float> a = arma::vectorise(wordMap);
        part = arma::hist (a,arma::linspace<arma::Col<float>>(0,dictSize,dictSize));
       // std::cout<<"part: "<<part<<std::endl;
        h.push_back(part);
    }
    return part;
}


@end

