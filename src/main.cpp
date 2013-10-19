#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>
#include <math.h>
#include <string>

using namespace std;

string demanderImage();

void calculGradient(cv::Mat& img);
cv::Mat norme(cv::Mat img);
cv::Mat applyFilter(cv::Mat& img, cv::Mat& filtre);

std::vector<cv::Mat> prewittFilter, sobelFilter, kirschFilter, usedFilter;
std::vector<cv::Mat> usedFilteredImg;
cv::Mat module;
cv::Mat pente;

int main()
{
    string cheminRepImg, cheminImage;
    cv::Mat imgO;
    bool bonChemin;

    double m2_1[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    double m2_2[3][3] = {{1, 1, 1}, {0, 0, 0}, {-1, -1,-1}};
    double m2_3[3][3] = {{1, 1, 0}, {1, 0, -1}, {0, -1, -1}};
    double m2_4[3][3] = {{0, 1, 1}, {-1, 0, 1}, {-1, -1, 0}};

    prewittFilter.push_back(cv::Mat(3, 3, CV_64FC1, m2_1));
    prewittFilter.push_back(cv::Mat(3, 3, CV_64FC1, m2_2));
    prewittFilter.push_back(cv::Mat(3, 3, CV_64FC1, m2_3));
    prewittFilter.push_back(cv::Mat(3, 3, CV_64FC1, m2_4));

    double m3_1[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    double m3_2[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    double m3_3[3][3] = {{2, 1, 0}, {1, 0, -1}, {0, -1, -2}};
    double m3_4[3][3] = {{0, 1, 2}, {-1, 0, 1}, {-2, -1, 0}};

    sobelFilter.push_back(cv::Mat(3, 3, CV_64FC1, m3_1));
    sobelFilter.push_back(cv::Mat(3, 3, CV_64FC1, m3_2));
    sobelFilter.push_back(cv::Mat(3, 3, CV_64FC1, m3_3));
    sobelFilter.push_back(cv::Mat(3, 3, CV_64FC1, m3_4));

    double m4_1[3][3] = {{-3, -3, 5}, {-3, 0, 5}, {-3, -3, 5}};
    double m4_2[3][3] = {{5, 5, 5}, {-3, 0, -3}, {-3, -3, -3}};
    double m4_3[3][3] = {{5, 5, -3}, {5, 0, -3}, {-3, -3, -3}};
    double m4_4[3][3] = {{-3, 5, 5}, {-3, 0, 5}, {-3, -3, -3}};

    kirschFilter.push_back(cv::Mat(3, 3, CV_64FC1, m4_1));
    kirschFilter.push_back(cv::Mat(3, 3, CV_64FC1, m4_2));
    kirschFilter.push_back(cv::Mat(3, 3, CV_64FC1, m4_3));
    kirschFilter.push_back(cv::Mat(3, 3, CV_64FC1, m4_4));


    usedFilter = prewittFilter;



    char key = '-';
    while (key != 'q' && key != 'Q')
    {
        cheminRepImg = "images/";
        bonChemin = false;

        while (!bonChemin)
        {
            cheminImage = cheminRepImg;
            cheminImage += demanderImage();

            imgO = cv::imread (cheminImage, CV_LOAD_IMAGE_GRAYSCALE);

            if (imgO.empty ())
                cerr << "Mauvais chemin d'image: " << cheminImage << endl;
            else
                bonChemin = true;
        }

        equalizeHist(imgO, imgO);
        calculGradient(imgO);

        cv::imshow("Image original", imgO);
        cvWaitKey();
        cv::imshow("module", norme(module));

        cout << "Cliquez sur une fenetre d'OpenCv puis (q) pour pour quitter, (s) pour segmenter une nouvelle image." << endl;
        key = '-';
        while(key != 'q' && key != 's' && key != 'Q' && key != 'S') key = cvWaitKey(50);

        cvDestroyAllWindows();

        cout << endl << "-----------------------------------------------------------------" << endl << endl;
    }
}

cv::Mat norme(cv::Mat img)
{
    cv::Mat img_out(img.rows, img.cols, CV_8UC1);

    // Récupération des valeurs min et max de l'image
    double min = img.at<double>(0, 0);
    double max = img.at<double>(0, 0);
    for (int i = 0; i < img.rows ; i++) {
        for(int j = 0; j < img.cols ; j++) {
            if (min > img.at<double>(i, j)) min = img.at<double>(i, j);
            if (max < img.at<double>(i, j)) max = img.at<double>(i, j);
        }
    }

    // Normage de l'image pour avoir des valeurs entre [O, 255]
    for (int i = 0; i < img.rows ; i++) {
        for(int j = 0; j < img.cols ; j++) {
            img_out.at<uchar>(i, j) = round(((img.at<double>(i, j)-min)/(max-min)) * 255);
            //if(round(((img.at<double>(i, j)-min)/(max-min)) * 255) > 0)std::cout << round(((img.at<double>(i, j)-min)/(max-min)) * 255) << std::endl;
        }
    }

    return img_out;
}


#define GRADIENT_MAX 0
#define GRADIENT_SUM 1
void calculGradient(cv::Mat& img)
{
    int modeCalculGradient = 1;
    int nbDirection = 2;
    int angle = 180/nbDirection;


    module = cv::Mat(img.rows, img.cols, CV_64FC1);
    pente = cv::Mat(img.rows, img.cols, CV_64FC1);

    usedFilteredImg.push_back(applyFilter(img, usedFilter[0]));
    usedFilteredImg.push_back(applyFilter(img, usedFilter[1]));
    usedFilteredImg.push_back(applyFilter(img, usedFilter[2]));
    usedFilteredImg.push_back(applyFilter(img, usedFilter[3]));


    cv::imshow("usedFilteredImg 0 ", norme(usedFilteredImg[0]));
    cv::imshow("usedFilteredImg 1 ", norme(usedFilteredImg[1]));
    cvWaitKey();

    std::cout << "usedFilteredImg[0].rows = " << usedFilteredImg[0].rows << "   usedFilteredImg[0].cols = " << usedFilteredImg[0].cols << std::endl;
/*
    for(int i = 0 ; i < img.rows ; i++)
        for(int j = 0 ; j < img.cols ; j++)
            if(usedFilteredImg[0].at<double>(i,j) > 1) std::cout << usedFilteredImg[0].at<double>(i,j) << "    ";
    */


    switch(modeCalculGradient)
    {
        case GRADIENT_SUM:
            for(int i = 0 ; i < img.rows ; i++) {
                for(int j = 0 ; j < img.cols ; j++) {
                    double ksomme = 0;
                    int kmax = 0;
                    for(int k = 0 ; k < nbDirection ; k++) {
                        ksomme += usedFilteredImg[k].at<double>(i, j);
                        if(usedFilteredImg[k].at<double>(i, j) > usedFilteredImg[kmax].at<double>(i, j)) {
                            kmax = k;
                        }
                    }
                    module.at<double>(i, j) = ksomme / nbDirection;
                    //module.at<double>(i, j) = usedFilteredImg[ksomme].at<double>(i, j);

                    if(nbDirection== 2)
                    {
                        double angle = atan( usedFilteredImg[1].at<double>(i, j) / usedFilteredImg[0].at<double>(i, j));
                        angle *= 180 / M_PI;
                        //printf("angle %f \n", angle);
                        pente.at<double>(i, j) = angle;
                    }
                    else
                        pente.at<double>(i, j) = kmax * angle;
                }
            }
            break;

    case GRADIENT_MAX:
        for(int i = 0 ; i < img.rows ; i++) {
            for(int j = 0 ; j < img.cols ; j++) {
                int kmax = 0;
                for(int k = 0 ; k < nbDirection ; k++) {
                    if(usedFilteredImg[k].at<double>(i, j) > usedFilteredImg[kmax].at<double>(i, j)) {
                        kmax = k;
                    }
                }
                module.at<double>(i, j) = usedFilteredImg[kmax].at<double>(i, j);

                if(nbDirection == 2)
                {
                    double angle = 90;
                    if( usedFilteredImg[0].at<double>(i, j) != 0)
                    {
                        angle = atan( usedFilteredImg[1].at<double>(i, j) / usedFilteredImg[0].at<double>(i, j));
                        angle *= 180 / M_PI;

                        //if (angle > 90 )printf("angle %f \n", angle);
                            pente.at<double>(i, j) = angle;
                        }
                        else if (usedFilteredImg[0].at<double>(i, j) == 0)
                            angle = 0;

                        pente.at<double>(i, j) = angle;
                    }
                    else
                        pente.at<double>(i, j) = kmax * angle;
                }
            }
            break;

        default:
            std::cout << "Mauvais mode de gradient" << std::endl;

    }

}

string demanderImage()
{

    std::vector<string> images;

    images.push_back("Tux.png (256*256)");
    images.push_back("David.jpg (300*200)");
    images.push_back("Picasso.jpg (319*400)");

    images.push_back("Paris.jpg (400*296)");
    images.push_back("Dufy.jpg (400*316)");
    images.push_back("Vangogh.jpg (500*300)");

    images.push_back("tulipes.jpg (964*565)");
    images.push_back("Cartoon.jpg (1024*768)");
    images.push_back("arton553.jpg (1931*1931)");

    images.push_back("lena.jpg ");

    images.push_back("MIF23/B.JPG ");
    images.push_back("MIF23/B1.JPG ");
    images.push_back("MIF23/balls0.jpg ");

    images.push_back("MIF23/bally_0.jpg ");
    images.push_back("MIF23/bin.jpg ");
    images.push_back("MIF23/cake_0.JPG ");

    images.push_back("MIF23/purple_2_bw.jpg ");
    images.push_back("MIF23/rose_2_bw.JPG ");


    cout << "Entrez le numero de l'image a segmenter : " << endl;

    for(unsigned int i = 0; i < images.size(); i++) cout << i << " - " << images[i] << endl;

    while(true)
    {
        string tmp;
        cin >> tmp;
        unsigned int choix = atof(tmp.c_str());
        if(choix < images.size() && choix >= 0)
        {
            tmp = images[choix];
            tmp = tmp.substr(0,tmp.find(" "));
            cout << tmp << endl << endl;
            return tmp;
        }
        cout << "Le numero d'image choisit n'existe pas. " << tmp << " Re-essayez : " << endl;
    }

}

cv::Mat applyFilter(cv::Mat& img, cv::Mat& filtre)
{
    cv::Mat m2(img.rows, img.cols, CV_64FC1);

    //On parcours toute la matrice de l'image
    for (int i = 0 ; i < img.rows ; i++)
    {
        for (int j = 0 ; j < img.cols ; j++)
        {
            int ki;
            ki = i - filtre.rows/2;
            int imin = (ki < 0)? 0 : ki;
            ki = i + filtre.rows/2;
            int imax = (ki > img.rows)? img.rows : ki;

            int kj;
            kj = j - filtre.cols/2;
            int jmin = (kj < 0)? 0 : kj;
            kj = j + filtre.cols/2;
            int jmax = (kj > img.cols)? img.cols : kj;

            double sum = 0;

            ki = imin;
            //On applique le filtre à l'image
            while (ki <= imax)
            {
                kj = jmin;
                while (kj <= jmax)
                {
                    int fi = ki - i + filtre.rows/2;
                    int fj = kj - j + filtre.cols/2;

                    uchar pix = img.at<uchar>(ki, kj);
                    double coeff = filtre.at<double>(fi, fj);
                    sum += pix * coeff;

                    kj++;
                }
                ki++;
            }
            m2.at<double>(i, j) = abs(sum);
        }
    }
    return m2;
}
