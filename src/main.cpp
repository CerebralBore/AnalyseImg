#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>
#include <math.h>
#include <string>

using namespace std;

string demanderImage();

void calculGradient(cv::Mat& img, cv::Mat& module, cv::Mat& pente, int modeCalculGradient, int nbDirection);
cv::Mat seuillage(cv::Mat img, int type);
cv::Mat norme(cv::Mat img);
cv::Mat applyFilter(cv::Mat& img, cv::Mat& filtre);
cv::Mat affinage(cv::Mat amplitude, cv::Mat orientation);

std::vector<cv::Mat> prewittFilter, sobelFilter, kirschFilter, usedFilter;
std::vector<cv::Mat> usedFilteredImg;

int main()
{
    string cheminRepImg, cheminImage;
    cv::Mat img;
    cv::Mat module;
    cv::Mat pente;
    cv::Mat moduleSeuille;
    cv::Mat moduleAffinage;
    cv::Mat contours;

    bool bonChemin;

    double m2_1[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
    double m2_2[3][3] = {{-1, -1, 0}, {-1, 0, 1}, {0, 1, 1}};
    double m2_3[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    double m2_4[3][3] = {{0, 1, 1}, {-1, 0, 1}, {-1, -1, 0}};

    prewittFilter.push_back(cv::Mat(3, 3, CV_64FC1, m2_1));
    prewittFilter.push_back(cv::Mat(3, 3, CV_64FC1, m2_2));
    prewittFilter.push_back(cv::Mat(3, 3, CV_64FC1, m2_3));
    prewittFilter.push_back(cv::Mat(3, 3, CV_64FC1, m2_4));

    double m3_1[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    double m3_2[3][3] = {{-2, -1, 0}, {-1, 0, 1}, {0, 1, 2}};
    double m3_3[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    double m3_4[3][3] = {{0, 1, 2}, {-1, 0, 1}, {-2, -1, 0}};

    sobelFilter.push_back(cv::Mat(3, 3, CV_64FC1, m3_1));
    sobelFilter.push_back(cv::Mat(3, 3, CV_64FC1, m3_2));
    sobelFilter.push_back(cv::Mat(3, 3, CV_64FC1, m3_3));
    sobelFilter.push_back(cv::Mat(3, 3, CV_64FC1, m3_4));

    double m4_1[3][3] = {{-3, -3, -3}, {-3, 0, -3}, {5, 5, 5}};
    double m4_2[3][3] = {{-3, -3, -3}, {-3, 0, 5}, {-3, 5, 5}};
    double m4_3[3][3] = {{-3, -3, 5}, {-3, 0, 5}, {-3, -3, 5}};
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

            img = cv::imread (cheminImage, CV_LOAD_IMAGE_GRAYSCALE);

            if (img.empty ())
                cerr << "Mauvais chemin d'image: " << cheminImage << endl;
            else
                bonChemin = true;
        }


        module = cv::Mat(img.rows, img.cols, CV_64FC1);
        pente = cv::Mat(img.rows, img.cols, CV_64FC1);
        moduleSeuille = cv::Mat(img.rows, img.cols, CV_64FC1);
        moduleAffinage = cv::Mat(img.rows, img.cols, CV_64FC1);
        contours = cv::Mat(img.rows, img.cols, CV_64FC1);

        equalizeHist(img, img);

        calculGradient(img, module, pente, 0, 4);

        moduleSeuille = seuillage(module, 3);

        moduleAffinage = affinage (moduleSeuille, pente);

        //contours = generation_contours(moduleAffinage, module, pente);



        cv::imshow("Image original", img);

        cv::imshow("module", norme(module));

        cv::imshow("module seuille", moduleSeuille);

        cv::imshow("module Affine", moduleAffinage);

        cout << "Cliquez sur une fenetre d'OpenCv puis (q) pour pour quitter, (s) pour segmenter une nouvelle image." << endl;
        key = '-';
        while(key != 'q' && key != 's' && key != 'Q' && key != 'S') key = cvWaitKey(50);

        cvDestroyAllWindows();

        cout << endl << "-----------------------------------------------------------------" << endl << endl;

        usedFilteredImg.clear();

    }
}

cv::Mat affinage(cv::Mat amplitude, cv::Mat orientation)
{
    cv::Mat img_out(amplitude.rows, amplitude.cols, CV_64FC1);
    int ki, kj;
    for(int i = 0; i < amplitude.rows; i++)
        for(int j = 0; j < amplitude.cols; j++)
            img_out.at<double>(i, j) = 0;


    for(int i = 0; i < amplitude.rows; i++)
        for(int j = 0; j < amplitude.cols; j++)
        {
            double pix = amplitude.at<double>(i, j);
            if(pix != 0)
            {
                int angle;
                angle = (int)orientation.at<double>(i, j);
                if(angle < 45/2) angle = 0;
                else if (angle < 90 - (45 / 2)) angle = 45;
                else if (angle < 135 - (45 / 2)) angle = 90;
                else angle = 135;
                switch(angle)
                {
                    case 0:
                        ki = -1;
                        kj = 0;
                        break;

                    case 45:
                        ki = -1;
                        kj = 1;
                        break;

                    case 90:
                        ki = 0;
                        kj = 1;
                        break;

                    case 135:
                        ki = -1;
                        kj = -1;
                        break;
                default: //std::cout << angle << " ";
                         break;

                }
            }

            cv::Point2i suivant;
            suivant.x = i + ki;
            suivant.y = j + kj;

            double max = pix;
            cv::Point2i pointmax;

            //on parcourt dans un sens
            while( suivant.x > 0 && suivant.x < amplitude.rows && suivant.y > 0 && suivant.y < amplitude.cols
                   && amplitude.at<double>(suivant.x, suivant.y) > 0 )
            {
                if( amplitude.at<double>(suivant.x, suivant.y) > max )
                {
                    max = amplitude.at<double>(suivant.x, suivant.y);
                    pointmax.x = suivant.x;
                    pointmax.y = suivant.y;
                }

                suivant.x += ki;
                suivant.y += kj;
            }

            suivant.x = i - ki;
            suivant.y = j - kj;

            //puis dans l'autre
            while( suivant.x > 0 && suivant.x < amplitude.rows && suivant.y > 0 && suivant.y < amplitude.cols
                   && amplitude.at<double>(suivant.x, suivant.y) > 0 )
            {
                if( amplitude.at<double>(suivant.x, suivant.y) > max )
                {
                    max = amplitude.at<double>(suivant.x, suivant.y);
                    pointmax.x = suivant.x;
                    pointmax.y = suivant.y;
                }

                suivant.x -= ki;
                suivant.y -= kj;
            }

        img_out.at<double>(pointmax.x, pointmax.y) = max;

        }

    return img_out;
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
            img_out.at<uchar>(i, j) = round(((img.at<double>(i, j) - min) / ( max - min)) * 255);
            //if(round(((img.at<double>(i, j)-min)/(max-min)) * 255) > 0)std::cout << round(((img.at<double>(i, j)-min)/(max-min)) * 255) << std::endl;
        }
    }

    return img_out;
}


#define GRADIENT_MAX 0
#define GRADIENT_SUM 1
void calculGradient(cv::Mat& img, cv::Mat& module, cv::Mat& pente, int modeCalculGradient, int nbDirection)
{
    int angle = 180 / nbDirection;

    usedFilteredImg.push_back(applyFilter(img, usedFilter[0]));
    usedFilteredImg.push_back(applyFilter(img, usedFilter[1]));
    usedFilteredImg.push_back(applyFilter(img, usedFilter[2]));
    usedFilteredImg.push_back(applyFilter(img, usedFilter[3]));

    switch(modeCalculGradient)
    {
        case GRADIENT_SUM:
                            for(int i = 0 ; i < img.rows ; i++)
                            {
                                for(int j = 0 ; j < img.cols ; j++)
                                {
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

                                    if(nbDirection == 2)
                                    {
                                        double angle = atan( usedFilteredImg[2].at<double>(i, j) / usedFilteredImg[0].at<double>(i, j));
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
                        for(int i = 0 ; i < img.rows ; i++)
                        {
                            for(int j = 0 ; j < img.cols ; j++)
                            {
                                int kmax = 0;
                                for(int k = 0 ; k < nbDirection ; k++)
                                    if(usedFilteredImg[k].at<double>(i, j) > usedFilteredImg[kmax].at<double>(i, j))
                                        kmax = k;

                                module.at<double>(i, j) = usedFilteredImg[kmax].at<double>(i, j);

                                if(nbDirection == 2)
                                {
                                    double angle = 90;
                                    if( usedFilteredImg[0].at<double>(i, j) != 0)
                                    {
                                        angle = atan( usedFilteredImg[2].at<double>(i, j) / usedFilteredImg[0].at<double>(i, j));
                                        angle *= 180 / M_PI;
                                        //printf("angle %f \n", angle);
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

cv::Mat seuillage(cv::Mat img, int type)
{
    cv::Mat img_out(img.rows, img.cols, CV_64FC1);
    int valCumule = 0;
    double seuil;
    double valeurMin = 255;
    double valeurMax = 0;
    int xMin, yMin, xMax, yMax, nouvelleLargeur, nouvelleHauteur;
    int largeur = 3;
    int hauteur = 3;

    double sum = 0;
    double avg = 0;
    double variance = 0;
    double seuil_h = 0;
    double seuil_b = 0;

    switch(type)
    {
        case 0 :    // seuillage simple
                    for (int i = 0; i < img.rows ; i++)
                    {
                        for(int j = 0; j < img.cols ; j++)
                        {
                            if (img.at<double>(i, j) > 150)
                                img_out.at<double>(i, j) = img.at<double>(i, j);
                            else img_out.at<double>(i, j) = 0;
                        }
                    }
                    break;
        case 1 :    // seuillage global
                    for(int i = 0; i < img.rows; i++)
                        for(int j = 0; j < img.cols; j++)
                        {
                            valCumule += img.at<double>(i,j);
                            /*if(img.at<double>(i,j) > valeurMax)
                                valeurMax = img.at<double>(i,j);
                            if(img.at<double>(i,j) < valeurMin)
                                valeurMin = img.at<double>(i,j);*/
                        }
                    /*seuilGlobal = (valeurMax + valeurMin) / 2;
                    std::cout << "valeurMin " << valeurMin << std::endl;
                    std::cout << "valeurMax " << valeurMax << std::endl;*/
                    seuil = valCumule / (img.rows * img.cols);
                    std::cout << "seuilGlobal " << seuil << std::endl;

                    for (int i = 0; i < img.rows ; i++)
                    {
                        for(int j = 0; j < img.cols ; j++)
                        {
                            if (img.at<double>(i, j) > seuil)
                                img_out.at<double>(i, j) = img.at<double>(i, j);
                            else img_out.at<double>(i, j) = 0;
                        }
                    }
                    break;
        case 2 :    // seuillage local
                    for(int i = 0; i < img.rows; i++)
                        for(int j = 0; j < img.cols; j++)
                        {
                            valCumule = 0;

                            xMin = j < (largeur / 2) ? 0 : j - (largeur / 2);
                            yMin = i < (hauteur / 2) ? 0 : i - (hauteur / 2);
                            xMax = (j + (largeur / 2) >= img.cols) ? img.cols - 1 : j + (largeur / 2);
                            yMax = (i + (hauteur / 2) >= img.rows) ? img.rows - 1 : i + (hauteur / 2);
                            //std::cout << "xMin " << xMin << ", yMin " << yMin << ", xMax " << xMax << ", yMax " << yMax << std::endl;
                            nouvelleLargeur = xMax - xMin + 1;
                            nouvelleHauteur = yMax - yMin + 1;

                            for(int k = yMin; k < yMax; k++)
                                for(int l = xMin; l < xMax; l++)
                                    valCumule += img.at<double>(i,j);
                            //std::cout << "taille " << nouvelleLargeur * nouvelleHauteur << std::endl;

                            seuil = valCumule / (nouvelleLargeur * nouvelleHauteur);

                            //cv::cout << "seuil " << seuil << std::endl;
                            if (img.at<double>(i, j) > seuil && seuil > 30)
                                img_out.at<double>(i, j) = img.at<double>(i, j);
                            else img_out.at<double>(i, j) = 0;
                        }
                    break;
        case 3 :    //seuillage hysteresis

                    for (int i = 0; i < img.rows ; i++)
                        for(int j = 0; j < img.cols ; j++)
                            sum += img.at<double>(i, j);

                    avg = sum / (img.rows*img.cols);

                    // Calcul de la variance
                    for (int i = 0; i < img.rows ; i++)
                        for(int j = 0; j < img.cols ; j++)
                            variance += pow(img.at<double>(i, j) - avg, 2);
                    variance /= (img.rows * img.cols);

                    seuil_h = avg + sqrt(variance);
                    seuil_b = MAX(0, avg);

                    //Seuil haut
                    for (int i = 0; i < img.rows ; i++)
                        for(int j = 0; j < img.cols ; j++)
                        {
                            if (img.at<double>(i, j) > seuil_h)
                                img_out.at<double>(i, j) = img.at<double>(i, j);
                            else img_out.at<double>(i, j) = 0;
                        }

                    //Seuil bas
                    for (int i = 0 ; i < img.rows ; i++)
                        for (int j = 0 ; j < img.cols ; j++)
                        {
                            int ki;
                            ki = i - 3/2;
                            int imin = (ki < 0)? 0 : ki;
                            ki = i + 3/2;
                            int imax = (ki > img.rows) ? img.rows : ki;

                            int kj;
                            kj = j - 3/2;
                            int jmin = (kj < 0) ? 0 : kj;
                            kj = j + 3/2;
                            int jmax = (kj > img.cols) ? img.cols : kj;

                            if (img.at<double>(i, j) > seuil_b)
                            {
                                ki = imin;
                                while (ki <= imax)
                                {
                                    kj = jmin;
                                    while (kj <= jmax)
                                    {
                                        if (img_out.at<double>(ki, kj) != 0)
                                        {
                                            img_out.at<double>(i, j) = img.at<double>(i, j);
                                            break;
                                        }
                                        kj++;
                                    }
                                    ki++;
                                }
                            }
                            else img_out.at<double>(i, j) = 0;
                        }
                    break;
        default:
                    break;

    }

    return img_out;

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
