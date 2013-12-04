#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>
#include <math.h>
#include <string>
#include <ctime>

struct Contour{
    std::vector<cv::Point2i> pix;
    std::vector<cv::Point2i> extremites;
};


struct LineHough{
    int ro;
    int teta;
    int min;
    int max;
};

struct CercleHough{
    int x;
    int y;
    int r;
};

std::string demanderImage();

void calculGradient(cv::Mat& img, cv::Mat& module, cv::Mat& pente, int modeCalculGradient, int nbDirection);
cv::Mat seuillage(cv::Mat img, int type);
cv::Mat norme(cv::Mat img);
cv::Mat applyFilter(cv::Mat& img, cv::Mat& filtre);
cv::Mat affinage(cv::Mat amplitude, cv::Mat orientation);
cv::Mat parcours(cv::Mat img_seuille, cv::Mat orientation, std::vector<Contour>& contours);
void etendre_contour(int i, int j, cv::Mat img_seuille, cv::Mat orientation, cv::Mat& imgContour, std::vector<Contour>& contours);
void recherche_extremites(cv::Mat& imgContour, std::vector<Contour>& contours);
void amelioration_contours(cv::Mat& imgContour, cv::Mat& orientation, std::vector<Contour>& contours);
void merge_contour(cv::Mat& imgContour, std::vector<Contour>& contours, int i, int j);
bool test_validite_merge(int a_y, int a_x, int b_y, int b_x, double angle_b);
cv::Mat recuperationPix(cv::Mat imgContour, std::vector<cv::Point2i> &listePoints);
cv::Mat hough(std::vector<cv::Point2i> &listePoints, int hauteur);
cv::Mat imageHough(cv::Mat imgH);
cv::Mat seuillageHough(cv::Mat imghoughNS, std::vector<cv::Point2i> &listePoints, std::vector<LineHough> &listesDroitesHough);
cv::Mat imageDroitesHough(std::vector<LineHough>& listesDroitesHough, int hauteur, int largeur);
cv::Mat imageToutesLesDroitesHough(cv::Mat imghoughNS, int hauteur, int largeur);
void houghCercle(std::vector<cv::Point2i> &listePoints, int* listeCercles, int sizeX, int sizeY, int sizeR);
cv::Mat seuillageCercleHough(int* listeCercles, int sizeX, int sizeY, int sizeR, std::vector<CercleHough> listesCerclesHough);


std::vector<cv::Mat> prewittFilter, sobelFilter, kirschFilter, usedFilter;

int main()
{
    std::string cheminRepImg, cheminImage;
    cv::Mat img;
    cv::Mat module;
    cv::Mat pente;
    cv::Mat moduleSeuille;
    cv::Mat moduleAffinage;
    cv::Mat imgContours;
    cv::Mat moduleAffinageAff;
    cv::Mat imgPix;
    cv::Mat imgHough;
    cv::Mat imgHoughSeuille;
    cv::Mat imgDroitesHough;
    cv::Mat imgToutesLesDroitesHough;
    cv::Mat imgCerclesHough;
    std::vector<Contour> contours;
    std::vector<cv::Point2i> listePoints;
    std::vector<LineHough> listesDroitesHough;
    std::vector<CercleHough> listesCerclesHough;
    bool bonChemin;

    /*int filterType = 0;
    int gradientType = 0;
    int directionType = 0;
    int seuillageType = 0;
    int angleAff;*/
    int hauteurValeur;

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
                std::cerr << "Mauvais chemin d'image: " << cheminImage << std::endl;
            else
                bonChemin = true;
        }

        module = cv::Mat(img.rows, img.cols, CV_64FC1);
        pente = cv::Mat(img.rows, img.cols, CV_64FC1);
        moduleSeuille = cv::Mat(img.rows, img.cols, CV_64FC1);
        moduleAffinage = cv::Mat(img.rows, img.cols, CV_64FC1);
        imgContours = cv::Mat(img.rows, img.cols, CV_64FC1);
        moduleAffinageAff = cv::Mat(img.rows, img.cols, CV_32FC3);

        equalizeHist(img, img);
        cv::imshow("Image originale", img);
        /*medianBlur(img, img, 3);
        cv::imshow("Image originale floue", img);

        key = '-';
        std::cout << "---------------------------" << std::endl << "choix du filtre" << std::endl;
        std::cout << "p : prewitt" << std::endl << "s : sobel" << std::endl << "k : kirsch" << std::endl;
        while(key != 'p' && key != 's' && key != 'k') key = cvWaitKey(50);
        if(key == 'p')
            filterType = 0;
        else if(key == 's')
            filterType = 1;
        else
            filterType = 2;

        switch(filterType)
        {
        case 0:
            usedFilter = prewittFilter;
            break;
        case 1:
            usedFilter = sobelFilter;
            break;
        case 2:
            usedFilter = kirschFilter;
            break;
        default:
            usedFilter = prewittFilter;
            break;
        }

        key = '-';
        std::cout << "choix du calcul du gradient" << std::endl;
        std::cout << "s : somme" << std::endl << "m : maximum" << std::endl;
        while(key != 's' && key != 'm') key = cvWaitKey(50);
        if(key == 'm')
            gradientType = 0;
        else
            gradientType = 1;

        key = '-';
        std::cout << "choix du nombre de direction" << std::endl;
        std::cout << "b : bidirectionnel" << std::endl << "m : multidirectionnel" << std::endl;
        while(key != 'b' && key != 'm') key = cvWaitKey(50);
        if(key == 'b')
            directionType = 2;
        else
            directionType = 4;

        calculGradient(img, module, pente, gradientType, directionType);
        cv::imshow("module", norme(module));

        key = '-';
        std::cout << "---------------------------" << std::endl << "choix du seuillage" << std::endl;
        std::cout << "s : seuil simple" << std::endl << "g : seuil global" << std::endl << "l : seuil local" << std::endl << "h : seuil hysteresis" << std::endl;
        while(key != 's' && key != 'g' && key != 'l' && key != 'h') key = cvWaitKey(50);
        if(key == 's')
            seuillageType = 0;
        else if(key == 'g')
            seuillageType = 1;
        else if(key == 'l')
            seuillageType = 2;
        else
            seuillageType = 3;

        moduleSeuille = seuillage(module, seuillageType);
        cv::imshow("module seuille", moduleSeuille);

        moduleAffinage = affinage(moduleSeuille, pente);
        for(int i = 0; i < moduleAffinageAff.rows; i ++)
            for(int j = 0; j < moduleAffinageAff.cols; j++)
            {
                if(moduleAffinage.at<double>(i,j) > 0)
                {
                    angleAff = (int)pente.at<double>(i, j);
                    if(angleAff >=0 && angleAff < 45/2) moduleAffinageAff.at<cv::Vec3f>(i,j) = cv::Vec3f(1,1,1);
                    else if (angleAff < 90 - (45 / 2)) moduleAffinageAff.at<cv::Vec3f>(i,j) = cv::Vec3f(0,1,1);
                    else if (angleAff < 135 - (45 / 2)) moduleAffinageAff.at<cv::Vec3f>(i,j) = cv::Vec3f(0,0,1);
                    else if (angleAff < 180 - (45 / 2)) moduleAffinageAff.at<cv::Vec3f>(i,j) = cv::Vec3f(1,0,1);
                    else if (angleAff < 225 - (45 / 2)) moduleAffinageAff.at<cv::Vec3f>(i,j) = cv::Vec3f(1,0,0);
                    else if (angleAff < 270 - (45 / 2)) moduleAffinageAff.at<cv::Vec3f>(i,j) = cv::Vec3f(1,1,0);
                    else if (angleAff < 315 - (45 / 2)) moduleAffinageAff.at<cv::Vec3f>(i,j) = cv::Vec3f(0,1,0);
                    else if (angleAff < 360 - (45 / 2)) moduleAffinageAff.at<cv::Vec3f>(i,j) = cv::Vec3f(0.5,0.5,0.5);
                    else moduleAffinageAff.at<cv::Vec3f>(i,j) = cv::Vec3f(1,1,1);
                }
                else
                    moduleAffinageAff.at<cv::Vec3f>(i,j) = cv::Vec3f(0,0,0);
            }
        cv::imshow("module Affine", moduleAffinageAff);*/
        //--
        moduleAffinage = img;
        //--
        cv::imshow("module Affine", moduleAffinage);

        std::cout << "l : lignes hough" << std::endl << "c : cercle hough" << std::endl;

        key = '-';
        while(key != 'L' && key != 'l' && key != 'c' && key != 'C') key = cvWaitKey(50);
        if(key == 'l' || key == 'L')
        {
            hauteurValeur = sqrt(pow(moduleAffinage.rows, 2) + pow(moduleAffinage.cols, 2));

            imgPix = recuperationPix(moduleAffinage, listePoints);
            cv::imshow("imgPix", norme(imgPix));

            imgHough = hough(listePoints, hauteurValeur);

            cv::imshow("hough", norme(imgHough));

            imgToutesLesDroitesHough = imageToutesLesDroitesHough(imgHough, moduleAffinage.rows, moduleAffinage.cols);
            cv::imshow("toutes les droites de hough", norme(imgToutesLesDroitesHough));

            imgHoughSeuille = seuillageHough(imgHough, listePoints, listesDroitesHough);
            cv::imshow("hough seuille", norme(imgHoughSeuille));

            std::cout << "nombre de droites de hough: " << listesDroitesHough.size() << std::endl;

            imgDroitesHough = imageDroitesHough(listesDroitesHough, img.rows, img.cols);
            cv::imshow("image droite hough", norme(imgDroitesHough));
            std::cout << "Droite de Hough : fini" << std::endl;
        }
        if(key == 'c' || key == 'C')
        {
            imgPix = recuperationPix(moduleAffinage, listePoints);
            cv::imshow("imgPix", norme(imgPix));

            int sizeX = img.rows;
            int sizeY = img.cols;
            int sizeR = round(sqrt(pow(img.rows,2) + pow(img.cols, 2)));

            std::cout << "x: " << sizeX << " | y: " << sizeY << " | R: " << sizeR << " = " << sizeX * sizeY * sizeR << std::endl;
            int* listeCercles = new int[sizeX * sizeY * sizeR];

            std::cout << "Cercle de Hough" << std::endl;
            houghCercle(listePoints, listeCercles, sizeX, sizeY, sizeR);
            std::cout << "SeuillageCercles " << std::endl;
            imgCerclesHough = seuillageCercleHough(listeCercles, sizeX, sizeY, sizeR, listesCerclesHough);

            cv::imshow("image cercle hough", norme(imgCerclesHough));
            std::cout << "Cercle de Hough : fini" << std::endl;
            free(listeCercles);
        }


        /*imgContours = parcours(moduleAffinage, pente, contours);
        cv::imshow("contours", imgContours);

        std::cout << "Cliquez sur une fenetre d'OpenCv puis (q) pour pour quitter." << std::endl;

        for(int i = 0; i < contours.size(); i++)
        {
            contours[i].pix.clear();
            contours[i].extremites.clear();
        }
        contours.clear();*/

        key = '-';
        while(key != 'q' /*&& key != 's'*/ && key != 'Q' /*&& key != 'S'*/) key = cvWaitKey(50);

        cvDestroyAllWindows();

        std::cout << std::endl << "-----------------------------------------------------------------" << std::endl << std::endl;
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------------

cv::Mat imageDroitesHough(std::vector<LineHough>& listesDroitesHough, int hauteur, int largeur)
{
    cv::Mat imgDroitesHough = cv::Mat(hauteur, largeur, CV_64FC1);
    int x1, y1, x2, y2;
    double a, b;

    for(unsigned int i = 0; i < listesDroitesHough.size(); i++)
    {
        a = cos(listesDroitesHough.at(i).teta / 360.0 * M_PI);
        b = sin(listesDroitesHough.at(i).teta / 360.0 * M_PI);

        if((listesDroitesHough.at(i).teta >= -90 && listesDroitesHough.at(i).teta <= 90) || listesDroitesHough.at(i).teta >= 270)
        {
            y1 = listesDroitesHough.at(i).min;
            x1 = (listesDroitesHough.at(i).ro - (y1 * b)) / a;

            y2 = listesDroitesHough.at(i).max;
            x2 = (listesDroitesHough.at(i).ro - (y2 * b)) / a;
        }
        else
        {
            x1 = listesDroitesHough.at(i).min;
            y1 = (listesDroitesHough.at(i).ro - (x1 * a)) / b;

            x2 = listesDroitesHough.at(i).max;
            y2 = (listesDroitesHough.at(i).ro - (x2 * a)) / b;
        }

        cv::line(imgDroitesHough, cv::Point(y1, x1), cv::Point(y2, x2), cv::Scalar( 1, 1, 1 ), 1, 8);
    }
    return imgDroitesHough;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

cv::Mat seuillageHough(cv::Mat imghoughNS, std::vector<cv::Point2i> &listePoints, std::vector<LineHough>& listesDroitesHough)
{
    cv::Mat imgHoughS = cv::Mat(imghoughNS.rows, imghoughNS.cols, CV_64FC1);

    double ro, roMax, max, teta, tetaMax;
    std::vector<int> elem;
    int jMax;
    std::cout << "nombre de points hough" << listePoints.size() << std::endl;

    for(unsigned int i = 0; i < listePoints.size(); i++)
    {
        max = 0;
        elem.clear();
        for(int j = 0; j <= (2 * 270); j++)
        {
            teta = -M_PI/2 + (M_PI / 360.0) * j;
            ro = listePoints.at(i).x * cos(teta) + listePoints.at(i).y * sin(teta);
            if(ro > 0)
            {
                if(imghoughNS.at<double>((int)round(ro), j) > max)
                {
                    max = imghoughNS.at<double>((int)round(ro), j);
                    roMax = ro;
                    tetaMax = teta;
                    jMax = j;
                }
            }
        }
        for(int j = 0; j <= (2 * 270); j++)
        {
            teta = -M_PI/2 + (M_PI / 360.0) * j;
            ro = listePoints.at(i).x * cos(teta) + listePoints.at(i).y * sin(teta);
            if(ro > 0)
            {
                if(imghoughNS.at<double>((int)round(ro), j) == max)
                    elem.push_back(j);
            }
        }
        for(unsigned int k = 0; k < elem.size(); k++)
        {
            jMax = elem.at(k);
            tetaMax = -M_PI/2 + (M_PI / 360.0) * jMax;
            roMax = listePoints.at(i).x * cos(tetaMax) + listePoints.at(i).y * sin(tetaMax);

            if(imgHoughS.at<double>((int)round(roMax), jMax) == 0)
            {
                std::cout << jMax << ", ";
                LineHough newLineHough;
                newLineHough.ro = (int)round(roMax);
                newLineHough.teta = jMax - 180;
                if((jMax >= 90 && jMax <= 270) || jMax >= 450)
                {
                    newLineHough.min = listePoints.at(i).y;
                    newLineHough.max = listePoints.at(i).y;
                }
                else
                {
                    newLineHough.min = listePoints.at(i).x;
                    newLineHough.max = listePoints.at(i).x;
                }
                listesDroitesHough.push_back(newLineHough);
                imgHoughS.at<double>((int)round(roMax), jMax) = 1;
            }
            else
            {
                unsigned int j = 0;
                while(j < listesDroitesHough.size())
                {
                    if(listesDroitesHough.at(j).ro == (int)round(roMax) && listesDroitesHough.at(j).teta == jMax - 180)
                    {
                        if((jMax >= 90 && jMax <= 270) || jMax >= 450)
                        {
                            listesDroitesHough.at(j).min = std::min(listePoints.at(i).y, listesDroitesHough.at(j).min);
                            listesDroitesHough.at(j).max = std::max(listePoints.at(i).y, listesDroitesHough.at(j).max);
                        }
                        else
                        {
                            listesDroitesHough.at(j).min = std::min(listePoints.at(i).x, listesDroitesHough.at(j).min);
                            listesDroitesHough.at(j).max = std::max(listePoints.at(i).x, listesDroitesHough.at(j).max);
                        }
                    }
                    j++;
                }
            }
        }
    }
    return imgHoughS;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

cv::Mat imageToutesLesDroitesHough(cv::Mat imghoughNS, int hauteur, int largeur)
{
    cv::Mat imgRes = cv::Mat(hauteur, largeur, CV_64FC1);

    double ro, teta, y, x, a, b;

    for(int i = 0; i < imghoughNS.rows; i++)
        for(int j = 0; j < imghoughNS.cols; j++)
        {
            if(imghoughNS.at<double>(i, j) != 0)
            {
                ro = i;
                teta = j;
                teta -= 180;

                a = cos(teta / 360.0 * M_PI);
                b = sin(teta / 360.0 * M_PI);

                if((teta >= -90 && teta <= 90) || teta >= 270)
                {
                    //std::cout << "y" << std::endl;
                    for(y = 0; y < imgRes.cols; y++)
                    {
                        x = round((ro - (y * b)) / a);
                        //std::cout << "x :" << x << ", y :" << y << std::endl;
                        if(x >= 0 && x < imgRes.rows)
                            if(imgRes.at<double>(x, y) < imghoughNS.at<double>(i, j))
                                imgRes.at<double>(x, y) = imghoughNS.at<double>(i, j);
                    }
                }
                else
                {
                    //std::cout << "x" << std::endl;
                    for(x = 0; x < imgRes.rows; x++)
                    {
                        y = round((ro - (x * a)) / b);
                        //std::cout << "x :" << x << ", y :" << y << std::endl;
                        if(y >= 0 && y < imgRes.cols)
                            if(imgRes.at<double>(x, y) < imghoughNS.at<double>(i, j))
                                imgRes.at<double>(x, y) = imghoughNS.at<double>(i, j);
                    }
                }
            }
        }
    return imgRes;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

cv::Mat hough(std::vector<cv::Point2i> &listePoints, int hauteur)
{
    cv::Mat imgHough = cv::Mat(hauteur, (2 * 270 + 1), CV_64FC1);
    double ro, teta;
    for(unsigned int i = 0; i < listePoints.size(); i++)
        for(int j = 0; j <= (2 * 270); j++)
        {
            teta = -M_PI/2 + (M_PI / 360.0) * j;
            ro = listePoints.at(i).x * cos(teta) + listePoints.at(i).y * sin(teta);
            if(ro > 0)
                imgHough.at<double>((int)round(ro), j)++;
        }
    return imgHough;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

cv::Mat seuillageCercleHough(int* listeCercles, int sizeX, int sizeY, int sizeR, std::vector<CercleHough> listesCerclesHough)
{
    cv::Mat CercleSeuille = cv::Mat(sizeX, sizeY, CV_64FC1);
    cv::Mat imgAccCercle = cv::Mat(sizeX, sizeY, CV_64FC1);

    std::cout << "Définissez un seuil: " << std::endl;
    std::string tmp;
    std::cin >> tmp;
    int seuil = atof(tmp.c_str());

    int max = 0;

    int xMax, yMax, rMax/*, enregistre*/;

    for(int x = 0; x < sizeX; x++)
        for(int y = 0; y < sizeY; y++)
            for(int r = 0; r < sizeR; r++)
            {
                /*enregistre = 1;
                if((r < sizeR - 1) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[x * sizeR * sizeY + y * sizeR + (r + 1)]))
                    enregistre = 0;
                if((r < sizeR - 1) && (y < sizeY - 1) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[x * sizeR * sizeY + (y + 1) * sizeR + (r + 1)]))
                    enregistre = 0;
                if((y < sizeY - 1) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[x * sizeR * sizeY + (y + 1) * sizeR + r]))
                    enregistre = 0;
                if((r > 0) && (y < sizeY - 1) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[x * sizeR * sizeY + (y + 1) * sizeR + (r - 1)]))
                    enregistre = 0;

                if((x < sizeX - 1) && (r > 0) && (y > 0) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[(x + 1) * sizeR * sizeY + (y - 1) * sizeR + (r - 1)]))
                    enregistre = 0;
                if((x < sizeX - 1) && (r > 0) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[(x + 1) * sizeR * sizeY + y * sizeR + (r - 1)]))
                    enregistre = 0;
                if((x < sizeX - 1) && (r > 0) && (y < sizeY - 1) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[(x + 1) * sizeR * sizeY + (y + 1) * sizeR + (r - 1)]))
                    enregistre = 0;
                if((x < sizeX - 1) && (y > 0) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[(x + 1) * sizeR * sizeY + (y - 1) * sizeR + r]))
                    enregistre = 0;
                if((x < sizeX - 1) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[(x + 1) * sizeR * sizeY + y * sizeR + r]))
                    enregistre = 0;
                if((x < sizeX - 1) && (y < sizeY - 1) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[(x + 1) * sizeR * sizeY + (y + 1) * sizeR + r]))
                    enregistre = 0;
                if((x < sizeX - 1) && (r < sizeR - 1) && (y > 0) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[(x + 1) * sizeR * sizeY + (y - 1) * sizeR + (r + 1)]))
                    enregistre = 0;
                if((x < sizeX - 1) && (r < sizeR - 1) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[(x + 1) * sizeR * sizeY + y * sizeR + (r + 1)]))
                    enregistre = 0;
                if((x < sizeX - 1) && (r < sizeR - 1) && (y < sizeY - 1) && (listeCercles[x * sizeR * sizeY + y * sizeR + r] <= listeCercles[(x + 1) * sizeR * sizeY + (y + 1) * sizeR + (r + 1)]))
                    enregistre = 0;

                if(enregistre)
                {
                    std::cout << x << " " << y << " " << r << std::endl;
                    unsigned int i;
                    for(i = 0; i < listesCerclesHough.size(); i++)
                    {
                        if(listesCerclesHough[i].x == x && listesCerclesHough[i].y == y && listesCerclesHough[i].r == r)
                            break;
                    }
                    std::cout << i << " " << listesCerclesHough.size() << std::endl;
                    if(i == listesCerclesHough.size())
                    {
                        CercleHough monCercle;
                        monCercle.x = x;
                        monCercle.y = y;
                        monCercle.r = r;
                        listesCerclesHough.push_back(monCercle);
                    }
                }*/
                if(listeCercles[x * sizeR * sizeY + y * sizeR + r] > seuil)
                {
                    CercleHough monCercle;
                    monCercle.x = x;
                    monCercle.y = y;
                    monCercle.r = r;
                    listesCerclesHough.push_back(monCercle);
                }
                if(listeCercles[x * sizeR * sizeY + y * sizeR + r] > max)
                {
                    max = listeCercles[x * sizeR * sizeY + y * sizeR + r];
                    xMax = x;
                    yMax = y;
                    rMax = r;
                }
            }

    for(unsigned int i = 0; i < listesCerclesHough.size(); i++)
    {
        //std::cout << "vfduj " << listesCerclesHough[i].x << " " << listesCerclesHough[i].y << " " << listesCerclesHough[i].r << std::endl;
        cv::circle(CercleSeuille, cv::Point(listesCerclesHough[i].y, listesCerclesHough[i].x), listesCerclesHough[i].r, cv::Scalar(1.0, 1.0, 1.0), 1, 8, 0);
    }

    if(max > 20)
    {
        cv::circle(CercleSeuille, cv::Point(yMax, xMax), rMax, cv::Scalar(1.0, 1.0, 1.0), 1, 8, 0);
        for(int x = 0; x < sizeX; x++)
            for(int y = 0; y < sizeY; y++)
            {
                imgAccCercle.at<double>(x, y) = listeCercles[x * sizeR * sizeY + y * sizeR + rMax];
            }
        cv::imshow("accumulateur cercle au meme r", norme(imgAccCercle));
    }
    //std::cout << "vfduj " << xMax << " " << yMax << " " << rMax << std::endl;
    return CercleSeuille;

}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

void houghCercle(std::vector<cv::Point2i> &listePoints, int* listeCercles, int sizeX, int sizeY, int sizeR)
{
    double distance_i;
    for(int x = 0; x < sizeX; x++)
        for(int y = 0; y < sizeY; y++)
            for(unsigned int i = 0; i < listePoints.size(); i++)
            {
                distance_i = round(sqrt(pow((double)x - (double)listePoints[i].x, 2) + pow((double)y - (double)listePoints[i].y, 2)));
                listeCercles[x * sizeR * sizeY + y * sizeR + (int)distance_i ]++;
            }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

cv::Mat recuperationPix(cv::Mat imgContour, std::vector<cv::Point2i> &listePoints)
{
    cv::Mat imgOut = cv::Mat(imgContour.rows, imgContour.cols, CV_64FC1);
    for(int i = 0; i < imgContour.rows; i ++)
        for(int j = 0; j < imgContour.cols; j++)
            if(imgContour.at<uchar>(i,j) > 0)
            {
                cv::Point2i p = cv::Point2i(i, j);
                listePoints.push_back(p);
            }
    for(unsigned int i = 0; i < listePoints.size(); i++)
        imgOut.at<double>(listePoints.at(i).x, listePoints.at(i).y)++;
    return imgOut;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------------

void merge_contour(cv::Mat& imgContour, std::vector<Contour>& contours, int i, int j)
{
    for(unsigned int m = 0; m < contours[j].pix.size(); m++)
    {
        imgContour.at<double>(contours[j].pix[m].x, contours[j].pix[m].y) = i;
        contours[i].pix.push_back(contours[j].pix[m]);
    }
    contours[j].pix.clear();

    for(unsigned int n = 0; n < contours[j].extremites.size(); n++)
        contours[i].extremites.push_back(contours[j].extremites[n]);
    contours[j].extremites.clear();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

double distance(int a_x, int a_y, int b_x, int b_y)
{
    return(sqrt((double)pow(b_x - a_x,2) + (double)pow(b_y - a_y, 2)));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

bool test_validite_merge(int a_y, int a_x, int b_y, int b_x, double angle_b)
{
    double d1 = distance(a_x, a_y, b_x, b_y);
    double d2 = distance(a_x, a_y, a_x + 1, a_y);
    double d3 = distance(a_x + 1, a_y, b_x, b_y);

    double angle_diff = acos((pow(d1,2) + pow(d2,2) - pow(d3,2)) / (2.0 * d1 * d2)) * 180.0 / M_PI;
    if(b_y > a_y)
        angle_diff = 360 - angle_diff;
    if((angle_diff <= (angle_b + (45.0)) && angle_diff >= (angle_b - (45.0)))
            || (angle_diff <= (angle_b + (45.0) + 180) && angle_diff >= (angle_b - (45.0) + 180))
            || (angle_diff <= (angle_b + (45.0) - 180) && angle_diff >= (angle_b - (45.0) - 180))
            || (angle_diff <= (angle_b + (45.0) + 360) && angle_diff >= (angle_b - (45.0) + 360)))
        return true;
    return false;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

void amelioration_contours(cv::Mat &imgContour, cv::Mat& orientation, std::vector<Contour> &contours)
{
    int i = 0;
    int compteur = 0;

    int xMin, yMin, xMax, yMax, nouvelleLargeur, nouvelleHauteur, alpha;
    bool merge;

    int largeur = 9;
    int hauteur = 9;

    while (i < contours.size())
    {
        for(unsigned int k = 0; k < contours[i].extremites.size(); k++)
        {
            xMin = contours[i].extremites[k].x < (largeur / 2) ? 0 : contours[i].extremites[k].x - (largeur / 2);
            yMin = contours[i].extremites[k].y < (hauteur / 2) ? 0 : contours[i].extremites[k].y - (hauteur / 2);
            xMax = (contours[i].extremites[k].x + (largeur / 2) >= imgContour.cols) ? imgContour.cols - 1 : contours[i].extremites[k].x + (largeur / 2);
            yMax = (contours[i].extremites[k].y + (hauteur / 2) >= imgContour.rows) ? imgContour.rows - 1 : contours[i].extremites[k].y + (hauteur / 2);

            nouvelleLargeur = xMax - xMin + 1;
            nouvelleHauteur = yMax - yMin + 1;

            for(int m = 0; m < nouvelleHauteur; m++)
            {
                for(int n = 0; n < nouvelleLargeur; n++)
                {
                    alpha = imgContour.at<double>(xMin + m, yMin + n);
                    if(alpha != -1 && alpha != i)
                    {
                        merge = false;
                        for(unsigned int p = 0; p < contours[alpha].extremites.size(); p++)
                        {
                            if(contours[alpha].extremites[p].x == xMin + m
                                    && contours[alpha].extremites[p].y == yMin + n)
                            {
                                if(abs(orientation.at<double>(xMin + m, yMin + n) - orientation.at<double>(contours[i].extremites[k].x, contours[i].extremites[k].y)) <= 45.0
                                        || abs(orientation.at<double>(xMin + m, yMin + n) - orientation.at<double>(contours[i].extremites[k].x, contours[i].extremites[k].y)) >= 315.0)
                                {
                                    if(test_validite_merge(contours[i].extremites[k].x, contours[i].extremites[k].y,
                                                           xMin + m, yMin + n,
                                                           orientation.at<double>(xMin + m, yMin + n)))
                                    {
                                        merge = true;
                                        merge_contour(imgContour, contours, i, alpha);
                                        compteur++;
                                    }
                                }
                            }
                            if(merge) break;
                        }
                    }
                }
            }
        }
        i++;
    }
    std::cout << "compteur: " << compteur << std::endl;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

void recherche_extremites(cv::Mat& imgContour, std::vector<Contour>& contours)
{
    cv::Point2i px;
    unsigned int i = 0;
    unsigned int nb_extremite = 0;
    int pixPos;
    int nbVoisCont;

    while( i < contours.size() )
    {
        for(unsigned int k = 0; k < contours[i].pix.size(); k++)
        {
            pixPos = 0;
            nbVoisCont = 0;
            px = contours[i].pix[k];
            if (px.x-1 >= 0 && px.y-1 >= 0 && imgContour.at<double>(px.x-1, px.y-1) != -1 )
            {
                if(imgContour.at<double>(px.x-1, px.y-1) == imgContour.at<double>(px.x, px.y))
                {
                    pixPos = 1;
                    nbVoisCont++;
                }
            }
            if (px.x-1 >= 0 && imgContour.at<double>(px.x-1, px.y) != -1 )
            {
                if(imgContour.at<double>(px.x-1, px.y) == imgContour.at<double>(px.x, px.y))
                {
                    if(nbVoisCont == 1 && pixPos != 1)
                        nbVoisCont = 3;
                    pixPos = 2;
                    nbVoisCont++;
                }
            }
            if (px.x-1 >= 0 && px.y+1 < imgContour.cols && imgContour.at<double>(px.x-1, px.y+1) != -1 )
            {
                if(imgContour.at<double>(px.x-1, px.y+1) == imgContour.at<double>(px.x, px.y))
                {
                    if(nbVoisCont == 1 && pixPos != 2)
                        nbVoisCont = 3;
                    pixPos = 3;
                    nbVoisCont++;
                }
            }
            if (px.y+1 < imgContour.cols && imgContour.at<double>(px.x, px.y+1) != -1 )
            {
                if(imgContour.at<double>(px.x, px.y+1) == imgContour.at<double>(px.x, px.y))
                {
                    if(nbVoisCont == 1 && pixPos != 3)
                        nbVoisCont = 3;
                    pixPos = 4;
                    nbVoisCont++;
                }
            }
            if (px.x+1 < imgContour.rows && px.y+1 < imgContour.cols && imgContour.at<double>(px.x+1, px.y+1) != -1 )
            {
                if(imgContour.at<double>(px.x+1, px.y+1) == imgContour.at<double>(px.x, px.y))
                {
                    if(nbVoisCont == 1 && pixPos != 4)
                        nbVoisCont = 3;
                    pixPos = 5;
                    nbVoisCont++;
                }
            }
            if (px.x+1 < imgContour.rows && imgContour.at<double>(px.x+1, px.y) != -1 )
            {
                if(imgContour.at<double>(px.x+1, px.y) == imgContour.at<double>(px.x, px.y))
                {
                    if(nbVoisCont == 1 && pixPos != 5)
                        nbVoisCont = 3;
                    pixPos = 6;
                    nbVoisCont++;
                }
            }
            if (px.x+1 < imgContour.rows && px.y-1 >= 0 && imgContour.at<double>(px.x+1, px.y-1) != -1 )
            {
                if(imgContour.at<double>(px.x+1, px.y-1) == imgContour.at<double>(px.x, px.y))
                {
                    if(nbVoisCont == 1 && pixPos != 6)
                        nbVoisCont = 3;
                    pixPos = 7;
                    nbVoisCont++;
                }
            }
            if (px.y-1 >= 0 && imgContour.at<double>(px.x, px.y-1) != -1 )
            {
                if(imgContour.at<double>(px.x, px.y-1) == imgContour.at<double>(px.x, px.y))
                {
                    if(nbVoisCont == 1 && pixPos != 7 && pixPos != 1)
                        nbVoisCont = 3;
                    pixPos = 8;
                    nbVoisCont++;
                }
            }

            if(nbVoisCont <= 2)
            {
                contours[i].extremites.push_back(px);
                nb_extremite++;
            }
        }
        i++;
    }
    std::cout << "nb_extremité = " << nb_extremite << std::endl;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

void etendre_contour(int i, int j, cv::Mat img_seuille, cv::Mat orientation, cv::Mat& imgContour, std::vector<Contour>& contours)
{
    double seuil = 45.0;
    if(i > 0 && j > 0  && img_seuille.at<double>(i-1, j-1) > 0 && imgContour.at<double>(i-1, j-1) == -1)
    {
        if(abs(orientation.at<double>(i-1, j-1) - orientation.at<double>(i, j)) <= seuil
                || abs(orientation.at<double>(i-1, j-1) - orientation.at<double>(i, j)) >= 360 - seuil)
        {
            //std::cout << orientation.at<double>(i-1, j-1) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i-1, j-1));
            imgContour.at<double>(i-1, j-1) = contours.size() - 1;
            etendre_contour(i -1, j -1, img_seuille, orientation, imgContour, contours);
        }
    }

    if(i > 0 && img_seuille.at<double>(i-1, j) > 0 && imgContour.at<double>(i-1, j) == -1)
    {
        if(abs(orientation.at<double>(i-1, j) - orientation.at<double>(i, j)) <= seuil
                || abs(orientation.at<double>(i-1, j) - orientation.at<double>(i, j)) >= 360 - seuil)
        {
            //std::cout << orientation.at<double>(i-1, j) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i-1, j));
            imgContour.at<double>(i-1, j) = contours.size() - 1;
            etendre_contour(i-1, j, img_seuille, orientation, imgContour, contours);
        }
    }

    if(i > 0 && j < img_seuille.cols  && img_seuille.at<double>(i-1, j+1) > 0 && imgContour.at<double>(i-1, j+1) == -1)
    {
        if(abs(orientation.at<double>(i-1, j+1) - orientation.at<double>(i, j)) <= seuil
                || abs(orientation.at<double>(i-1, j+1) - orientation.at<double>(i, j)) >= 360 - seuil)
        {
            //std::cout << orientation.at<double>(i-1, j+1) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i-1, j+1));
            imgContour.at<double>(i-1, j+1) = contours.size() - 1;
            etendre_contour(i-1, j+1, img_seuille, orientation, imgContour, contours);
        }
    }

    if(j > 0  && img_seuille.at<double>(i, j-1) > 0 && imgContour.at<double>(i, j-1) == -1)
    {
        if(abs(orientation.at<double>(i, j-1) - orientation.at<double>(i, j)) <= seuil
                || abs(orientation.at<double>(i, j-1) - orientation.at<double>(i, j)) >= 360 - seuil)
        {
            //std::cout << orientation.at<double>(i, j-1) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i, j-1));
            imgContour.at<double>(i, j-1) = contours.size() - 1;
            etendre_contour(i, j-1, img_seuille, orientation, imgContour, contours);
        }
    }

    if(j < img_seuille.cols  && img_seuille.at<double>(i, j+1) > 0 && imgContour.at<double>(i, j+1) == -1)
    {
        if(abs(orientation.at<double>(i, j+1) - orientation.at<double>(i, j)) <= seuil
                || abs(orientation.at<double>(i, j+1) - orientation.at<double>(i, j)) >= 360 - seuil)
        {
            //std::cout << orientation.at<double>(i, j+1) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i, j+1));
            imgContour.at<double>(i, j+1) = contours.size() - 1;
            etendre_contour(i, j+1, img_seuille, orientation, imgContour, contours);
        }
    }

    if(i < img_seuille.rows && j > 0  && img_seuille.at<double>(i+1, j-1) > 0 && imgContour.at<double>(i+1, j-1) == -1)
    {
        if(abs(orientation.at<double>(i+1, j-1) - orientation.at<double>(i, j)) <= seuil
                || abs(orientation.at<double>(i+1, j-1) - orientation.at<double>(i, j)) >= 360 - seuil)
        {
            //std::cout << orientation.at<double>(i+1, j-1) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i+1, j-1));
            imgContour.at<double>(i+1, j-1) = contours.size() - 1;
            etendre_contour(i+1, j-1, img_seuille, orientation, imgContour, contours);
        }
    }

    if(i < img_seuille.rows && img_seuille.at<double>(i+1, j) > 0 && imgContour.at<double>(i+1, j) == -1)
    {
        if(abs(orientation.at<double>(i+1, j) - orientation.at<double>(i, j)) <= seuil
                || abs(orientation.at<double>(i+1, j) - orientation.at<double>(i, j)) >= 360 - seuil)
        {
            //std::cout << orientation.at<double>(i+1, j) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i+1, j));
            imgContour.at<double>(i+1, j) = contours.size() - 1;
            etendre_contour(i+1, j, img_seuille, orientation, imgContour, contours);
        }
    }

    if(i < img_seuille.rows && j < img_seuille.cols  && img_seuille.at<double>(i+1, j+1) > 0 && imgContour.at<double>(i+1, j+1) == -1)
    {
        if(abs(orientation.at<double>(i+1, j+1) - orientation.at<double>(i, j)) <= seuil
                || abs(orientation.at<double>(i+1, j+1) - orientation.at<double>(i, j)) >= 360 - seuil)
        {
            //std::cout << orientation.at<double>(i+1, j+1) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i+1, j+1));
            imgContour.at<double>(i+1, j+1) = contours.size() - 1;
            etendre_contour(i+1, j+1, img_seuille, orientation, imgContour, contours);
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

cv::Mat parcours(cv::Mat img_seuille, cv::Mat orientation, std::vector<Contour>& contours)
{
    cv::Mat imgContour(img_seuille.rows, img_seuille.cols, CV_64FC1);
    cv::Mat imgContourAffichable(img_seuille.rows, img_seuille.cols, CV_32FC3);

    //init
    for(int i = 0; i < imgContour.rows; i ++)
        for(int j = 0; j < imgContour.cols; j++)
        {
            imgContour.at<double>(i,j) = -1;
            imgContourAffichable.at<cv::Vec3f>(i,j) = cv::Vec3f(0,0,0);
        }

    for(int i = 0; i < img_seuille.rows; i ++)
        for(int j = 0; j < img_seuille.cols; j++)
            if(img_seuille.at<double>(i,j) > 0 && imgContour.at<double>(i,j) == -1)
            {
                Contour nouveauContour;
                nouveauContour.pix.push_back(cv::Point2i(i,j));
                contours.push_back(nouveauContour);
                imgContour.at<double>(i,j) = contours.size() - 1;
                etendre_contour(i, j, img_seuille, orientation, imgContour, contours);
            }

    recherche_extremites(imgContour, contours);
    amelioration_contours(imgContour, orientation, contours);

    std::cout << "nb de contours = " << contours.size() << std::endl;

    srand(time(NULL));
    cv::Vec3f couleur;

    for(unsigned int i = 0; i < contours.size(); i++)
    {
        couleur[0] = rand()/(float)RAND_MAX;
        couleur[1] = rand()/(float)RAND_MAX;
        couleur[2] = rand()/(float)RAND_MAX;

        for(unsigned int j = 0; j < contours[i].pix.size(); j++)
        {
            int x = contours[i].pix[j].x;
            int y = contours[i].pix[j].y;
            imgContourAffichable.at<cv::Vec3f>(x, y) = couleur;
        }
        for(unsigned int j = 0; j < contours[i].extremites.size(); j++)
        {
            int x = contours[i].extremites[j].x;
            int y = contours[i].extremites[j].y;
            imgContourAffichable.at<cv::Vec3f>(x, y) = cv::Vec3f(1, 1, 1);
        }
    }
    return imgContourAffichable;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------------

cv::Mat affinage(cv::Mat amplitude, cv::Mat orientation)
{
    cv::Mat img_out(amplitude.rows, amplitude.cols, CV_64FC1);
    int ki, kj;

    int angle;

    for(int i = 0; i < amplitude.rows; i++)
        for(int j = 0; j < amplitude.cols; j++)
            img_out.at<double>(i, j) = 0;


    for(int i = 0; i < amplitude.rows; i++)
        for(int j = 0; j < amplitude.cols; j++)
        {
            double pix = amplitude.at<double>(i, j);
            if(pix != 0)
            {
                angle = (int)orientation.at<double>(i, j);
                if(angle >=0 && angle < 45/2) angle = 0;
                else if (angle < 90 - (45 / 2)) angle = 45;
                else if (angle < 135 - (45 / 2)) angle = 90;
                else if (angle < 180 - (45 / 2)) angle = 135;
                else if (angle < 225 - (45 / 2)) angle = 180;
                else if (angle < 270 - (45 / 2)) angle = 225;
                else if (angle < 315 - (45 / 2)) angle = 270;
                else if (angle < 360 - (45 / 2)) angle = 315;
                else angle = 0;

                switch(angle)
                {
                case 0:
                    ki = 1;
                    kj = 0;
                    break;

                case 45:
                    ki = 1;
                    kj = -1;
                    break;

                case 90:
                    ki = 0;
                    kj = -1;
                    break;

                case 135:
                    ki = -1;
                    kj = -1;
                    break;

                case 180:
                    ki = -1;
                    kj = 0;
                    break;

                case 225:
                    ki = -1;
                    kj = 1;
                    break;

                case 270:
                    ki = 0;
                    kj = 1;
                    break;

                case 315:
                    ki = 1;
                    kj = 1;
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
                if((amplitude.at<double>(suivant.x, suivant.y) > max) || (amplitude.at<double>(suivant.x, suivant.y) == max && (pointmax.x > suivant.x || pointmax.y > suivant.y)))
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
                if((amplitude.at<double>(suivant.x, suivant.y) > max) || (amplitude.at<double>(suivant.x, suivant.y) == max && (pointmax.x > suivant.x || pointmax.y > suivant.y)))
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

    int enleve;
    for(int i = 0; i < img_out.rows; i++)
        for(int j = 0; j < img_out.cols; j++)
        {
            if(img_out.at<double>(i,j) != 0)
            {
                enleve = 1;
                if((i > 0 && j > 0 && img_out.at<double>(i-1, j-1) != 0)
                    || (i > 0 && img_out.at<double>(i-1, j) != 0)
                    || (i > 0 && img_out.at<double>(i-1, j) != 0)
                    || (i > 0 && j < img_out.cols && img_out.at<double>(i-1, j+1) != 0)

                    ||(j > 0 && img_out.at<double>(i, j-1) != 0)
                    ||(j < img_out.cols && img_out.at<double>(i, j+1) != 0)

                    ||(i < img_out.rows && j > 0 && img_out.at<double>(i+1, j-1) != 0)
                    ||(i < img_out.rows && img_out.at<double>(i+1, j) != 0)
                    ||(i < img_out.rows && j < img_out.cols && img_out.at<double>(i+1, j+1) != 0))
                    enleve = 0;
                if(enleve == 1)
                    img_out.at<double>(i,j) = 0;
            }
        }
    return img_out;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------------

cv::Mat seuillage(cv::Mat img, int type)
{
    cv::Mat img_out(img.rows, img.cols, CV_64FC1);
    int valCumule = 0;
    double seuil;
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
                valCumule += img.at<double>(i,j);
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
        seuil_b = MAX(0, avg) ;

        //Seuil haut
        for (int i = 0; i < img.rows ; i++)
            for(int j = 0; j < img.cols ; j++)
            {
                if (img.at<double>(i, j) > seuil_h)
                    img_out.at<double>(i, j) = img.at<double>(i, j);
                else img_out.at<double>(i, j) = 0;
            }

        //Seuil bas
        int ki, imin, imax, kj, jmin, jmax;

        for (int i = 0 ; i < img.rows ; i++)
            for (int j = 0 ; j < img.cols ; j++)
            {
                ki = i - 3/2;
                imin = (ki < 0)? 0 : ki;
                ki = i + 3/2;
                imax = (ki > img.rows) ? img.rows : ki;

                kj = j - 3/2;
                jmin = (kj < 0) ? 0 : kj;
                kj = j + 3/2;
                jmax = (kj > img.cols) ? img.cols : kj;

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

//-------------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------------

#define GRADIENT_MAX 0
#define GRADIENT_SUM 1
void calculGradient(cv::Mat& img, cv::Mat& module, cv::Mat& pente, int modeCalculGradient, int nbDirection)
{
    double angle = 360 / (nbDirection * 2);

    std::vector<cv::Mat> usedFilteredImg;

    usedFilteredImg.push_back(applyFilter(img, usedFilter[0]));
    usedFilteredImg.push_back(applyFilter(img, usedFilter[1]));
    usedFilteredImg.push_back(applyFilter(img, usedFilter[2]));
    usedFilteredImg.push_back(applyFilter(img, usedFilter[3]));

    double ksomme, gk, gx, gy;
    int kmax, kmaxTemp, k;

    for(int i = 0 ; i < img.rows ; i++)
    {
        for(int j = 0 ; j < img.cols ; j++)
        {
            ksomme = 0;
            kmax = 0;
            k = 0;
            do
            {
                if(nbDirection == 2 && k == 1)
                    k++;
                gk = usedFilteredImg[k].at<double>(i, j);
                ksomme += abs(gk);
                if(k == 0)
                {
                    kmax = 0;
                    if(nbDirection == 4)
                    {
                        if(gk >= 0)
                            kmaxTemp = kmax;
                        else
                            kmaxTemp = kmax + 4;
                    }
                }
                else if(abs(gk) > abs(usedFilteredImg[kmax].at<double>(i, j)))
                {
                    kmax = k;
                    if(nbDirection == 4)
                    {
                        if(gk >= 0)
                            kmaxTemp = kmax;
                        else
                            kmaxTemp = kmax + 4;
                    }
                }
                k++;
            }while(k < nbDirection);
            if(modeCalculGradient == GRADIENT_SUM)
                module.at<double>(i, j) = ksomme / nbDirection;
            else //gradient max
                module.at<double>(i, j) = abs(usedFilteredImg[kmax].at<double>(i, j));
            kmax = kmaxTemp;
            if(nbDirection == 2)
            {
                gx = usedFilteredImg[0].at<double>(i, j);
                gy = usedFilteredImg[2].at<double>(i, j);
                if(gx != 0)
                {
                    angle = atan(gy / gx);
                    angle *= 180 / M_PI;
                    if(gx < 0)
                    {
                        if(gy >= 0)
                            angle += 180;
                        else
                            angle = 180 + angle;
                    }
                    else
                        if(gy < 0)
                            angle += 360;
                }
                else
                {
                    if(gy >= 0)
                        angle = 90;
                    else
                        angle = 270;
                }
                pente.at<double>(i, j) = angle;
            }
            else
                pente.at<double>(i, j) = kmax * angle;
        }
    }
    usedFilteredImg.clear();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------------------------------------------

cv::Mat norme(cv::Mat img)
{
    cv::Mat img_out(img.rows, img.cols, CV_8UC1);

    // Récupération des valeurs min et max de l'image
    double min = img.at<double>(0, 0);
    double max = img.at<double>(0, 0);
    for (int i = 0; i < img.rows ; i++)
        for(int j = 0; j < img.cols ; j++)
        {
            if (min > img.at<double>(i, j)) min = img.at<double>(i, j);
            if (max < img.at<double>(i, j)) max = img.at<double>(i, j);
        }

    // Normage de l'image pour avoir des valeurs entre [O, 255]
    for (int i = 0; i < img.rows ; i++)
        for(int j = 0; j < img.cols ; j++)
            img_out.at<uchar>(i, j) = round(((img.at<double>(i, j) - min) / ( max - min)) * 255);
    //if(round(((img.at<double>(i, j)-min)/(max-min)) * 255) > 0)std::cout << round(((img.at<double>(i, j)-min)/(max-min)) * 255) << std::endl;

    return img_out;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

std::string demanderImage()
{
    std::vector<std::string> images;

    /*images.push_back("ville.jpg");
    images.push_back("ville2.jpg");
    images.push_back("ville3.jpg");
    images.push_back("villa.png");*/
    images.push_back("formes_droites-cercle.png");
    images.push_back("formes_droites-cercles.png");
    images.push_back("formes.png");
    images.push_back("droite.png");
    images.push_back("cercle.png");
    images.push_back("cercleDroite.png");

    std::cout << "Entrez le numero de l'image a segmenter : " << std::endl;

    for(unsigned int i = 0; i < images.size(); i++) std::cout << i << " - " << images[i] << std::endl;

    while(true)
    {
        std::string tmp;
        std::cin >> tmp;
        unsigned int choix = atof(tmp.c_str());
        if(choix < images.size() && choix >= 0)
        {
            tmp = images[choix];
            tmp = tmp.substr(0,tmp.find(" "));
            std::cout << tmp << std::endl << std::endl;
            images.clear();
            return tmp;
        }
        std::cout << "Le numero d'image choisit n'existe pas. " << tmp << " Re-essayez : " << std::endl;
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------

cv::Mat applyFilter(cv::Mat& img, cv::Mat& filtre)
{
    cv::Mat m2(img.rows, img.cols, CV_64FC1);

    double sum, coeff;
    int ki, imin, imax, kj, jmin, jmax, fi, fj;

    //On parcours toute la matrice de l'image
    for (int i = 0 ; i < img.rows ; i++)
    {
        for (int j = 0 ; j < img.cols ; j++)
        {
            ki = i - filtre.rows/2;
            imin = (ki < 0)? 0 : ki;
            ki = i + filtre.rows/2;
            imax = (ki > img.rows)? img.rows : ki;

            kj = j - filtre.cols/2;
            jmin = (kj < 0)? 0 : kj;
            kj = j + filtre.cols/2;
            jmax = (kj > img.cols)? img.cols : kj;

            sum = 0;

            ki = imin;
            //On applique le filtre à l'image
            while (ki <= imax)
            {
                kj = jmin;
                while (kj <= jmax)
                {
                    fi = ki - i + filtre.rows/2;
                    fj = kj - j + filtre.cols/2;

                    uchar pix = img.at<uchar>(ki, kj);
                    coeff = filtre.at<double>(fi, fj);
                    sum += pix * coeff;

                    kj++;
                }
                ki++;
            }
            m2.at<double>(i, j) = sum;
        }
    }
    return m2;
}
