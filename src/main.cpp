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
bool test_validite_merge(int a_x, int a_y, int b_x, int b_y, double angle_a, double angle_b);

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
    std::vector<Contour> contours;

    bool bonChemin;

    int filterType = 0;
    int gradientType = 0;
    int directionType = 0;
    int seuillageType = 0;

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

        equalizeHist(img, img);
        cv::imshow("Image original", img);

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

        moduleAffinage = affinage (moduleSeuille, pente);
        cv::imshow("module Affine", moduleAffinage);

        imgContours = parcours(moduleAffinage, pente, contours);
        cv::imshow("contours", imgContours);

        /*
        for(int i = 0; i < imgContours.rows; i++)
            for(int j = 0; j < imgContours.cols; j++)
                std::cout << imgContours.at<double>(i,j) << " ";
        */

        std::cout << "Cliquez sur une fenetre d'OpenCv puis (q) pour pour quitter, (s) pour segmenter une nouvelle image." << std::endl;

        contours.clear();

        key = '-';
        while(key != 'q' && key != 's' && key != 'Q' && key != 'S') key = cvWaitKey(50);

        cvDestroyAllWindows();

        std::cout << std::endl << "-----------------------------------------------------------------" << std::endl << std::endl;
    }
}

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
    }
    return imgContourAffichable;
}

void etendre_contour(int i, int j, cv::Mat img_seuille, cv::Mat orientation, cv::Mat& imgContour, std::vector<Contour>& contours)
{
    double seuil = 45.0;
    if(i > 0 && j > 0  && img_seuille.at<double>(i-1, j-1) > 0 && imgContour.at<double>(i-1, j-1) == -1)
    {
        if(abs(orientation.at<double>(i-1, j-1) - orientation.at<double>(i, j) <= seuil))
        {
            //std::cout << orientation.at<double>(i-1, j-1) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i-1, j-1));
            imgContour.at<double>(i-1, j-1) = contours.size() - 1;
            etendre_contour(i -1, j -1, img_seuille, orientation, imgContour, contours);
        }
    }

    if(i > 0 && img_seuille.at<double>(i-1, j) > 0 && imgContour.at<double>(i-1, j) == -1)
    {
        if(abs(orientation.at<double>(i-1, j) - orientation.at<double>(i, j) <= seuil))
        {
            //std::cout << orientation.at<double>(i-1, j) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i-1, j));
            imgContour.at<double>(i-1, j) = contours.size() - 1;
            etendre_contour(i-1, j, img_seuille, orientation, imgContour, contours);
        }
    }

    if(i > 0 && j < img_seuille.cols  && img_seuille.at<double>(i-1, j+1) > 0 && imgContour.at<double>(i-1, j+1) == -1)
    {
        if(abs(orientation.at<double>(i-1, j+1) - orientation.at<double>(i, j) <= seuil))
        {
            //std::cout << orientation.at<double>(i-1, j+1) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i-1, j+1));
            imgContour.at<double>(i-1, j+1) = contours.size() - 1;
            etendre_contour(i-1, j+1, img_seuille, orientation, imgContour, contours);
        }
    }

    if(j > 0  && img_seuille.at<double>(i, j-1) > 0 && imgContour.at<double>(i, j-1) == -1)
    {
        if(abs(orientation.at<double>(i, j-1) - orientation.at<double>(i, j) <= seuil))
        {
            //std::cout << orientation.at<double>(i, j-1) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i, j-1));
            imgContour.at<double>(i, j-1) = contours.size() - 1;
            etendre_contour(i, j-1, img_seuille, orientation, imgContour, contours);
        }
    }

    if(j < img_seuille.cols  && img_seuille.at<double>(i, j+1) > 0 && imgContour.at<double>(i, j+1) == -1)
    {
        if(abs(orientation.at<double>(i, j+1) - orientation.at<double>(i, j) <= seuil))
        {
            //std::cout << orientation.at<double>(i, j+1) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i, j+1));
            imgContour.at<double>(i, j+1) = contours.size() - 1;
            etendre_contour(i, j+1, img_seuille, orientation, imgContour, contours);
        }
    }

    if(i < img_seuille.rows && j > 0  && img_seuille.at<double>(i+1, j-1) > 0 && imgContour.at<double>(i+1, j-1) == -1)
    {
        if(abs(orientation.at<double>(i+1, j-1) - orientation.at<double>(i, j) <= seuil))
        {
            //std::cout << orientation.at<double>(i+1, j-1) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i+1, j-1));
            imgContour.at<double>(i+1, j-1) = contours.size() - 1;
            etendre_contour(i+1, j-1, img_seuille, orientation, imgContour, contours);
        }
    }

    if(i < img_seuille.rows && img_seuille.at<double>(i+1, j) > 0 && imgContour.at<double>(i+1, j) == -1)
    {
        if(abs(orientation.at<double>(i+1, j) - orientation.at<double>(i, j) <= seuil))
        {
            //std::cout << orientation.at<double>(i+1, j) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i+1, j));
            imgContour.at<double>(i+1, j) = contours.size() - 1;
            etendre_contour(i+1, j, img_seuille, orientation, imgContour, contours);
        }
    }

    if(i < img_seuille.rows && j < img_seuille.cols  && img_seuille.at<double>(i+1, j+1) > 0 && imgContour.at<double>(i+1, j+1) == -1)
    {
        if(abs(orientation.at<double>(i+1, j+1) - orientation.at<double>(i, j) <= seuil))
        {
            //std::cout << orientation.at<double>(i+1, j+1) - orientation.at<double>(i, j) << " ";
            contours[contours.size()-1].pix.push_back(cv::Point2i(i+1, j+1));
            imgContour.at<double>(i+1, j+1) = contours.size() - 1;
            etendre_contour(i+1, j+1, img_seuille, orientation, imgContour, contours);
        }
    }
}

void recherche_extremites(cv::Mat& imgContour, std::vector<Contour>& contours)
{
    unsigned int i = 0;
    unsigned int nb_extremite = 0;

    while( i < contours.size() )
    {
        for(unsigned int k = 0; k < contours[i].pix.size(); k++)
        {
            cv::Point2i px = contours[i].pix[k];
            int nb_voisin = 0;
            if (px.x-1 > 0 && px.y-1 > 0 && imgContour.at<double>(px.x-1, px.y-1) != -1 ) nb_voisin++;
            if (px.x-1 > 0 && imgContour.at<double>(px.x-1, px.y) != -1 ) nb_voisin++;
            if (px.x-1 > 0 && px.y+1 < imgContour.cols && imgContour.at<double>(px.x-1, px.y+1) != -1 ) nb_voisin++;
            if (px.y-1 > 0 && imgContour.at<double>(px.x, px.y-1) != -1 ) nb_voisin++;
            if (px.y+1 < imgContour.cols && imgContour.at<double>(px.x, px.y+1) != -1 ) nb_voisin++;
            if (px.x+1 < imgContour.rows && px.y-1 > 0 && imgContour.at<double>(px.x+1, px.y-1) != -1 ) nb_voisin++;
            if (px.x+1 < imgContour.rows && imgContour.at<double>(px.x+1, px.y) != -1 ) nb_voisin++;
            if (px.x+1 < imgContour.rows && px.y+1 < imgContour.cols && imgContour.at<double>(px.x+1, px.y+1) != -1 ) nb_voisin++;

            if(nb_voisin < 2)
            {
                contours[i].extremites.push_back(px);
                nb_extremite++;
            }
        }
        i++;
    }
    std::cout << "nb_extremité = " << nb_extremite << std::endl;
}

void amelioration_contours(cv::Mat &imgContour, cv::Mat& orientation, std::vector<Contour> &contours)
{
    int i = 0;
    int compteur = 0;

    int largeur = 9;
    int hauteur = 9;

    while (i < contours.size())
    {
        for(unsigned int k = 0; k < contours[i].pix.size(); k++)
        {
            int xMin = contours[i].pix[k].x < (largeur / 2) ? 0 : contours[i].pix[k].x - (largeur / 2);
            int yMin = contours[i].pix[k].y < (hauteur / 2) ? 0 : contours[i].pix[k].y - (hauteur / 2);
            int xMax = (contours[i].pix[k].x + (largeur / 2) >= imgContour.cols) ? imgContour.cols - 1 : contours[i].pix[k].x + (largeur / 2);
            int yMax = (contours[i].pix[k].y + (hauteur / 2) >= imgContour.rows) ? imgContour.rows - 1 : contours[i].pix[k].y + (hauteur / 2);

            int nouvelleLargeur = xMax - xMin + 1;
            int nouvelleHauteur = yMax - yMin + 1;

            for(int m = 0; m < nouvelleHauteur; m++)
            {
                for(int n = 0; n < nouvelleLargeur; n++)
                {
                    int alpha = imgContour.at<double>(xMin + m, yMin + n);
                    if(alpha != -1 && alpha != i)
                    {
                        bool merge = false;
                        for(unsigned int p = 0; p < contours[alpha].extremites.size(); p++)
                        {
                            if(contours[alpha].extremites[p].x == xMin + m
                                    && contours[alpha].extremites[p].y == yMin + n)
                            {
                                if(orientation.at<double>(xMin + m, yMin + n) - orientation.at<double>(contours[i].pix[k].x, contours[i].pix[k].y) <= 45.0
                                        && orientation.at<double>(xMin + m, yMin + n) - orientation.at<double>(contours[i].pix[k].x, contours[i].pix[k].y) >= 0.0)
                                {
                                    if(test_validite_merge(contours[i].pix[k].x, contours[i].pix[k].y,
                                                           contours[alpha].extremites[p].x, contours[alpha].extremites[p].y,
                                                           orientation.at<double>(contours[i].pix[k].x, contours[i].pix[k].y), orientation.at<double>(xMin + m, yMin + n)))
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

double distance(int a_x, int a_y, int b_x, int b_y)
{
    return(sqrt(pow(b_x - a_x,2) + pow(b_y - a_y, 2)));
}

bool test_validite_merge(int a_x, int a_y, int b_x, int b_y, double angle_a, double angle_b)
{
    double d1 = distance(a_x, a_y, b_x, b_y);
    double d2 = distance(a_x, a_y, a_x + 1, a_y);
    double d3 = distance(a_x + 1, a_y, b_x, b_y);

    double angle_diff = acos((pow(d1,2) + pow(d2,2) - pow(d3,2)) / ( 2 * d1 * d2)) * 180 / M_PI;

    if(angle_diff < (angle_a + 90 + (angle_b - angle_a) + 20.0)
            && angle_diff > (angle_a + 90 + (angle_b - angle_a) - 20))
        return true;
    //std::cout << angle_diff << " " << (angle_a + 90.0 + (angle_b - angle_a)) << " " << (angle_a + 90.0 - (angle_b - angle_a)) << std::endl;
    return false;
}

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

    /*for(int i = 0; i < img_out.rows; i++)
        for(int j = 0; j < img_out.cols; j++)
        {
            if( (img_out.at<double>(i,j) != 0)
                    && (i > 0 && j > 0 && img_out.at<double>(i-1, j-1) == 0)
                    && (i > 0 && img_out.at<double>(i-1, j) == 0)
                    && (i > 0 && j < img_out.cols && img_out.at<double>(i-1, j+1) == 0)

                    && (j > 0 && img_out.at<double>(i, j-1) == 0)
                    && (j < img_out.cols && amplitude.at<double>(i, j+1) == 0)

                    && (i < img_out.rows && j > 0 && img_out.at<double>(i+1, j-1) == 0)
                    && (i < img_out.rows && img_out.at<double>(i+1, j) == 0)
                    && (i < img_out.rows && j < img_out.cols && img_out.at<double>(i+1, j+1) == 0))
                img_out.at<double>(i,j) = 0;
        }*/
    return img_out;
}

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
    int kmax, kmaxTemp;

    for(int i = 0 ; i < img.rows ; i++)
    {
        for(int j = 0 ; j < img.cols ; j++)
        {
            ksomme = 0;
            kmax = 0;
            for(int k = 0 ; k < nbDirection ; k++)
            {
                if(nbDirection == 2)
                    k *= 2;
                gk = usedFilteredImg[k].at<double>(i, j);
                ksomme += abs(gk);
                if(nbDirection == 4 && abs(gk) > abs(usedFilteredImg[kmax].at<double>(i, j)))
                {
                    kmax = k;
                    if(gk >= 0)
                        kmaxTemp = kmax;
                    else
                        kmaxTemp = kmax + 4;
                }
            }
            kmax = kmaxTemp;
            if(modeCalculGradient == GRADIENT_SUM)
                module.at<double>(i, j) = ksomme / nbDirection;
            else //gradient max
            {
                if(kmax >= 4)
                    module.at<double>(i, j) = abs(usedFilteredImg[kmax - 4].at<double>(i, j));
                else
                    module.at<double>(i, j) = abs(usedFilteredImg[kmax].at<double>(i, j));
            }

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
}

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

std::string demanderImage()
{
    std::vector<std::string> images;

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
    images.push_back("feuille_collee_carton.png");

    images.push_back("MIF23/B.JPG ");
    images.push_back("MIF23/B1.JPG ");
    images.push_back("MIF23/balls0.jpg ");

    images.push_back("MIF23/bally_0.jpg ");
    images.push_back("MIF23/bin.jpg ");
    images.push_back("MIF23/cake_0.JPG ");

    images.push_back("MIF23/purple_2_bw.jpg ");
    images.push_back("MIF23/rose_2_bw.JPG ");

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
            return tmp;
        }
        std::cout << "Le numero d'image choisit n'existe pas. " << tmp << " Re-essayez : " << std::endl;
    }
}

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
