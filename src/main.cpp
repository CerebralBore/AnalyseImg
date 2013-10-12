#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <vector>
#include <string>

using namespace std;

string demanderImage();

cv::Mat gris(cv::Mat*);
cv::Mat filtreImage(cv::Mat*, cv::Mat*);

int main()
{
    string cheminRepImg, cheminImage;
    cv::Mat imgO, imgGris, imgFlou;
    bool bonChemin;

    double m1[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    cv::Mat flou = cv::Mat(3, 3, CV_64F, m1);
    double m2[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
    cv::Mat prewitt = cv::Mat(3, 3, CV_64F, m2);
    double m3[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    cv::Mat sobeL = cv::Mat(3, 3, CV_64F, m3);
    double m4[3][3] = {{-3, -3, 5}, {-3, 0, 5}, {-3, -3, 5}};
    cv::Mat kirsch = cv::Mat(3, 3, CV_64F, m4);

    char key = '-';
    while (key != 'q' && key != 'Q')
    {
        cheminRepImg = "images/";
        bonChemin = false;

        while (!bonChemin)
        {
            cheminImage = cheminRepImg;
            cheminImage += demanderImage();

            imgO = cv::imread (cheminImage);

            if (imgO.empty ())
                cerr << "Mauvais chemin d'image: " << cheminImage << endl;
            else
                bonChemin = true;
        }

        imgGris = gris(&imgO);
        imgFlou = filtreImage(&imgO, &flou);

        cv::imshow("Image original", imgO);
        cvWaitKey();
        cv::imshow("Image niveau de gris", imgGris);
        cvWaitKey();
        cv::imshow("Image flou", imgFlou);

        cout << "Cliquez sur une fenetre d'OpenCv puis (q) pour pour quitter, (s) pour segmenter une nouvelle image." << endl;
        key = '-';
        while(key != 'q' && key != 's' && key != 'Q' && key != 'S') key = cvWaitKey(50);

        cvDestroyAllWindows();

        cout << endl << "-----------------------------------------------------------------" << endl << endl;
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
        cout << "Le numero d'image choisit n'existe pas. " << tmp << " Reeseyez : " << endl;
    }

}

cv::Vec3b filtreMatrice(cv::Mat* matImage, cv::Mat* matFiltre)
{
    double coef = 0;

    double totalR = 0;
    double totalV = 0;
    double totalB = 0;

    cv::Vec3b pix;

    //parcourt les matrices 3x3 des deux images
    for(int i=0; i<matImage->rows; i++)
    {
        for(int j=0; j<matImage->cols; j++)
        {
            if(matFiltre->at<double>(i,j) > 0)
                coef = coef + matFiltre->at<double>(i,j);
            totalR = totalR + matImage->at<cv::Vec3b>(i,j)[2] * matFiltre->at<double>(i,j);
            totalV = totalV + matImage->at<cv::Vec3b>(i,j)[1] * matFiltre->at<double>(i,j);
            totalB = totalB + matImage->at<cv::Vec3b>(i,j)[0] * matFiltre->at<double>(i,j);
        }
    }

    if(coef !=0)
    {
        pix[2] = totalR/coef;
        pix[1] = totalV/coef;
        pix[0] = totalB/coef;
    }
    else
    {
        pix[2] = totalR;
        pix[1] = totalV;
        pix[0] = totalB;
    }
    //renvoie la couleur du pixel central
    return(pix);
}

cv::Mat filtreImage(cv::Mat* imgSource, cv::Mat* filtre)
{
    cv::Range lignes;
    cv::Range colonnes;

    cv::Mat imgFiltre = imgSource->clone();
    cv::Mat matImage;

    for(int i=1; i<(imgSource->rows-1); i++)
    {
        for(int j=1; j<(imgSource->cols-1); j++)
        {
            //récuprération de la matrice 3x3 autour du point de l'image courrante
            lignes = cv::Range((i-1),(i+2));
            colonnes = cv::Range((j-1),(j+2));
            matImage = cv::Mat(imgFiltre, lignes, colonnes);
            imgFiltre.at<cv::Vec3b>(i,j) = filtreMatrice(&matImage, filtre);
        }
    }
    return imgFiltre;
}

cv::Mat gris(cv::Mat* imgSource)
{
    cv::Mat imgFiltre = imgSource->clone();
    int rouge, bleu, vert;

    for(int i=0; i<(imgSource->rows); i++)
    {
        for(int j=0; j<(imgSource->cols); j++)
        {
            //récupération des couleurs
            rouge = imgSource->at<cv::Vec3b>(i,j)[2];
            bleu = imgSource->at<cv::Vec3b>(i,j)[0];
            vert = imgSource->at<cv::Vec3b>(i,j)[1];
            for(int k=0; k<3; k++){
                imgFiltre.at<cv::Vec3b>(i,j)[k] = 0.2125*rouge + 0.0721*bleu + 0.7154*vert ;
            }
        }
    }
    return imgFiltre;
}
